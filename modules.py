# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from vector_quantize_pytorch import ResidualVQ
from transformers import WavLMModel

def get_padding(k: int, d: int = 1) -> int:
    return (k * d - d) // 2

def get_2d_padding(kernel_size: tuple, dilation: tuple = (1, 1)) -> tuple:
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )

def norm_conv2d(
    in_ch: int,
    out_ch: int,
    kernel_size: tuple,
    stride: tuple = (1, 1),
    dilation: tuple = (1, 1),
    padding: tuple = (0, 0),
    norm: str = "weight_norm",
) -> nn.Module:
    conv = nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
    )
    if norm == "weight_norm":
        return nn.utils.weight_norm(conv)
    if norm == "spectral_norm":
        return nn.utils.spectral_norm(conv)
    return conv

LRELU_SLOPE = 0.1

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel: int = None, transpose: bool = False):
        super().__init__()
        # Residual unit: two k=3 convolutions + skip connection
        self.res_conv1 = nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.res_conv2 = nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.res_norm1 = nn.LayerNorm(in_ch)
        self.res_norm2 = nn.LayerNorm(in_ch)
        self.act = nn.GELU()

        # Sampling layer: k = 2 * stride (paper setting)
        if kernel is None:
            kernel = 2 * stride

        if not transpose:
            padding = (kernel - stride) // 2
            self.sample = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        else:
            # For odd stride, use (padding, output_padding) to keep exact scale-up by stride
            padding = (stride + 1) // 2
            output_padding = stride % 2
            self.sample = nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size=kernel, stride=stride,
                padding=padding, output_padding=output_padding
            )
        self.sample_norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        r = self.res_conv1(x)
        r = self.act(self.res_norm1(r.transpose(1, 2)).transpose(1, 2))
        r = self.res_conv2(r)
        r = self.res_norm2(r.transpose(1, 2)).transpose(1, 2)
        x = self.act(r + x)

        x = self.sample(x)
        x = self.act(self.sample_norm(x.transpose(1, 2)).transpose(1, 2))
        return x

class SpeechEncoder(nn.Module):
    def __init__(self, strides: list = [2, 4, 5, 8], hidden: int = 512, lstm_layers: int = 2):
        super().__init__()
        # 论文描述：Speech Encoder 由 4 个步长卷积层组成，后接 2 层双向 LSTM
        # 通道数随深度倍增
        ch = [64, 128, 256, hidden]
        self.convs = nn.ModuleList([
            ConvBlock(1 if i == 0 else ch[i-1], c, stride=s) 
            for i, (c, s) in enumerate(zip(ch, strides))
        ])
        self.lstm = nn.LSTM(hidden, hidden, lstm_layers, batch_first=True, bidirectional=True)
        # 论文描述为末端 1D Conv(kernel=7, out_channels=512)；padding=3 以保持时序长度
        self.proj = nn.Conv1d(hidden * 2, hidden, kernel_size=7, padding=3) 
    
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T]
        x = wav
        for c in self.convs: 
            x = c(x)
        # x: [B, hidden, T_feat]
        x, _ = self.lstm(x.transpose(1, 2))  # [B, T_feat, hidden*2]
        return self.proj(x.transpose(1, 2)).transpose(1, 2)  # [B, T_feat, hidden]

class Conv1dGLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch * 2, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B, T, 2C]
        h = F.glu(h, dim=-1)  # [B, T, out_ch]
        h = self.dropout(h)
        return h + x

class SpeakerEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        in_dim = cfg['n_mels']
        hidden = cfg.get('style_hidden', 128)
        out_dim = cfg['dim']
        kernel = cfg.get('style_tcn_kernel', 5)
        num_heads = cfg.get('style_head', 2)
        dropout = cfg.get('dropout', 0.1)

        # 1) Spectral processing (Linear + Mish + Dropout) x2
        self.spectral = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Dropout(dropout),
        )

        # 2) Temporal processing (2 x Conv1dGLU with residual)
        self.temporal = nn.Sequential(
            Conv1dGLU(hidden, hidden, kernel, dropout),
            Conv1dGLU(hidden, hidden, kernel, dropout),
        )

        # 3) Frame-level multi-head self-attention + frame-wise FC
        self.slf_attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, out_dim)

    def temporal_avg_pool(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return torch.mean(x, dim=1)
        lengths = (~mask).sum(dim=1).unsqueeze(1).clamp_min(1)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x.sum(dim=1) / lengths

    def forward(self, mel: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Compatible input:
        # [B, 1, F, T] -> [B, T, F], or already [B, T, F]
        if mel.dim() == 4:
            x = mel.squeeze(1).transpose(1, 2)
        elif mel.dim() == 3:
            x = mel
        else:
            raise ValueError(f"Unexpected mel shape: {tuple(mel.shape)}")

        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None

        # spectral
        x = self.spectral(x)
        # temporal
        x = self.temporal(x)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        # nn.MultiheadAttention uses key_padding_mask; slf_attn_mask kept for parity/readability
        _ = slf_attn_mask
        residual = x
        x, _ = self.slf_attn(x, x, x, key_padding_mask=mask)
        x = self.attn_dropout(x) + residual
        # fc
        x = self.fc(x)
        # temporal average pooling
        return self.temporal_avg_pool(x, mask=mask)

class ResidualBottleneck(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        bc = cfg['model']['bottleneck']
        self.proj_in = nn.Linear(cfg['model']['hidden_dim'], bc['codebook_dim'])
        self.rvq = ResidualVQ(
            dim=bc['codebook_dim'], num_quantizers=bc['num_quantizers'],
            codebook_size=bc['codebook_size'], codebook_dim=bc['codebook_dim'],
            decay=bc['decay'], commitment_weight=bc['commitment_weight'],
            use_cosine_sim=bc['use_cosine_sim'], kmeans_init=bc['kmeans_init'],
            threshold_ema_dead_code=bc['threshold_ema_dead_code']
        )
        self.proj_out = nn.Linear(bc['codebook_dim'], cfg['model']['hidden_dim'])
        self.q1_layer = self.rvq.layers[0]
        self.q2_layer = self.rvq.layers[1]
    
    def forward(self, x: torch.Tensor):
        # x: [B, T_feat, hidden] → proj_in: [B, T_feat, codebook_dim]
        h = self.proj_in(x)  # [B, T_feat, codebook_dim]
        quantized, _, commit_loss = self.rvq(h)  # quantized:[B, T_feat, codebook_dim]
        com = commit_loss.mean() if commit_loss.dim() > 0 else commit_loss # scalar
        
        # 显式提取 q1, q2 用于蒸馏
        _, idx1, _ = self.q1_layer(h)  # idx1:[B, T_feat]
        q1 = self.q1_layer.codebook[idx1.long()]  # [B, T_feat, codebook_dim]
        
        _, idx2, _ = self.q2_layer(h - q1.detach())  # idx2:[B, T_feat]
        q2 = self.q2_layer.codebook[idx2.long()]  # [B, T_feat, codebook_dim]
        
        return self.proj_out(quantized), q1, q2, com  # out:[B, T_feat, hidden], q1/q2:[B, T_feat, codebook_dim]

class Decoder(nn.Module):
    def __init__(self, strides: list = [2, 4, 5, 8], hidden: int = 512, lstm_layers: int = 2):
        super().__init__()
        # 论文描述：Decoder 结构镜像 Encoder
        # Encoder 顺序：ConvBlocks -> LSTM -> Proj
        # Decoder 顺序：Proj_inv -> LSTM -> ConvBlocks_inv
        
        # 1. 镜像 Encoder 的 Proj 层 (kernel=7)
        self.proj_in = nn.Conv1d(hidden, hidden * 2, kernel_size=7, padding=3)
        
        # 2. 镜像 Encoder 的 LSTM 层 (hidden * 2 -> hidden)
        # 因为是双向，所以每向维度是 hidden // 2
        self.lstm = nn.LSTM(hidden * 2, hidden // 2, lstm_layers, batch_first=True, bidirectional=True)
        
        # 3. 镜像 Encoder 的 ConvBlocks 层 (hidden -> 256 -> 128 -> 64 -> 1)
        t_strides = list(reversed(strides))
        ch = [hidden, 256, 128, 64, 1]
        self.blocks = nn.ModuleList([
            ConvBlock(ch[i], ch[i+1], stride=s, transpose=True)
            for i, s in enumerate(t_strides)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_feat, hidden]
        x = self.proj_in(x.transpose(1, 2)).transpose(1, 2) # [B, T_feat, hidden*2]
        x, _ = self.lstm(x) # [B, T_feat, hidden]
        x = x.transpose(1, 2) # [B, hidden, T_feat]
        for b in self.blocks:
            x = b(x)
        return x.squeeze(1)  # [B, T]

class WavLMExtractor(nn.Module):
    def __init__(self, name: str = "microsoft/wavlm-large", layer: int = 12):
        super().__init__()
        self.model = WavLMModel.from_pretrained(name)
        self.layer = layer
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
    
    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T] → squeeze: [B, T] → WavLM → hidden_states[layer]: [B, T_wavlm, D_wavlm]
        out = self.model(wav.squeeze(1), output_hidden_states=True)
        return out.hidden_states[self.layer]  # [B, T_wavlm, D_wavlm]

class DiscriminatorSTFT(nn.Module):
    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tuple = (3, 9),
        dilations: list = None,
        stride: tuple = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation_slope: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.activation = nn.LeakyReLU(negative_slope=activation_slope)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=normalized,
            center=False,
            pad_mode=None,
            power=None,
        )

        spec_channels = 2 * in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            norm_conv2d(
                spec_channels,
                filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
                norm=norm,
            )
        )

        in_chs = min(filters_scale * filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * filters, max_filters)
            self.convs.append(
                norm_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs

        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        self.convs.append(
            norm_conv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = norm_conv2d(
            out_chs,
            out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 1, T] -> complex spectrogram -> concat(real, imag) -> conv2d stack
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.spec_transform(x)
        z = torch.cat([z.real, z.imag], dim=1)
        z = z.transpose(-2, -1)
        fmap = []
        for layer in self.convs:
            z = self.activation(layer(z))
            fmap.append(z)
        z = self.conv_post(z)
        fmap.append(z)
        return z, fmap

class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: list = None,
        hop_lengths: list = None,
        win_lengths: list = None,
    ):
        super().__init__()
        if n_ffts is None:
            n_ffts = [2048, 1024, 512, 256, 128]
        if hop_lengths is None:
            hop_lengths = [512, 256, 128, 64, 32]
        if win_lengths is None:
            win_lengths = [2048, 1024, 512, 256, 128]
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)

        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(
                filters=filters,
                in_channels=in_channels,
                out_channels=out_channels,
                n_fft=n_ffts[i],
                hop_length=hop_lengths[i],
                win_length=win_lengths[i],
            )
            for i in range(len(n_ffts))
        ])

    def forward(self, x: torch.Tensor):
        logits, fmaps = [], []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps

class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel: int = 5, stride: int = 3, use_sn: bool = False):
        super().__init__()
        self.period = period
        nf = nn.utils.spectral_norm if use_sn else nn.utils.weight_norm
        ch = [32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList([
            nf(nn.Conv2d(1, ch[0], (kernel, 1), (stride, 1), padding=(get_padding(kernel), 0))),
            nf(nn.Conv2d(ch[0], ch[1], (kernel, 1), (stride, 1), padding=(get_padding(kernel), 0))),
            nf(nn.Conv2d(ch[1], ch[2], (kernel, 1), (stride, 1), padding=(get_padding(kernel), 0))),
            nf(nn.Conv2d(ch[2], ch[3], (kernel, 1), (stride, 1), padding=(get_padding(kernel), 0))),
            nf(nn.Conv2d(ch[3], ch[4], (kernel, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = nf(nn.Conv2d(ch[4], 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))
    
    def forward(self, x: torch.Tensor):
        # x: [B, 1, T] → reshape: [B, 1, T//period, period] → conv2d → flatten
        fmap, b, c, t = [], *x.shape
        if t % self.period != 0: x = F.pad(x, (0, self.period - t % self.period), "reflect")
        x = x.view(b, c, -1, self.period)  # [B, 1, T//period, period]
        for l in self.convs:
            x = l(x); x = F.leaky_relu(x, LRELU_SLOPE); fmap.append(x)  # x:[B, C_i, T_i, period]
        x = self.conv_post(x); fmap.append(x)  # [B, 1, T_out, period]
        return torch.flatten(x, 1, -1), fmap  # score:[B, *], fmap:List of [B, C, T, period]

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(nn.Module):
    def __init__(self, use_sn: bool = False):
        super().__init__()
        nf = nn.utils.spectral_norm if use_sn else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            nf(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)),
            nf(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4, padding=20)),
            nf(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16, padding=20)),
            nf(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20)),
            nf(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20)),
            nf(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, groups=16, padding=20)),
            nf(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
        ])
        self.conv_post = nf(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x: torch.Tensor):
        # x: [B, 1, T] → conv1d chain → flatten
        fmap = []
        for l in self.convs: x = l(x); x = F.leaky_relu(x, LRELU_SLOPE); fmap.append(x)  # x:[B, C_i, T_i]
        x = self.conv_post(x); fmap.append(x)  # [B, 1, T_out]
        return torch.flatten(x, 1, -1), fmap  # score:[B, *], fmap:List of [B, C, T]

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_sn=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.mstftd = MultiScaleSTFTDiscriminator(
            filters=32,
            n_ffts=[2048, 1024, 512, 256, 128],
            hop_lengths=[512, 256, 128, 64, 32],
            win_lengths=[2048, 1024, 512, 256, 128],
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        # y/y_hat: [B, 1, T]
        if y.dim() == 2:
            y = y.unsqueeze(1)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)

        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        y_mpd_r, y_mpd_g, fmap_mpd_r, fmap_mpd_g = self.mpd(y, y_hat)
        y_d_rs.extend(y_mpd_r)
        y_d_gs.extend(y_mpd_g)
        fmap_rs.extend(fmap_mpd_r)
        fmap_gs.extend(fmap_mpd_g)

        y_msd_r, y_msd_g, fmap_msd_r, fmap_msd_g = self.msd(y, y_hat)
        y_d_rs.extend(y_msd_r)
        y_d_gs.extend(y_msd_g)
        fmap_rs.extend(fmap_msd_r)
        fmap_gs.extend(fmap_msd_g)

        y_stft_r, fmap_stft_r = self.mstftd(y)
        y_stft_g, fmap_stft_g = self.mstftd(y_hat)
        y_d_rs.extend(y_stft_r)
        y_d_gs.extend(y_stft_g)
        fmap_rs.extend(fmap_stft_r)
        fmap_gs.extend(fmap_stft_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
