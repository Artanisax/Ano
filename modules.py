# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel
from vector_quantize_pytorch import ResidualVQ
from discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator

LRELU_SLOPE = 0.1

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel: int = None, transpose: bool = False):
        super().__init__()
        self.res_conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        )
        self.res_conv2 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        )
        self.act = nn.LeakyReLU(LRELU_SLOPE)
        auto_kernel = kernel is None
        if kernel is None:
            kernel = 2 * stride + (stride % 2)
        padding = (kernel - stride) // 2
        if not transpose:
            self.sample = nn.utils.weight_norm(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, padding_mode='reflect')
            )
        else:
            output_padding = 0 if auto_kernel else (stride % 2)
            self.sample = nn.utils.weight_norm(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding))
        self.apply_sample_post = not (transpose and out_ch == 1)
        if transpose:
            self.sample_norm = nn.Identity()
            self.sample_act = nn.LeakyReLU(LRELU_SLOPE)
        else:
            groups = max(1, out_ch // 16)
            self.sample_norm = nn.GroupNorm(groups, out_ch, eps=1e-6, affine=True)
            self.sample_act = nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.res_conv1(self.act(x))
        r = self.res_conv2(self.act(r))
        x = r + x
        x = self.sample(x)
        if self.apply_sample_post:
            x = self.sample_norm(x)
            x = self.sample_act(x)
        return x

class SpeechEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        enc_cfg = cfg.get('speechencoder', {})
        hidden = cfg['dimension']
        strides = list(reversed(enc_cfg.get('strides', [8, 5, 4, 2])))
        lstm_layers = enc_cfg.get('lstm_layers', 2)
        ch = [64, 128, 256, hidden]
        self.convs = nn.ModuleList([ConvBlock(1 if i == 0 else ch[i - 1], c, stride=s) for i, (c, s) in enumerate(zip(ch, strides))])
        self.lstm = nn.LSTM(hidden, hidden, lstm_layers, batch_first=True, bidirectional=True)
        self.proj = nn.utils.weight_norm(
            nn.Conv1d(hidden * 2, hidden, kernel_size=7, padding=3, padding_mode='reflect')
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav
        for c in self.convs:
            x = c(x)
        x, _ = self.lstm(x.transpose(1, 2))
        return self.proj(x.transpose(1, 2)).transpose(1, 2)

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
        spk_cfg = cfg['speaker']
        in_dim = cfg['n_mels']
        hidden = spk_cfg.get('style_hidden', 128)
        out_dim = cfg['dimension']
        kernel = spk_cfg.get('style_tcn_kernel', 5)
        num_heads = spk_cfg.get('style_head', 2)
        dropout = spk_cfg.get('dropout', 0.1)

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
        hidden_dim = cfg['model']['dimension']
        self.num_quantizers = bc.get('n_q', bc.get('num_quantizers', 8))
        self.tap_layers = (0, 1)
        self.rvq = ResidualVQ(
            dim=hidden_dim,
            codebook_size=bc['codebook_size'],
            num_quantizers=self.num_quantizers,
            decay=bc['decay'],
            kmeans_init=bc['kmeans_init'],
            threshold_ema_dead_code=int(bc['threshold_ema_dead_code']),
            commitment_weight=bc['commitment_weight'],
        )

    def forward(self, x: torch.Tensor):
        h = x.transpose(1, 2)  # [B, T, D]
        residual = h
        quantized_out = torch.zeros_like(h)
        tapped_quantized = {}
        commit_losses = []

        for i, layer in enumerate(self.rvq.layers[:self.num_quantizers]):
            quantized_i, _, commit_i = layer(residual)
            residual = residual - quantized_i
            quantized_out = quantized_out + quantized_i
            commit_losses.append(commit_i.reshape(-1).mean())
            if i in self.tap_layers:
                tapped_quantized[i] = quantized_i

        com = torch.stack(commit_losses).mean() if commit_losses else torch.tensor(0.0, device=x.device)
        q1 = tapped_quantized.get(0, torch.zeros_like(h)).transpose(1, 2)
        q2 = tapped_quantized.get(1, torch.zeros_like(h)).transpose(1, 2)
        out = quantized_out.transpose(1, 2)
        return out, q1, q2, com

class Decoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        enc_cfg = cfg.get('speechencoder', {})
        hidden = cfg['dimension']
        strides = enc_cfg.get('strides', [8, 5, 4, 2])
        lstm_layers = enc_cfg.get('lstm_layers', 2)
        self.proj_in = nn.utils.weight_norm(
            nn.Conv1d(hidden, hidden * 2, kernel_size=7, padding=3, padding_mode='reflect')
        )
        self.lstm = nn.LSTM(hidden * 2, hidden // 2, lstm_layers, batch_first=True, bidirectional=True)
        ch = [hidden, 256, 128, 64, 1]
        self.blocks = nn.ModuleList([ConvBlock(ch[i], ch[i + 1], stride=s, transpose=True) for i, s in enumerate(strides)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        return torch.tanh(x.squeeze(1))

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

class Discriminator(nn.Module):
    def __init__(self, disc_cfg: dict = None):
        super().__init__()
        disc_cfg = disc_cfg or {}
        self.enable_mpd = disc_cfg.get("enable_mpd", True)
        self.enable_msd = disc_cfg.get("enable_msd", True)
        self.enable_mstft = disc_cfg.get("enable_mstft", True)

        if not (self.enable_mpd or self.enable_msd or self.enable_mstft):
            raise ValueError("At least one discriminator branch must be enabled.")

        self.mpd = MultiPeriodDiscriminator() if self.enable_mpd else None
        self.msd = MultiScaleDiscriminator() if self.enable_msd else None
        self.mstftd = (
            MultiScaleSTFTDiscriminator(
                filters=32,
                n_ffts=[2048, 1024, 512, 256, 128],
                hop_lengths=[512, 256, 128, 64, 32],
                win_lengths=[2048, 1024, 512, 256, 128],
            )
            if self.enable_mstft else None
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor, return_fmaps: bool = True):
        # y/y_hat: [B, 1, T]
        if y.dim() == 2:
            y = y.unsqueeze(1)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)

        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        if self.mpd is not None:
            y_mpd_r, y_mpd_g, fmap_mpd_r, fmap_mpd_g = self.mpd(y, y_hat, return_fmaps=return_fmaps)
            y_d_rs.extend(y_mpd_r)
            y_d_gs.extend(y_mpd_g)
            if return_fmaps:
                fmap_rs.extend(fmap_mpd_r)
                fmap_gs.extend(fmap_mpd_g)

        if self.msd is not None:
            y_msd_r, y_msd_g, fmap_msd_r, fmap_msd_g = self.msd(y, y_hat, return_fmaps=return_fmaps)
            y_d_rs.extend(y_msd_r)
            y_d_gs.extend(y_msd_g)
            if return_fmaps:
                fmap_rs.extend(fmap_msd_r)
                fmap_gs.extend(fmap_msd_g)

        if self.mstftd is not None:
            y_stft_r, y_stft_g, fmap_stft_r, fmap_stft_g = self.mstftd(y, y_hat, return_fmaps=return_fmaps)
            y_d_rs.extend(y_stft_r)
            y_d_gs.extend(y_stft_g)
            if return_fmaps:
                fmap_rs.extend(fmap_stft_r)
                fmap_gs.extend(fmap_stft_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
