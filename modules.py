# modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ
from transformers import WavLMModel

def get_padding(k: int, d: int = 1) -> int:
    return (k * d - d) // 2

LRELU_SLOPE = 0.1

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel: int = None, transpose: bool = False):
        super().__init__()
        if kernel is None:
            kernel = stride * 2 + stride % 2
        padding = (kernel - stride) // 2
        
        if not transpose:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
            self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride) if stride > 1 else nn.Identity()
        else:
            # ConvTranspose1d 的 output_padding 用于确保输出长度严格等于 input_length * stride
            self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
            self.res = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=1, stride=stride) if stride > 1 else nn.Identity()
            
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        out = self.conv(x)
        res = self.res(x)
        
        # 处理残差连接可能的长度极小差异（通常 padding 已经对齐）
        # if out.shape[-1] != res.shape[-1]:
            # min_len = min(out.shape[-1], res.shape[-1])
            # out = out[..., :min_len]
            # res = res[..., :min_len]
            
        return self.act(self.norm(out.transpose(1, 2)).transpose(1, 2) + res)

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
        self.proj = nn.Conv1d(hidden * 2, hidden, kernel_size=1) 
    
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: [B, 1, T]
        x = wav
        for c in self.convs: 
            x = c(x)
        # x: [B, hidden, T_feat]
        x, _ = self.lstm(x.transpose(1, 2))  # [B, T_feat, hidden*2]
        return self.proj(x.transpose(1, 2)).transpose(1, 2)  # [B, T_feat, hidden]

class SpeakerEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        K = len(cfg['ref_enc_filters'])
        filters = [1] + cfg['ref_enc_filters']
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=filters[i], 
                out_channels=filters[i+1], 
                kernel_size=(cfg['ref_enc_size'], cfg['ref_enc_size']),
                stride=(cfg['ref_enc_strides'][i], cfg['ref_enc_strides'][i]), 
                padding=cfg['ref_enc_pad']
            )
            for i in range(K)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(f) for f in filters[1:]])

        # 动态计算卷积压缩后的频率维度
        def get_conv_out(in_size, kernel, stride, pad):
            return (in_size + 2 * pad - kernel) // stride + 1

        H = cfg['n_mels']  # 初始 80
        for s in cfg['ref_enc_strides']:
            H = get_conv_out(H, cfg['ref_enc_size'], s, cfg['ref_enc_pad'])
            
        out_dim = filters[-1] * H  # e.g., 128 * 3 = 384
        self.gru = nn.GRU(out_dim, cfg['ref_enc_gru_size'], batch_first=True)
        
        self.attn = nn.MultiheadAttention(cfg['ref_enc_gru_size'], cfg['num_heads'], batch_first=True)
        self.proj = nn.Linear(cfg['ref_enc_gru_size'], cfg['dim'])
        self.heads = nn.ModuleList([nn.Linear(cfg['dim'], cfg['dim']) for _ in range(cfg['num_heads'])])
        self.fusion = nn.Linear(cfg['dim'] * cfg['num_heads'], cfg['dim'])

    def forward(self, mel: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # mel: [B, 1, F, T_mel] → convs: [B, C_last, H_last, T'] → permute/view: [B, T', C_last*H_last]
        x = mel  # [B, 1, F, T_mel]
        for c, b in zip(self.convs, self.bns): 
            x = F.leaky_relu(b(c(x)), LRELU_SLOPE)  # [B, C_i, H_i, T_i]
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * H)  # [B, T, C*H]
        x, _ = self.gru(x)  # [B, T, gru_size]
        q = x.mean(dim=1, keepdim=True)  # [B, 1, gru_size]
        attn, _ = self.attn(q, x, x, key_padding_mask=mask)  # [B, 1, gru_size]
        base = self.proj(attn.squeeze(1))  # [B, dim]
        tokens = [h(base) for h in self.heads]  # List of [B, dim]
        return self.fusion(torch.cat(tokens, dim=-1))  # [B, dim]

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
        
        # 1. 镜像 Encoder 的 Proj 层 (hidden -> hidden * 2)
        self.proj_in = nn.Conv1d(hidden, hidden * 2, kernel_size=1)
        
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

class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel: int = 5, stride: int = 3, use_sn: bool = False):
        super().__init__()
        self.period = period
        nf = nn.utils.spectral_norm if use_sn else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            nf(nn.Conv2d(1, 32, (kernel, 1), (stride, 1), padding=(get_padding(kernel), 0))),
            nf(nn.Conv2d(32, 128, (kernel, 1), (stride, 1), padding=(get_padding(kernel), 0))),
            nf(nn.Conv2d(128, 512, (kernel, 1), (stride, 1), padding=(get_padding(kernel), 0))),
            nf(nn.Conv2d(512, 1024, (kernel, 1), (stride, 1), padding=(get_padding(kernel), 0))),
            nf(nn.Conv2d(1024, 1024, (kernel, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = nf(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))
    
    def forward(self, x: torch.Tensor):
        # x: [B, 1, T] → reshape: [B, 1, T//period, period] → conv2d → flatten
        fmap, b, c, t = [], *x.shape
        if t % self.period != 0: x = F.pad(x, (0, self.period - t % self.period), "reflect")
        x = x.view(b, c, -1, self.period)  # [B, 1, T//period, period]
        for l in self.convs:
            x = l(x); x = F.leaky_relu(x, LRELU_SLOPE); fmap.append(x)  # x:[B, C_i, T_i, period]
        x = self.conv_post(x); fmap.append(x)  # [B, 1, T_out, period]
        return torch.flatten(x, 1, -1), fmap  # score:[B, *], fmap:List of [B, C, T, period]

class DiscriminatorS(nn.Module):
    def __init__(self, use_sn: bool = False):
        super().__init__()
        nf = nn.utils.spectral_norm if use_sn else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            nf(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)),
            nf(nn.Conv1d(128, 128, kernel_size=41, stride=4, groups=4, padding=20)),
            nf(nn.Conv1d(128, 256, kernel_size=41, stride=4, groups=16, padding=20)),
            nf(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20)),
            nf(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20)),
            nf(nn.Conv1d(1024, 1024, kernel_size=41, stride=2, groups=16, padding=20)),
            nf(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
        ])
        self.conv_post = nf(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x: torch.Tensor):
        # x: [B, 1, T] → conv1d chain → flatten
        fmap = []
        for l in self.convs: x = l(x); x = F.leaky_relu(x, LRELU_SLOPE); fmap.append(x)  # x:[B, C_i, T_i]
        x = self.conv_post(x); fmap.append(x)  # [B, 1, T_out]
        return torch.flatten(x, 1, -1), fmap  # score:[B, *], fmap:List of [B, C, T]

class HiFiGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = nn.ModuleList([DiscriminatorP(p) for p in [2, 3, 5, 7, 11]])
        self.msd = nn.ModuleList([DiscriminatorS(use_sn=(i == 0)) for i in range(3)])
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        # y/y_hat: [B, 1, T] → MPD+MSD → scores & feature maps
        y_dr, y_dg, f_r, f_g = [], [], [], []
        for d in self.mpd:
            r, fr = d(y); g, fg = d(y_hat)  # r/g:[B,*], fr/fg:List
            y_dr.append(r); y_dg.append(g); f_r.append(fr); f_g.append(fg)
        for i, d in enumerate(self.msd):
            y_, yh_ = (self.pools[i-1](y), self.pools[i-1](y_hat)) if i > 0 else (y, y_hat)
            r, fr = d(y_); g, fg = d(yh_)
            y_dr.append(r); y_dg.append(g); f_r.append(fr); f_g.append(fg)
        return y_dr, y_dg, f_r, f_g  # Lists of [B, ...] / List of Lists