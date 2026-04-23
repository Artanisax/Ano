# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpkDistillLoss(nn.Module):
    def __init__(self, dim: int, num_speakers: int):
        super().__init__()
        self.clf = nn.Linear(dim, num_speakers)
    def forward(self, s1: torch.Tensor, s2: torch.Tensor, spk_ids: torch.Tensor) -> torch.Tensor:
        ce1 = -F.log_softmax(self.clf(s1), dim=1).gather(1, spk_ids.unsqueeze(1)).mean()
        ce2 = -F.log_softmax(self.clf(s2), dim=1).gather(1, spk_ids.unsqueeze(1)).mean()
        cos = 1.0 - F.cosine_similarity(s1, s2, dim=-1).mean()
        return ce1 + ce2 + cos

class LinDistillLoss(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab)
    def forward(self, q1: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.proj(q1).transpose(1, 2), tokens.long())

# ✅ 1. EmoDistillLoss: 改为 log-F0 的 MSE 回归
class EmoDistillLoss(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        # 🔑 热启动：log(150Hz) ≈ 5.01，加速收敛
        with torch.no_grad():
            self.proj.bias.fill_(5.0)
    
    def forward(self, q2: torch.Tensor, f0_log: torch.Tensor) -> torch.Tensor:
        pred = self.proj(q2).squeeze(-1)  # [B, T]
        # ✅ 帧级 MSE，零 NaN 风险，梯度稳定
        return F.mse_loss(pred, f0_log)

# ✅ 2. ChromaDistillLoss: 辅助情感蒸馏（24 维色度图）
class ChromaDistillLoss(nn.Module):
    def __init__(self, dim: int, n_chroma: int = 24):
        super().__init__()
        self.proj = nn.Linear(dim, n_chroma)
        # 归一化后 chroma 均值≈0，偏置初始化为 0
        with torch.no_grad():
            self.proj.bias.zero_()
    
    def forward(self, q2: torch.Tensor, chroma: torch.Tensor) -> torch.Tensor:
        # q2: [B, T, dim], chroma: [B, T, n_chroma] (已 log+L2 归一化)
        pred = self.proj(q2)  # [B, T, n_chroma]
        return F.mse_loss(pred, chroma)

# ✅ 3. MultiResolutionSTFTLoss: 辅助重建损失（时频域联合监督）
class STFTLoss(nn.Module):
    def __init__(self, fft_size: int, hop_size: int, win_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', torch.hann_window(win_size))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # ✅ 关键修复：对齐波形长度，防止解码器 stride 导致的首尾微小偏差
        min_len = min(x.shape[-1], y.shape[-1])
        x, y = x[..., :min_len], y[..., :min_len]

        x_spec = torch.stft(x, self.fft_size, self.hop_size, self.win_size,
                        window=self.window, return_complex=True, center=False)
        y_spec = torch.stft(y, self.fft_size, self.hop_size, self.win_size,
                        window=self.window, return_complex=True, center=False)
        
        # 幅度谱 + 对数压缩
        x_mag = torch.abs(x_spec)
        y_mag = torch.abs(y_spec)
        x_log = torch.log(x_mag.clamp(min=1e-7))
        y_log = torch.log(y_mag.clamp(min=1e-7))
        
        # L1 + L2 联合损失
        return F.l1_loss(x_log, y_log) + F.mse_loss(x_log, y_log)

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, resolutions: list = None):
        super().__init__()
        if resolutions is None:
            # 标准配置：[fft_size, hop_size, win_size]
            resolutions = [
                # [2048, 240, 2048],
                # [1024, 120, 1024],
                # [512, 50, 512],
                # [256, 25, 256],
                # [128, 12, 128],
                [1024, 160, 640],
                [512, 80, 320],
                [256, 40, 160],
                [128, 20, 80],
            ]
        self.stft_losses = nn.ModuleList()
        for fft_size, hop_size, win_size in resolutions:
            self.stft_losses.append(STFTLoss(fft_size, hop_size, win_size))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0
        for stft_loss in self.stft_losses:
            loss += stft_loss(x, y)
        return loss / len(self.stft_losses)

class AdvLoss(nn.Module):
    def forward(self, disc_fake: list, disc_real: list, fmap_fake: list, fmap_real: list, mode: str = 'gen') -> torch.Tensor:
        if mode == 'gen':
            l_adv = torch.stack([torch.mean((1 - d) ** 2) for d in disc_fake]).mean()
            fm_loss = 0.0
            count = 0
            for fr_list, fg_list in zip(fmap_real, fmap_fake):
                for fr, fg in zip(fr_list, fg_list):
                    min_t = min(fr.shape[2], fg.shape[2])
                    fr_s = fr[:, :, :min_t, ...]
                    fg_s = fg[:, :, :min_t, ...]
                    fm_loss += torch.mean(torch.abs(fr_s - fg_s))
                    count += 1
            l_fm = fm_loss / count if count > 0 else 0.0
            return l_adv + 2.0 * l_fm
        else:
            l_real = torch.stack([torch.mean((1 - d) ** 2) for d in disc_real]).mean()
            l_fake = torch.stack([torch.mean(d ** 2) for d in disc_fake]).mean()
            return l_real + l_fake