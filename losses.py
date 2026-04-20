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

class EmoDistillLoss(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
    def forward(self, q2: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        pred = self.proj(q2).squeeze(-1)
        return (1.0 - F.cosine_similarity(f0, pred, dim=-1)).mean()

class AdvLoss(nn.Module):
    def forward(self, disc_fake: list, disc_real: list, fmap_fake: list, fmap_real: list, mode: str = 'gen') -> torch.Tensor:
        if mode == 'gen':
            l_adv = torch.stack([torch.mean((1 - d) ** 2) for d in disc_fake]).mean()
            
            fm_loss = 0.0
            count = 0
            for fr_list, fg_list in zip(fmap_real, fmap_fake):
                for fr, fg in zip(fr_list, fg_list):
                    # ✅ 修复：时间维度恒为 dim=2，显式对齐避免 period/channel 误截
                    min_t = min(fr.shape[2], fg.shape[2])
                    fr_s = fr[:, :, :min_t, ...]
                    fg_s = fg[:, :, :min_t, ...]
                    fm_loss += torch.mean(torch.abs(fr_s - fg_s))
                    count += 1
                    
            l_fm = fm_loss / count if count > 0 else 0.0
            return l_adv + 2.0 * l_fm  # 2.0 对齐 HiFi-GAN λ_FM
        else:
            l_real = torch.stack([torch.mean((1 - d) ** 2) for d in disc_real]).mean()
            l_fake = torch.stack([torch.mean(d ** 2) for d in disc_fake]).mean()
            return l_real + l_fake