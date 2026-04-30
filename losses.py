# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpkDistillLoss(nn.Module):
    def __init__(self, dim: int, num_speakers: int):
        super().__init__()
        self.clf = nn.Linear(dim, num_speakers)

    def components(self, s1: torch.Tensor, s2: torch.Tensor, spk_ids: torch.Tensor):
        ce1 = -F.log_softmax(self.clf(s1), dim=1).gather(1, spk_ids.unsqueeze(1)).mean()
        ce2 = -F.log_softmax(self.clf(s2), dim=1).gather(1, spk_ids.unsqueeze(1)).mean()
        cos = self.consistency_only(s1, s2)
        return ce1, ce2, cos

    def consistency_only(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(s1, s2, dim=-1).mean()

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, spk_ids: torch.Tensor) -> torch.Tensor:
        ce1, ce2, cos = self.components(s1, s2, spk_ids)
        return ce1 + ce2 + cos

class LinDistillLoss(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab)
    def forward(self, q1: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.proj(q1).transpose(1, 2), tokens.long())

class EmoDistillLoss(nn.Module):
    def __init__(self, dim: int, f0_type: str = "log"):
        super().__init__()
        self.f0_type = f0_type
        self.proj = nn.Linear(dim, 1)
        with torch.no_grad():
            if self.f0_type == "log":
                self.proj.bias.fill_(5.0)
            else:
                self.proj.bias.fill_(150.0)
    
    def forward(self, q2: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        pred = self.proj(q2).squeeze(-1)
        if self.f0_type == "log":
            return F.mse_loss(pred, f0)
        else: # "abs"
            # Cosine embedding loss expects target to be 1 or -1
            # To measure similarity between two 1D sequences, we use cosine similarity along time dimension
            # or treat them as vectors and compute cosine similarity
            # but usually F0 is just a scalar sequence.
            # Using cosine similarity on scalars is just sign(x) * sign(y), which is not useful for F0.
            # If the paper implies Cosine Loss for "abs" F0, it might mean computing cosine similarity 
            # between the predicted and target F0 vectors (shape [B, T]).
            # Let's implement cosine loss along the sequence dimension T.
            
            # cosine_similarity expects [B, T], output [B]
            cos_sim = F.cosine_similarity(pred, f0, dim=-1)
            # Loss is 1 - cos_sim
            return (1.0 - cos_sim).mean()

class ChromaDistillLoss(nn.Module):
    def __init__(self, dim: int, n_chroma: int = 24):
        super().__init__()
        self.proj = nn.Linear(dim, n_chroma)
    
    def forward(self, q2: torch.Tensor, chroma: torch.Tensor) -> torch.Tensor:
        pred = self.proj(q2)
        return F.mse_loss(pred, chroma)

class STFTLoss(nn.Module):
    def __init__(self, fft_size: int, hop_size: int, win_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', torch.hann_window(win_size))
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != y.shape[-1]:
            raise RuntimeError(
                f"STFTLoss 输入长度不一致: x={x.shape[-1]}, y={y.shape[-1]}"
            )

        x_spec = torch.stft(x, self.fft_size, self.hop_size, self.win_size,
                        window=self.window, return_complex=True, center=False)
        y_spec = torch.stft(y, self.fft_size, self.hop_size, self.win_size,
                        window=self.window, return_complex=True, center=False)
        
        x_mag = torch.abs(x_spec)
        y_mag = torch.abs(y_spec)
        x_log = torch.log(x_mag.clamp(min=1e-7))
        y_log = torch.log(y_mag.clamp(min=1e-7))
        
        return F.l1_loss(x_log, y_log) + F.mse_loss(x_log, y_log)

class MultiResolutionSTFTLoss(nn.Module):
    """✅ 支持从配置传入 resolutions 参数"""
    def __init__(self, resolutions: list = None):
        super().__init__()
        if resolutions is None:
            resolutions = [
                [2048, 320, 1280],
                [1024, 160, 640],
                [512, 80, 320],
                [256, 40, 160]
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
    def forward(
        self,
        disc_fake: list,
        disc_real: list,
        fmap_fake: list,
        fmap_real: list,
        mode: str = 'gen',
        return_components: bool = False,
    ):
        if mode == 'gen':
            # Adversarial term: sum over enabled discriminator branches.
            l_adv = 0.0
            for dg in disc_fake:
                l_adv += torch.mean((1 - dg) ** 2)

            fm_loss = 0.0
            for fr_list, fg_list in zip(fmap_real, fmap_fake):
                branch_fm_loss = 0.0
                for fr, fg in zip(fr_list, fg_list):
                    branch_fm_loss += torch.mean(torch.abs(fr - fg))
                fm_loss += branch_fm_loss / len(fr_list)
            
            total = l_adv + fm_loss
            if return_components:
                return total, l_adv, fm_loss
            return total
        else:
            l_real = 0.0
            for dr in disc_real:
                l_real += torch.mean((1 - dr) ** 2)
            
            l_fake = 0.0
            for dg in disc_fake:
                l_fake += torch.mean(dg ** 2)
                
            total = l_real + l_fake
            if return_components:
                return total, l_real, l_fake
            return total
