# utils.py
import os, logging, random
import numpy as np
import torch
import torchaudio
import pyworld
from scipy.interpolate import interp1d
from torch.nn import functional as F

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir: str, name: str = "vpc") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(path)
    if wav.dim() > 1: wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr: wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav

def compute_mel(wav: torch.Tensor, n_mels: int = 80, sr: int = 16000, hop: int = 320) -> torch.Tensor:
    # ✅ 修复：将 transform 动态移动到 wav 所在的设备（自动兼容 CPU/GPU）
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=1024, win_length=hop * 2, hop_length=hop
    ).to(wav.device)
    mel = mel_transform(wav)
    return torch.log(mel.clamp(min=1e-5))

def extract_f0_aligned(wav_np: np.ndarray, sr: int = 16000, f0_min: float = 60.0, 
                       f0_max: float = 600.0, target_frames: int = None) -> torch.Tensor:
    wav_np = wav_np.squeeze().astype(np.float64)
    
    # ✅ Harvest 提取（替代 dio+stonemask）
    frame_period = 320.0 / sr * 1000.0
    f0, t = pyworld.harvest(wav_np, sr, frame_period=frame_period, 
                            f0_floor=f0_min, f0_ceil=f0_max)
    
    # ✅ 清洗前检验
    if np.any(np.isnan(f0)):
        print(f"[F0-WARN] dynamic F0 提取含 NaN | 数量={np.isnan(f0).sum()}")
    if np.any(np.isinf(f0)):
        print(f"[F0-WARN] dynamic F0 提取含 Inf | 数量={np.isinf(f0).sum()}")
        
    f0 = np.nan_to_num(f0, nan=0.0, posinf=f0_max, neginf=f0_min)
        
    voiced = f0 > 0.0
    f0_abs = np.zeros_like(f0)
    
    if voiced.any():
        f0_abs[voiced] = f0[voiced]
        if not voiced.all():
            t_valid = np.where(voiced)[0]
            if len(t_valid) >= 2:
                with warnings.catch_warnings(record=True) as w_list:
                    warnings.simplefilter("always")
                    interp_func = interp1d(t_valid, f0_abs[voiced], kind='linear',
                                           bounds_error=False, fill_value="extrapolate")
                    f0_abs = interp_func(np.arange(len(f0_abs)))
                    if w_list and any(issubclass(w.category, RuntimeWarning) for w in w_list):
                        print(f"[F0-Warn] 插值警告 (dynamic)")
            else:
                f0_abs[:] = f0_abs[t_valid[0]] if len(t_valid) == 1 else f0_min
        f0_abs = np.clip(f0_abs, f0_min, f0_max)
    else:
        f0_abs[:] = f0_min
            
    f0_tensor = torch.tensor(f0_abs, dtype=torch.float32)
    if target_frames is not None and f0_tensor.shape[0] != target_frames:
        f0_tensor = np.nan_to_num(f0_tensor.numpy(), nan=f0_min, posinf=f0_max, neginf=f0_min)
        f0_tensor = torch.tensor(f0_tensor)
        f0_tensor = F.interpolate(
            f0_tensor.unsqueeze(0).unsqueeze(0), 
            size=target_frames, mode='linear', align_corners=False
        ).squeeze()
        
    return f0_tensor

def save_audio(wav: torch.Tensor, path: str, sr: int = 16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, wav, sr)

def normalize_audio(wav: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """
    统一音频响度并避免削波：峰值归一化 + 硬截断保护
    
    Args:
        wav: 输入波形 [*, T]，支持任意前导维度
        target_peak: 目标峰值 (默认 0.95，留 5% 余量防 TB 截断)
    
    Returns:
        归一化后的波形，范围严格 [-1.0, 1.0]
    """
    peak = wav.abs().max()
    if peak > 1e-6:  # 避免除零
        wav = wav / peak * target_peak
    return torch.clamp(wav, -1.0, 1.0)  # 最终保护，防止数值溢出