# utils.py
import os, logging, random, warnings
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

def resample_audio(wav: torch.Tensor, orig_sr: int, target_sr: int = 16000) -> torch.Tensor:
    """通用重采样函数"""
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_sr, target_sr)

def load_audio(path: str, sr: int = 16000) -> torch.Tensor:
    wav, orig_sr = torchaudio.load(path)
    if wav.dim() > 1: wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr: wav = resample_audio(wav, orig_sr, sr)
    return wav

def compute_mel(wav: torch.Tensor, n_mels: int = 80, sr: int = 16000, 
                hop: int = 320, win: int = 640, n_fft: int = 1024) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft,
        win_length=win, hop_length=hop,
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect"
    ).to(wav.device)
    mel = mel_transform(wav)
    return torch.log(mel.clamp(min=1e-5))

def get_stft_params(cfg: dict, prefix: str = 'mel') -> dict:
    """
    从配置中获取 STFT 相关参数，支持模块级覆盖 + 全局回退
    返回: {'hop_length': int, 'win_length': int, 'n_fft': int}
    """
    model_cfg = cfg.get('model', {})
    global_cfg = cfg.get('stft', {})
    
    def _get(key, default):
        return model_cfg.get(f'{prefix}_{key}', global_cfg.get(key, default))
    
    return {
        'hop': _get('hop_length', 320),
        'win': _get('win_length', 640),
        'n_fft': _get('n_fft', 1024),
    }

def extract_f0_aligned(wav_np: np.ndarray, sr: int = 16000, f0_min: float = 60.0, 
                       f0_max: float = 600.0, target_frames: int = None, 
                       hop_length: int = 320) -> torch.Tensor:
    """通用 F0 提取函数，返回 log-F0（与缓存 f0_log 对齐）"""
    wav_np = wav_np.squeeze().astype(np.float64)
    
    frame_period = hop_length / sr * 1000.0
    f0, t = pyworld.harvest(wav_np, sr, frame_period=frame_period, 
                            f0_floor=f0_min, f0_ceil=f0_max)
    
    if np.any(np.isnan(f0)):
        print(f"[F0-WARN] dynamic F0 提取含 NaN | 数量={np.isnan(f0).sum()}")
    if np.any(np.isinf(f0)):
        print(f"[F0-WARN] dynamic F0 提取含 Inf | 数量={np.isinf(f0).sum()}")
        
    f0 = np.nan_to_num(f0, nan=0.0, posinf=f0_max, neginf=f0_min)
        
    voiced = f0 > 0.0
    f0_abs = np.zeros_like(f0)
    f0_log = np.zeros_like(f0)
    log_min, log_max = np.log(f0_min), np.log(f0_max)
    
    if voiced.any():
        f0_abs[voiced] = f0[voiced]
        f0_log[voiced] = np.log(f0[voiced])
        if not voiced.all():
            t_valid = np.where(voiced)[0]
            if len(t_valid) >= 2:
                with warnings.catch_warnings(record=True) as w_list:
                    warnings.simplefilter("always")
                    interp_func_abs = interp1d(
                        t_valid,
                        f0_abs[voiced],
                        kind='linear',
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    f0_abs = interp_func_abs(np.arange(len(f0_abs)))
                    interp_func_log = interp1d(
                        t_valid,
                        f0_log[voiced],
                        kind='linear',
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    f0_log = interp_func_log(np.arange(len(f0_log)))
                    if w_list and any(issubclass(w.category, RuntimeWarning) for w in w_list):
                        print(f"[F0-Warn] 插值警告 (dynamic)")
            else:
                f0_abs[:] = f0_abs[t_valid[0]] if len(t_valid) == 1 else f0_min
                f0_log[:] = f0_log[t_valid[0]] if len(t_valid) == 1 else log_min
        f0_abs = np.clip(f0_abs, f0_min, f0_max)
        f0_log = np.clip(f0_log, log_min, log_max)
    else:
        f0_abs[:] = f0_min
        f0_log[:] = log_min
            
    f0_tensor = torch.tensor(f0_log, dtype=torch.float32)
    if target_frames is not None and f0_tensor.shape[0] != target_frames:
        f0_tensor = np.nan_to_num(f0_tensor.numpy(), nan=log_min, posinf=log_max, neginf=log_min)
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
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak * target_peak
    return torch.clamp(wav, -1.0, 1.0)
