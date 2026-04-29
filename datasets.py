# datasets.py
import os, random
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import load_audio, extract_f0_aligned

class VPDataset(Dataset):
    def __init__(self, manifest: str, cfg: dict, training: bool = True):
        from tqdm import tqdm  # ✅ 局部导入，避免全局依赖
        
        self.cfg, self.training = cfg, training
        self.use_cache = cfg['data']['use_cache']
        self.f0_dir, self.tok_dir = cfg['data']['f0_dir'], cfg['data']['token_dir']
        
        MIN_DURATION_SEC = cfg['data'].get('min_duration_sec', 2.0)
        MIN_SAMPLES = int(MIN_DURATION_SEC * cfg['model']['sample_rate'])
        
        # ───────── 1. 预扫描：统计有效行数（用于进度条总数） ─────────
        total_lines = 0
        with open(manifest) as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2 and os.path.exists(parts[0]):
                    total_lines += 1
        
        # ───────── 2. 主循环：带进度条过滤短音频 ─────────
        self.entries = []
        with open(manifest) as f:
            for line in tqdm(f, total=total_lines, desc=f"[{manifest.split('/')[-1]}] Filtering", unit="utts"):
                parts = line.strip().split('|')
                if len(parts) != 2 or not os.path.exists(parts[0]):
                    continue
                    
                wav_path, spk_id = parts[0], int(parts[1])
                
                try:
                    info = torchaudio.info(wav_path)
                    if info.num_frames / info.sample_rate < MIN_DURATION_SEC:
                        continue
                except Exception:
                    continue
                    
                self.entries.append({'wav': wav_path, 'spk': spk_id})
        
        self.T = cfg['training']['target_samples']
        self.S = cfg['model']['hop_length']
        self.target_frames = self.T // self.S
        
        # ───────── 3. 最终统计（使用 tqdm.write 避免多进程输出交错） ─────────
        tqdm.write(f"[Dataset] Loaded {len(self.entries)}/{total_lines} utts "
                   f"(filtered <{MIN_DURATION_SEC}s | {manifest.split('/')[-1]})")

    def __len__(self) -> int:
        return len(self.entries)

    def _get_segment_starts(self, wav_len: int, num_segments: int) -> list:
        mx = max(0, wav_len - self.T)
        if self.training:
            return [(random.randint(0, mx) // self.S) * self.S for _ in range(num_segments)]

        if mx == 0:
            return [0 for _ in range(num_segments)]

        last_start = (mx // self.S) * self.S
        if num_segments == 2:
            return [0, last_start]
        elif num_segments == 3:
            mid_start = (mx // 2 // self.S) * self.S
            return [0, mid_start, last_start]

        return [0 for _ in range(num_segments - 1)] + [last_start]

    def _slice_1d_feature(self, feat: torch.Tensor, start: int, end: int, pad_value: float) -> torch.Tensor:
        seg = feat[start:end]
        if seg.shape[0] < self.target_frames:
            seg = F.pad(seg, (0, self.target_frames - seg.shape[0]), value=pad_value)
        else:
            seg = seg[:self.target_frames]
        return seg

    def _slice_2d_feature(self, feat: torch.Tensor, start: int, end: int) -> torch.Tensor:
        seg = feat[start:end]
        if seg.shape[0] < self.target_frames:
            seg = F.pad(seg, (0, 0, 0, self.target_frames - seg.shape[0]), value=0)
        else:
            seg = seg[:self.target_frames]
        return seg

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        wav = load_audio(e['wav'])
        if wav.shape[-1] < self.T:
            wav = F.pad(wav, (0, self.T - wav.shape[-1]))
            
        uid = os.path.basename(e['wav']).split('.')[0]
        
        # F0 加载
        f0_path = os.path.join(self.f0_dir, f"{uid}.npy")
        if self.use_cache and os.path.exists(f0_path):
            f0_full = torch.tensor(np.load(f0_path), dtype=torch.float32)
        else:
            f0_full = extract_f0_aligned(wav.squeeze().numpy(), target_frames=wav.shape[-1] // self.S)
        
        # ✅ Chroma 加载（可选）
        chroma_dir = self.cfg['data'].get('chroma_dir')
        chroma_full = None
        if chroma_dir and self.use_cache:
            chroma_path = os.path.join(chroma_dir, f"{uid}.npy")
            if os.path.exists(chroma_path):
                chroma_full = torch.tensor(np.load(chroma_path), dtype=torch.float32)  # [T, 24]
        
        # Token 加载
        tok_path = os.path.join(self.tok_dir, f"{uid}.npy")
        tok_full = torch.tensor(np.load(tok_path), dtype=torch.long) if (self.use_cache and os.path.exists(tok_path)) else None

        num_segments = 3
        starts = self._get_segment_starts(wav.shape[-1], num_segments)
        ends = [s + self.T for s in starts]

        wav_segments = [wav[:, s:e] for s, e in zip(starts, ends)]
        f0_segments, chroma_segments, tok_segments = [], [], []
        for start, end in zip(starts, ends):
            fs, fe = start // self.S, end // self.S
            f0_segments.append(self._slice_1d_feature(f0_full, fs, fe, float(np.log(60.0))))

            if chroma_full is not None:
                chroma_segments.append(self._slice_2d_feature(chroma_full, fs, fe))

            if tok_full is not None:
                tok_segments.append(self._slice_1d_feature(tok_full, fs, fe, 0))

        return {
            'wav': torch.stack(wav_segments, dim=0),
            'f0': torch.stack(f0_segments, dim=0),
            'chroma': torch.stack(chroma_segments, dim=0) if chroma_segments else None,
            'tok': torch.stack(tok_segments, dim=0) if tok_full is not None else None,
            'spk': e['spk'],
            'uid': uid,
        }

def collate_fn(batch: list) -> dict:
    if batch[0]['wav'].dim() == 3:
        chroma_batch = torch.stack([b['chroma'] for b in batch], dim=0) if batch[0].get('chroma') is not None else None
        return {
            'wav': torch.stack([b['wav'] for b in batch]),
            'f0': torch.stack([b['f0'] for b in batch]),
            'chroma': chroma_batch,
            'tok': torch.stack([b['tok'] for b in batch]) if batch[0]['tok'] is not None else None,
            'spk_ids': torch.tensor([b['spk'] for b in batch]),
            'uid': [b.get('uid', '') for b in batch],
        }

    lengths = [b['wav'].shape[-1] for b in batch]
    max_len = max(lengths)
    wav_padded = torch.stack([F.pad(b['wav'], (0, max_len - b['wav'].shape[-1])) for b in batch])
    return {
        'wav': wav_padded,
        'lengths': torch.tensor(lengths),
        'uid': [b.get('uid', '') for b in batch]
    }
