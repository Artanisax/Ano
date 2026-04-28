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
        
        if self.training:
            mx = max(0, wav.shape[-1] - self.T)
            starts = [(random.randint(0, mx) // self.S) * self.S for _ in range(3)]
            ends = [s + self.T for s in starts]
            
            # F0/Chroma/Token 对齐逻辑
            f0_segments, chroma_segments, tok_segments = [], [], []
            for i in range(3):
                fs, fe = starts[i] // self.S, ends[i] // self.S
                
                # F0 对齐
                f0_seg = f0_full[fs:fe]
                if f0_seg.shape[0] < self.target_frames:
                    f0_seg = F.pad(f0_seg, (0, self.target_frames - f0_seg.shape[0]), value=np.log(60.0))  # log-F0 兜底
                else:
                    f0_seg = f0_seg[:self.target_frames]
                f0_segments.append(f0_seg)

                # ✅ Chroma 对齐
                if chroma_full is not None:
                    c_seg = chroma_full[fs:fe]  # [T_seg, 24]
                    if c_seg.shape[0] < self.target_frames:
                        c_seg = F.pad(c_seg, (0, 0, 0, self.target_frames - c_seg.shape[0]), value=0)
                    else:
                        c_seg = c_seg[:self.target_frames]
                    chroma_segments.append(c_seg)

                # Token 对齐
                if tok_full is not None:
                    tok_seg = tok_full[fs:fe]
                    if tok_seg.shape[0] < self.target_frames:
                        tok_seg = F.pad(tok_seg, (0, self.target_frames - tok_seg.shape[0]), value=0)
                    else:
                        tok_seg = tok_seg[:self.target_frames]
                    tok_segments.append(tok_seg)

            return {
                'wav': torch.stack([wav[:, starts[0]:ends[0]], 
                                    wav[:, starts[1]:ends[1]], 
                                    wav[:, starts[2]:ends[2]]], dim=0),
                'f0': torch.stack(f0_segments, dim=0),  # [3, 300]
                'chroma': torch.stack(chroma_segments, dim=0) if chroma_segments else None,  # [3, 300, 24] ✅
                'tok': torch.stack(tok_segments, dim=0) if tok_full is not None else None,
                'spk': e['spk']
            }
        return {'wav': wav, 'f0': f0_full, 'chroma': chroma_full, 'tok': tok_full, 'spk': e['spk'], 'uid': uid}

def collate_fn(batch: list) -> dict:
    # 训练模式：已在 __getitem__ 中强制对齐，直接 stack
    if batch[0]['wav'].dim() == 3:  # 训练模式
        chroma_batch = torch.stack([b['chroma'] for b in batch], dim=0) if batch[0].get('chroma') is not None else None
        return {
            'wav': torch.stack([b['wav'] for b in batch]),
            'f0': torch.stack([b['f0'] for b in batch]),
            'chroma': chroma_batch,  # ✅ [B, 3, 300, 24]
            'tok': torch.stack([b['tok'] for b in batch]) if batch[0]['tok'] is not None else None,
            'spk_ids': torch.tensor([b['spk'] for b in batch])
        }
    
    # 验证/测试模式：动态长度，需对音频做 Padding
    lengths = [b['wav'].shape[-1] for b in batch]
    max_len = max(lengths)
    wav_padded = torch.stack([F.pad(b['wav'], (0, max_len - b['wav'].shape[-1])) for b in batch])  # [B, 1, T_max]
    return {
        'wav': wav_padded,              # [B, 1, T_max]
        'lengths': torch.tensor(lengths),  # [B]
        'uid': [b.get('uid', '') for b in batch]  # List[B]
    }
