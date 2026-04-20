# preprocess.py
import os, sys, yaml, json, random, argparse
import numpy as np
import torch
import torchaudio
import pyworld
import librosa
import warnings
from scipy.interpolate import interp1d
import joblib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from torch.nn import functional as F

def load_cfg(path: str = "configs.yaml") -> dict:
    with open(path) as f: return yaml.safe_load(f)

def run_manifest(cfg: dict):
    splits = {"train": cfg['paths']['train_dirs'], "val": cfg['paths']['val_dirs'], "test": cfg['paths']['test_dirs']}
    os.makedirs(cfg['paths']['manifest_dir'], exist_ok=True)
    seed = cfg.get('random_seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    for split, dirs in splits.items():
        mf = os.path.join(cfg['paths']['manifest_dir'], f"{split}_manifest.txt")
        spk_map_path = os.path.join(cfg['paths']['manifest_dir'], f"{split}_manifest_spk_map.json")
        if os.path.exists(mf): print(f"[{split}] Manifest exists, skipping."); continue
        entries = []
        for d in dirs:
            if not os.path.isdir(d): continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(('.wav', '.flac')):
                        spk = os.path.basename(root)
                        if spk.isdigit(): entries.append((os.path.join(root, f), int(spk)))
        if not entries: continue
        unique_spks = sorted(set(s for _, s in entries))
        spk_map = {old: new for new, old in enumerate(unique_spks)}
        with open(spk_map_path, 'w') as fp: json.dump(spk_map, fp)
        with open(mf, 'w') as fp:
            for wav, old_spk in entries: fp.write(f"{wav}|{spk_map[old_spk]}\n")
        print(f"[{split}] Generated: {len(entries)} utts, {len(unique_spks)} spk.")

def _f0_worker(args: tuple) -> int:
    wav_path, f0_dir, f0_min, f0_max = args
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    uid = os.path.basename(wav_path).split('.')[0]
    out = os.path.join(f0_dir, f"{uid}.npy")
    if os.path.exists(out): return 1
    try:
        # ───────── 1. 音频加载与重采样 ─────────
        wav, sr = torchaudio.load(wav_path)
        if wav.dim() > 1: wav = wav.mean(0, keepdim=True)
        if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
        wav_np = wav.squeeze().numpy().astype(np.float64)  # pyworld 强制要求 float64

        target = len(wav_np) // 320  # 目标帧数 (严格对齐 mel_hop_length=320)
        if target < 2: return 0

        # ───────── 2. pyworld Harvest F0 提取 ─────────
        frame_period = 320 / 16000 * 1000  # 20ms
        f0, t = pyworld.harvest(wav_np, sr, frame_period=frame_period,
                                f0_floor=f0_min, f0_ceil=f0_max)

        # ✅ 清洗前检验：捕获 NaN/Inf 并警告
        if np.any(np.isnan(f0)):
            print(f"[F0-WARN] harvest 输出含 NaN: {wav_path} | 数量={np.isnan(f0).sum()}")
        if np.any(np.isinf(f0)):
            print(f"[F0-WARN] harvest 输出含 Inf: {wav_path} | 数量={np.isinf(f0).sum()}")

        # 立即清洗源头，防止污染后续流程
        f0 = np.nan_to_num(f0, nan=0.0, posinf=f0_max, neginf=f0_min)

        # ───────── 3 & 4. 线性域 F0 + 清音插值 ─────────
        voiced = f0 > 0.0
        f0_abs = np.zeros_like(f0)

        if voiced.any():
            f0_abs[voiced] = f0[voiced]
            if not voiced.all():
                t_valid = np.where(voiced)[0]
                if len(t_valid) >= 2:
                    # 安全插值：点数充足时才启用 linear extrapolation
                    with warnings.catch_warnings(record=True) as w_list:
                        warnings.simplefilter("always")
                        interp_func = interp1d(t_valid, f0_abs[voiced], kind='linear',
                                               bounds_error=False, fill_value="extrapolate")
                        f0_abs = interp_func(np.arange(len(f0_abs)))
                        # 捕获插值过程中的数值警告
                        if w_list and any(issubclass(w.category, RuntimeWarning) for w in w_list):
                            print(f"[F0-Warn] 插值警告: {wav_path}")
                else:
                    # 仅 1 帧浊音：无法计算斜率，全局填充该值
                    f0_abs[:] = f0_abs[t_valid[0]] if len(t_valid) == 1 else f0_min

            # ✅ 插值后立即截断越界值
            f0_abs = np.clip(f0_abs, f0_min, f0_max)
        else:
            # 全清音/静音兜底
            f0_abs[:] = f0_min

        # ───────── 5. 对齐帧数 + 二次清洗 ─────────
        if f0_abs.shape[0] != target:
            # 插值前确保无 NaN/inf，防止 torch 传播
            f0_abs = np.nan_to_num(f0_abs, nan=f0_min, posinf=f0_max, neginf=f0_min)
            f0_abs = F.interpolate(
                torch.tensor(f0_abs).unsqueeze(0).unsqueeze(0),
                size=target, mode='linear', align_corners=False
            ).squeeze().numpy()

        # ───────── 6. 最终兜底检查 + 保存 ─────────
        if np.any(np.isnan(f0_abs)) or np.any(np.isinf(f0_abs)):
            print(f"[F0-FIX] 最终清洗: {wav_path}")
            f0_abs = np.nan_to_num(f0_abs, nan=f0_min, posinf=f0_max, neginf=f0_min)

        os.makedirs(f0_dir, exist_ok=True)
        np.save(out, f0_abs.astype(np.float32))
        return 1
    except Exception as e:
        print(f"[F0 ERROR] {wav_path}: {e}", file=sys.stderr)
        return 0

def run_f0(cfg: dict, workers: int = 8):
    os.makedirs(cfg['data']['f0_dir'], exist_ok=True)
    f0_min, f0_max = cfg['preprocess']['f0_min'], cfg['preprocess']['f0_max']
    for split in ["train", "val", "test"]:
        mf = os.path.join(cfg['paths']['manifest_dir'], f"{split}_manifest.txt")
        if not os.path.exists(mf): print(f"[{split}] Manifest missing."); continue
        with open(mf) as f: paths = [l.strip().split('|')[0] for l in f if l.strip()]
        tasks = [(p, cfg['data']['f0_dir'], f0_min, f0_max) for p in paths]
        print(f"[{split}] Caching F0 ({workers} workers)...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(tqdm(ex.map(_f0_worker, tasks), total=len(tasks), desc=f"F0-{split}"))
        print(f"[{split}] Done: {sum(res)}/{len(paths)}")

def _chroma_worker(args: tuple) -> int:
    """
    单文件 Chroma 提取 worker
    args: (wav_path, chroma_dir, target_frames)
    """
    wav_path, chroma_dir, target_frames = args
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    uid = os.path.basename(wav_path).split('.')[0]
    out = os.path.join(chroma_dir, f"{uid}.npy")
    if os.path.exists(out): return 1
    try:
        # ───────── 1. 音频加载与重采样 ─────────
        wav, sr = torchaudio.load(wav_path)
        if wav.dim() > 1: wav = wav.mean(0, keepdim=True)
        if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
        wav_np = wav.squeeze().numpy().astype(np.float32)  # librosa 支持 float32

        # 计算目标帧数（严格对齐 F0/Encoder: hop=320 @ 16kHz）
        if target_frames is None:
            target_frames = len(wav_np) // 320
        if target_frames < 2: return 0

        # ───────── 2. Chroma 提取（librosa） ─────────
        # n_fft=1024, hop_length=320 是 16kHz 语音的标准配置
        chroma = librosa.feature.chroma_stft(
            y=wav_np, sr=16000, n_fft=1024, hop_length=320, n_chroma=24
        )  # [24, T_chroma]

        # ✅ 清洗前检验：捕获 NaN/Inf 并警告
        if np.any(np.isnan(chroma)):
            print(f"[Chroma-WARN] 提取含 NaN: {wav_path} | 数量={np.isnan(chroma).sum()}")
        if np.any(np.isinf(chroma)):
            print(f"[Chroma-WARN] 提取含 Inf: {wav_path} | 数量={np.isinf(chroma).sum()}")

        # 立即清洗源头
        chroma = np.nan_to_num(chroma, nan=0.0, posinf=1.0, neginf=0.0)

        # ───────── 3. Log 压缩 + L2 归一化（对齐 log-F0 量纲） ─────────
        # log 压缩：抑制高频谐波能量主导，使 12 维分布更均衡
        chroma_log = np.log(chroma + 1e-5)  # [12, T]
        # L2 归一化：每帧独立归一化，消除绝对能量差异，聚焦频谱形状
        chroma_norm = torch.nn.functional.normalize(
            torch.tensor(chroma_log), dim=0, p=2
        ).numpy()  # [12, T]

        # ───────── 4. 对齐到目标帧数 ─────────
        if chroma_norm.shape[1] != target_frames:
            # 先转 [1, 12, T] 再 interpolate，保持 12 维通道独立
            chroma_norm = F.interpolate(
                torch.tensor(chroma_norm).unsqueeze(0),
                size=target_frames, mode='linear', align_corners=False
            ).squeeze(0).numpy()  # [12, target_frames]

        # ───────── 5. 最终兜底检查 + 转置保存 ─────────
        if np.any(np.isnan(chroma_norm)) or np.any(np.isinf(chroma_norm)):
            print(f"[Chroma-FIX] 最终清洗: {wav_path}")
            chroma_norm = np.nan_to_num(chroma_norm, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 转置为 [T, 12] 便于后续加载 (与 F0 的 [T] 格式统一)
        chroma_final = chroma_norm.T.astype(np.float32)  # [target_frames, 12]
        
        os.makedirs(chroma_dir, exist_ok=True)
        np.save(out, chroma_final)
        return 1
    except Exception as e:
        print(f"[Chroma ERROR] {wav_path}: {e}", file=sys.stderr)
        return 0

def run_chroma(cfg: dict, workers: int = 8):
    """并行预处理 Chroma 缓存"""
    chroma_dir = cfg['data'].get('chroma_dir')
    os.makedirs(chroma_dir, exist_ok=True)
    
    for split in ["train", "val", "test"]:
        mf = os.path.join(cfg['paths']['manifest_dir'], f"{split}_manifest.txt")
        if not os.path.exists(mf): 
            print(f"[{split}] Manifest missing, skip chroma."); 
            continue
        with open(mf) as f: 
            paths = [l.strip().split('|')[0] for l in f if l.strip()]
        
        # 预计算 target_frames（与 F0 严格对齐）
        tasks = []
        for p in paths:
            try:
                info = torchaudio.info(p)
                target = info.num_frames // 320  # hop=320 @ 16kHz
                if target >= 2:
                    tasks.append((p, chroma_dir, target))
            except:
                continue  # 跳过损坏文件
        
        if not tasks: 
            print(f"[{split}] No valid utts for chroma."); 
            continue
            
        print(f"[{split}] Caching Chroma ({workers} workers)...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(tqdm(ex.map(_chroma_worker, tasks), total=len(tasks), desc=f"Chroma-{split}"))
        print(f"[{split}] Done: {sum(res)}/{len(tasks)}")

def run_kmeans(cfg: dict, gpu: int = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from transformers import AutoFeatureExtractor, WavLMModel
    from cuml import KMeans
    seed = cfg.get('random_seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    device = torch.device("cuda:0")
    mf = os.path.join(cfg['paths']['manifest_dir'], "train_manifest.txt")
    if not os.path.exists(mf): raise FileNotFoundError("Train manifest missing.")
    with open(mf) as f: paths = [l.strip().split('|')[0] for l in f if l.strip()]
    random.shuffle(paths)
    
    print("Filtering short utts...")
    valid = [p for p in tqdm(paths, desc="Scan") if os.path.exists(p) and torchaudio.info(p).num_frames // 320 >= 100]
    print(f"Loading WavLM...")
    ext = AutoFeatureExtractor.from_pretrained(cfg['preprocess']['wavlm_model'])
    model = WavLMModel.from_pretrained(cfg['preprocess']['wavlm_model']).to(device).eval()
    layer_idx = cfg['preprocess']['wavlm_layer_idx']
    target = cfg['preprocess']['kmeans_sample_frames']
    max_per = cfg['preprocess'].get('kmeans_max_per_utt', 200)
    
    batch_size = 4
    feats, collected = [], 0
    print("Sampling balanced features...")
    with torch.no_grad():
        for i in tqdm(range(0, len(valid), batch_size), desc="Extract"):
            batch = []
            for p in valid[i:i+batch_size]:
                w, sr = torchaudio.load(p)
                if w.dim() > 1: w = w.mean(0, keepdim=True)
                if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
                batch.append(w.squeeze())
            if not batch: continue
            pad = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True).to(device)
            inp = ext(pad.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            out = model(**inp, output_hidden_states=True).hidden_states[layer_idx]
            for b in range(len(batch)):
                vl = min(out.shape[1], int(np.ceil(len(batch[b]) / 320)))
                if vl <= 0: continue
                f = out[b, :vl].cpu().numpy()
                if collected < target:
                    rem = target - collected
                    step = max(1, int(np.ceil(len(f) / min(max_per, rem))))
                    feats.append(f[::step][:min(max_per, rem)])
                    collected += len(feats[-1])
                else: break
            if collected >= target: break
    del model, ext, pad, inp, out
    torch.cuda.empty_cache()
    X = np.concatenate(feats, axis=0).astype(np.float32)
    print(f"Training K-Means ({X.shape[0]} frames)...")
    km = KMeans(
        n_clusters=cfg['preprocess']['kmeans_n_clusters'],
        max_iter=1000,
        init='k-means++',
        random_state=seed,
        verbose=False,
    )
    km.fit(X)
    os.makedirs(os.path.dirname(cfg['paths']['kmeans_path']), exist_ok=True)
    joblib.dump(km, cfg['paths']['kmeans_path'])
    centers = km.cluster_centers_
    print(f"Saved to {cfg['paths']['kmeans_path']}")
    
    if hasattr(centers, 'get'): centers = centers.get()  # cuML CuPy Array -> NumPy
    kmeans_npy_path = cfg['paths']['kmeans_path'].replace('.pkl', '.npy')
    np.save(kmeans_npy_path, centers)

def run_tokens(cfg: dict, gpu: int = 0):
    """缓存 K-Means Token (GPU)"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from transformers import AutoFeatureExtractor, WavLMModel
    seed = cfg.get('random_seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    device = torch.device("cuda:0")
    km = joblib.load(cfg['paths']['kmeans_path'])
    ext = AutoFeatureExtractor.from_pretrained(cfg['preprocess']['wavlm_model'])
    model = WavLMModel.from_pretrained(cfg['preprocess']['wavlm_model']).to(device).eval()
    tok_dir = cfg['data']['token_dir']
    os.makedirs(tok_dir, exist_ok=True)
    layer_idx = cfg['preprocess']['wavlm_layer_idx']
    batch_size = 4
    
    for split in ["train", "val", "test"]:
        mf = os.path.join(cfg['paths']['manifest_dir'], f"{split}_manifest.txt")
        if not os.path.exists(mf): continue
        with open(mf) as f: paths = [l.strip().split('|')[0] for l in f if l.strip()]
        print(f"[{split}] Caching tokens...")
        with torch.no_grad():
            for i in tqdm(range(0, len(paths), batch_size), desc=f"Token-{split}"):
                batch, p_batch = [], paths[i:i+batch_size]
                for p in p_batch:
                    if not os.path.exists(p): continue
                    w, sr = torchaudio.load(p)
                    if w.dim() > 1: w = w.mean(0, keepdim=True)
                    if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
                    batch.append(w.squeeze())
                if not batch: continue
                pad = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True).to(device)
                inp = ext(pad.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).to(device)
                feats = model(**inp, output_hidden_states=True).hidden_states[layer_idx]
                for b, p in enumerate(p_batch):
                    uid = os.path.basename(p).split('.')[0]
                    out = os.path.join(tok_dir, f"{uid}.npy")
                    if os.path.exists(out): continue
                    vl = min(feats.shape[1], int(np.ceil(len(batch[b]) / 320)))
                    if vl <= 0: continue
                    np.save(out, km.predict(feats[b, :vl].cpu().numpy()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs.yaml')
    parser.add_argument('--mode', choices=['manifest', 'f0', 'chroma', 'kmeans', 'tokens'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=9)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    seed = cfg.get('random_seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    
    if args.mode == 'manifest': run_manifest(cfg)
    elif args.mode == 'f0': run_f0(cfg, args.workers)
    elif args.mode == 'chroma': run_chroma(cfg, args.workers)
    elif args.mode == 'kmeans': run_kmeans(cfg, args.gpu)
    elif args.mode == 'tokens': run_tokens(cfg, args.gpu)

if __name__ == "__main__":
    main()