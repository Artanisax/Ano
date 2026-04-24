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
import multiprocessing as mp

# 🔑 关键修复 1: 强制使用 spawn 启动方法，避免 fork 继承损坏的 CUDA 上下文
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

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

# 🔑 关键修复 2: Worker 初始化函数，限制子进程内部库的线程数
def _init_worker():
    """每个子进程启动时执行一次，防止 OpenMP/MKL 线程爆炸"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 🔑 关键修复 3: 纯 NumPy 帧对齐函数，替代 F.interpolate
def _align_frames(arr: np.ndarray, target: int) -> np.ndarray:
    """使用 scipy 线性插值对齐帧数，完全避免 torch 依赖"""
    if arr.shape[0] == target:
        return arr
    x_old = np.linspace(0, 1, arr.shape[0], endpoint=True)
    x_new = np.linspace(0, 1, target, endpoint=True)
    return interp1d(x_old, arr, kind='linear', bounds_error=False, fill_value="extrapolate")(x_new)

def _f0_worker(args: tuple) -> int:
    """
    单文件 F0 提取 worker，同时生成 abs 和 log 两个版本
    args: (wav_path, f0_abs_dir, f0_log_dir, f0_min, f0_max, hop_length)
    """
    wav_path, f0_abs_dir, f0_log_dir, f0_min, f0_max, hop_length = args
    # 子进程内再次确保不使用 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    uid = os.path.basename(wav_path).split('.')[0]
    
    out_abs = os.path.join(f0_abs_dir, f"{uid}.npy")
    out_log = os.path.join(f0_log_dir, f"{uid}.npy")
    
    if os.path.exists(out_abs) and os.path.exists(out_log): 
        return 1
    
    try:
        # ───────── 1. 音频加载与重采样 ─────────
        wav, sr = torchaudio.load(wav_path)
        if wav.dim() > 1: wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        wav_np = wav.squeeze().numpy().astype(np.float64)

        target = len(wav_np) // hop_length
        if target < 2: return 0

        # ───────── 2. pyworld Harvest F0 提取 ─────────
        frame_period = hop_length / 16000 * 1000
        f0, t = pyworld.harvest(wav_np, sr, frame_period=frame_period,
                                f0_floor=f0_min, f0_ceil=f0_max)

        if np.any(np.isnan(f0)):
            print(f"[F0-WARN] harvest 输出含 NaN: {wav_path} | 数量={np.isnan(f0).sum()}")
        if np.any(np.isinf(f0)):
            print(f"[F0-WARN] harvest 输出含 Inf: {wav_path} | 数量={np.isinf(f0).sum()}")

        f0 = np.nan_to_num(f0, nan=0.0, posinf=f0_max, neginf=f0_min)

        # ───────── 3 & 4. 双版本处理 ─────────
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
                        interp_func_abs = interp1d(t_valid, f0_abs[voiced], kind='linear',
                                                   bounds_error=False, fill_value="extrapolate")
                        f0_abs = interp_func_abs(np.arange(len(f0_abs)))
                        
                        interp_func_log = interp1d(t_valid, f0_log[voiced], kind='linear',
                                                   bounds_error=False, fill_value="extrapolate")
                        f0_log = interp_func_log(np.arange(len(f0_log)))
                        
                        if w_list and any(issubclass(w.category, RuntimeWarning) for w in w_list):
                            print(f"[F0-Warn] 插值警告: {wav_path}")
                else:
                    fill_abs = f0_abs[t_valid[0]] if len(t_valid) == 1 else f0_min
                    fill_log = f0_log[t_valid[0]] if len(t_valid) == 1 else log_min
                    f0_abs[:] = fill_abs
                    f0_log[:] = fill_log

            f0_abs = np.clip(f0_abs, f0_min, f0_max)
            f0_log = np.clip(f0_log, log_min, log_max)
        else:
            f0_abs[:] = f0_min
            f0_log[:] = log_min

        # ───────── 5. 对齐帧数 + 二次清洗（✅ 使用纯 NumPy 实现） ─────────
        if f0_abs.shape[0] != target:
            f0_abs = np.nan_to_num(f0_abs, nan=f0_min, posinf=f0_max, neginf=f0_min)
            f0_log = np.nan_to_num(f0_log, nan=log_min, posinf=log_max, neginf=log_min)
            f0_abs = _align_frames(f0_abs, target)
            f0_log = _align_frames(f0_log, target)

        # ───────── 6. 最终兜底检查 + 保存 ─────────
        if np.any(np.isnan(f0_abs)) or np.any(np.isinf(f0_abs)):
            print(f"[F0-FIX] 最终清洗 (abs): {wav_path}")
            f0_abs = np.nan_to_num(f0_abs, nan=f0_min, posinf=f0_max, neginf=f0_min)
        if np.any(np.isnan(f0_log)) or np.any(np.isinf(f0_log)):
            print(f"[F0-FIX] 最终清洗 (log): {wav_path}")
            f0_log = np.nan_to_num(f0_log, nan=log_min, posinf=log_max, neginf=log_min)

        os.makedirs(f0_abs_dir, exist_ok=True)
        os.makedirs(f0_log_dir, exist_ok=True)
        np.save(out_abs, f0_abs.astype(np.float32))
        np.save(out_log, f0_log.astype(np.float32))
        return 1
    except Exception as e:
        print(f"[F0 ERROR] {wav_path}: {e}", file=sys.stderr)
        return 0

def run_f0(cfg: dict, workers: int = 8):
    f0_abs_dir = cfg['data']['f0_abs_dir']
    f0_log_dir = cfg['data']['f0_log_dir']
    
    os.makedirs(f0_abs_dir, exist_ok=True)
    os.makedirs(f0_log_dir, exist_ok=True)
    
    f0_min, f0_max = cfg['preprocess']['f0_min'], cfg['preprocess']['f0_max']
    hop_length = cfg['model'].get('mel_hop_length', 320)
    
    for split in ["train", "val", "test"]:
        mf = os.path.join(cfg['paths']['manifest_dir'], f"{split}_manifest.txt")
        if not os.path.exists(mf): print(f"[{split}] Manifest missing."); continue
        with open(mf) as f: paths = [l.strip().split('|')[0] for l in f if l.strip()]
        tasks = [(p, f0_abs_dir, f0_log_dir, f0_min, f0_max, hop_length) for p in paths]
        print(f"[{split}] Caching F0 (abs+log, {workers} workers)...")
        # 🔑 关键修复：传入 initializer=_init_worker
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            res = list(tqdm(ex.map(_f0_worker, tasks), total=len(tasks), desc=f"F0-{split}"))
        print(f"[{split}] Done: {sum(res)}/{len(paths)}")

# ✅ 核心修复：移除 log 与额外归一化，直接使用 librosa 默认输出
def _chroma_worker(args: tuple) -> int:
    wav_path, chroma_dir, target_frames, n_fft, hop_length, win_length = args
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    uid = os.path.basename(wav_path).split('.')[0]
    out = os.path.join(chroma_dir, f"{uid}.npy")
    if os.path.exists(out): return 1
    try:
        wav, sr = torchaudio.load(wav_path)
        if wav.dim() > 1: wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        wav_np = wav.squeeze().numpy().astype(np.float32)

        if target_frames is None:
            target_frames = len(wav_np) // hop_length
        if target_frames < 2: return 0

        # ✅ librosa 默认已按帧归一化（最大值为1），直接保留原始能量比
        chroma = librosa.feature.chroma_stft(
            y=wav_np, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_chroma=24
        )  # [24, T_chroma]

        if np.any(np.isnan(chroma)):
            print(f"[Chroma-WARN] 提取含 NaN: {wav_path}")
        if np.any(np.isinf(chroma)):
            print(f"[Chroma-WARN] 提取含 Inf: {wav_path}")

        chroma = np.nan_to_num(chroma, nan=0.0, posinf=1.0, neginf=0.0)

        # ✅ 沿时间轴 (axis=1) 插值，避免维度错位
        if chroma.shape[1] != target_frames:
            x_old = np.linspace(0, 1, chroma.shape[1], endpoint=True)
            x_new = np.linspace(0, 1, target_frames, endpoint=True)
            interp_func = interp1d(x_old, chroma, kind='linear', axis=1,
                                   bounds_error=False, fill_value="extrapolate")
            chroma = interp_func(x_new)

        if np.any(np.isnan(chroma)) or np.any(np.isinf(chroma)):
            print(f"[Chroma-FIX] 最终清洗: {wav_path}")
            chroma = np.nan_to_num(chroma, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 转置为 [T, 24] 便于后续加载
        chroma_final = chroma.T.astype(np.float32)
        os.makedirs(chroma_dir, exist_ok=True)
        np.save(out, chroma_final)
        return 1
    except Exception as e:
        print(f"[Chroma ERROR] {wav_path}: {e}", file=sys.stderr)
        return 0

def run_chroma(cfg: dict, workers: int = 8):
    chroma_dir = cfg['data'].get('chroma_dir')
    os.makedirs(chroma_dir, exist_ok=True)
    
    n_fft = cfg['model'].get('mel_n_fft', 1024)
    hop_length = cfg['model'].get('mel_hop_length', 320)
    win_length = cfg['model'].get('mel_win_length', 640)
    
    for split in ["train", "val", "test"]:
        mf = os.path.join(cfg['paths']['manifest_dir'], f"{split}_manifest.txt")
        if not os.path.exists(mf): 
            print(f"[{split}] Manifest missing, skip chroma."); 
            continue
        with open(mf) as f: 
            paths = [l.strip().split('|')[0] for l in f if l.strip()]
        
        tasks = []
        for p in paths:
            try:
                info = torchaudio.info(p)
                target = info.num_frames // hop_length
                if target >= 2:
                    tasks.append((p, chroma_dir, target, n_fft, hop_length, win_length))
            except:
                continue
        
        if not tasks: 
            print(f"[{split}] No valid utts for chroma."); 
            continue
            
        print(f"[{split}] Caching Chroma ({workers} workers)...")
        # 🔑 关键修复：传入 initializer=_init_worker
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
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
    hop_length = cfg['model'].get('mel_hop_length', 320)
    valid = [p for p in tqdm(paths, desc="Scan") if 
             os.path.exists(p) and torchaudio.info(p).num_frames / torchaudio.info(p).sample_rate >= 2.0]
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
                if sr != 16000:
                    w = torchaudio.functional.resample(w, sr, 16000)
                    sr = 16000
                batch.append(w.squeeze())
            if not batch: continue
            pad = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True).to(device)
            inp = ext(pad.cpu().numpy(), sampling_rate=sr, return_tensors="pt", padding=True).to(device)
            out = model(**inp, output_hidden_states=True).hidden_states[layer_idx]
            for b in range(len(batch)):
                vl = min(out.shape[1], int(np.ceil(len(batch[b]) / hop_length)))
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

def run_tokens(cfg: dict, gpu: int = 0):
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
    hop_length = cfg['model'].get('mel_hop_length', 320)
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
                    vl = min(feats.shape[1], int(np.ceil(len(batch[b]) / hop_length)))
                    if vl <= 0: continue
                    np.save(out, km.predict(feats[b, :vl].cpu().numpy()))

def run_cpu(cfg: dict, workers: int = 8):
    print("\n[1/3] Generating manifests...")
    run_manifest(cfg)
    
    print("\n[2/3] Extracting F0 (abs+log)...")
    run_f0(cfg, workers)
    
    print("\n[3/3] Extracting Chroma...")
    run_chroma(cfg, workers)
    
    print("\n✅ CPU 预处理完成！")

def run_gpu(cfg: dict, gpu: int = 0):
    print("\n[1/2] Training K-Means (GPU)...")
    run_kmeans(cfg, gpu)
    
    print("\n[2/2] Caching tokens (GPU)...")
    run_tokens(cfg, gpu)
    
    print("\n✅ GPU 预处理完成！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs.yaml')
    parser.add_argument('--mode', choices=['manifest', 'f0', 'chroma', 'kmeans', 'tokens', 'cpu', 'gpu'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    seed = cfg.get('random_seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if args.mode == 'manifest': run_manifest(cfg)
    elif args.mode == 'f0': run_f0(cfg, args.workers)
    elif args.mode == 'chroma': run_chroma(cfg, args.workers)
    elif args.mode == 'kmeans': run_kmeans(cfg, args.gpu)
    elif args.mode == 'tokens': run_tokens(cfg, args.gpu)
    elif args.mode == 'cpu': run_cpu(cfg, args.workers)
    elif args.mode == 'gpu': run_gpu(cfg, args.gpu)

if __name__ == "__main__":
    main()
