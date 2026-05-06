import argparse
import os
import shutil
import yaml
import torch
import torch.nn.functional as F
import subprocess
import io
import torchaudio
from pathlib import Path
from tqdm import tqdm

from system import AnonSystem
from utils import save_audio, compute_mel, normalize_audio, get_stft_params, setup_seed

def read_kaldi_format(filename):
    """读取 Kaldi 格式的文件 (如 wav.scp, utt2spk)"""
    key_list = []
    value_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            splitted_line = line.split()
            if len(splitted_line) == 2:
                key_list.append(splitted_line[0].strip())
                value_list.append(splitted_line[1].strip())
            elif len(splitted_line) > 2:
                key_list.append(splitted_line[0].strip())
                value_list.append(' '.join([x.strip() for x in splitted_line[1:]]))
    return dict(zip(key_list, value_list))

def load_wav_from_scp(wav_path_or_cmd, target_sr: int = 16000):
    """处理 Kaldi wav.scp 中的路径或管道命令并返回单声道张量 [1, T]"""
    if isinstance(wav_path_or_cmd, list):
        wav_path_or_cmd = " ".join(str(x) for x in wav_path_or_cmd)

    if wav_path_or_cmd.strip().endswith("|"):
        devnull = open(os.devnull, "w")
        try:
            wav_read_process = subprocess.Popen(
                wav_path_or_cmd.strip()[:-1], stdout=subprocess.PIPE, shell=True, stderr=devnull
            )
            sample, sr = torchaudio.backend.soundfile_backend.load(
                io.BytesIO(wav_read_process.communicate()[0])
            )
        except Exception as e:
            raise IOError("Error processing wav file: {}\n{}".format(wav_path_or_cmd, e))
    else:
        sample, sr = torchaudio.backend.soundfile_backend.load(wav_path_or_cmd)

    if sample.dim() > 1 and sample.shape[0] > 1:
        sample = sample.mean(dim=0, keepdim=True)
    elif sample.dim() == 1:
        sample = sample.unsqueeze(0)

    if sr != target_sr:
        sample = torchaudio.functional.resample(sample, sr, target_sr)

    return sample

def generate_anon_output(model, wav, alpha, vctk_pool, device, num_candidates: int):
    """批量匿名化推理，wav: [B, 1, T]"""
    with torch.inference_mode():
        bsz = wav.shape[0]
        wav = wav.reshape(bsz, 1, -1)
        mel_params = get_stft_params(model.cfg, prefix='mel')
        mel = compute_mel(wav.squeeze(1), model.cfg['model']['n_mels'],
                          model.cfg['model']['sample_rate'], **mel_params).unsqueeze(1)
        feat = model.enc(wav)
        s_orig = model.spk_enc(mel)
        r1 = feat - s_orig.unsqueeze(1)
        recon, _, _, _ = model.bottleneck(r1)

        pool_mean = vctk_pool.mean(dim=0, keepdim=True)
        pool_std = vctk_pool.std(dim=0, keepdim=True)
        anon_vecs = []
        for i in range(bsz):
            n_select = min(num_candidates, vctk_pool.size(0))
            pool_idx = torch.randperm(vctk_pool.size(0), device=device)[:n_select]
            s_bar = vctk_pool[pool_idx].mean(dim=0, keepdim=True)
            s_hat = torch.randn(1, model.cfg['model']['dimension'], device=device)
            s_hat = s_hat * pool_std + pool_mean
            s_anon = alpha * s_bar + (1.0 - alpha) * s_hat
            s_orig_norm = torch.linalg.vector_norm(s_orig[i:i+1], dim=-1, keepdim=True)
            s_anon_norm = torch.linalg.vector_norm(s_anon, dim=-1, keepdim=True)
            anon_vecs.append(s_anon * (s_orig_norm / (s_anon_norm + 1e-8)))

        s_anon = torch.cat(anon_vecs, dim=0)
        recon_anon = recon + s_anon.unsqueeze(1)
        return model.dec(recon_anon)

def process_dataset(dataset_name, dataset_path, out_dir, anon_suffix, model, cfg, vctk_pool, alpha, num_candidates, device, batch_size):
    """处理单个 VPC 数据集，遵循 Kaldi 格式"""
    in_dir = Path(dataset_path)
    if not in_dir.exists() or not (in_dir / "wav.scp").exists():
        print(f"⚠️ 找不到数据集 {dataset_name} 或其 wav.scp 文件：{in_dir}")
        return
            
    # 输出目录名加上 suffix，例如: libri_dev_npu
    out_dataset_dir = Path(out_dir) / f"{dataset_name}{anon_suffix}"
    out_wav_dir = out_dataset_dir / "wav"
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    
    # ───────── 1. 拷贝 Kaldi 辅助文件 ─────────
    # 需要拷贝的文件清单 (不拷贝 wav 目录和旧的 wav.scp)
    files_to_copy = ['text', 'utt2spk', 'spk2utt', 'trials_f', 'trials_m', 'enrolls']
    for f in files_to_copy:
        src_f = in_dir / f
        if src_f.exists():
            shutil.copy2(src_f, out_dataset_dir / f)
            
    # ───────── 2. 读取 wav.scp 并准备处理 ─────────
    wav_scp_path = in_dir / "wav.scp"
    scp_dict = read_kaldi_format(str(wav_scp_path))
    
    print(f"\n📂 开始处理数据集: {dataset_name} ({len(scp_dict)} 个音频段)")
    print(f"   输出路径: {out_dataset_dir}")
    
    hop_length = cfg['model'].get('hop_length', 320)
    success, fail = 0, 0
    new_scp_lines = []
    
    items = list(scp_dict.items())
    for start in tqdm(range(0, len(items), batch_size), desc=f"Processing {dataset_name}", unit="batch"):
        batch_items = items[start:start + batch_size]
        batch_wavs, batch_meta = [], []

        for utid, wav_path_or_cmd in batch_items:
            try:
                out_wav_path = out_wav_dir / f"{utid}.wav"
                wav = load_wav_from_scp(
                    wav_path_or_cmd,
                    target_sr=cfg['model'].get('sample_rate', 16000),
                ).to(device)
                orig_len = wav.shape[-1]
                pad_len = (hop_length - orig_len % hop_length) % hop_length
                if pad_len > 0:
                    wav = F.pad(wav, (0, pad_len), mode='reflect')
                batch_wavs.append(wav.squeeze(0))
                batch_meta.append((utid, out_wav_path, orig_len))
            except Exception as e:
                tqdm.write(f"❌ 失败 {utid}: {e}")
                fail += 1

        if not batch_wavs:
            continue

        wav_batch = torch.nn.utils.rnn.pad_sequence(batch_wavs, batch_first=True).unsqueeze(1)
        wav_anon_batch = generate_anon_output(model, wav_batch, alpha, vctk_pool, device, num_candidates).float()

        for i, (utid, out_wav_path, orig_len) in enumerate(batch_meta):
            wav_anon = wav_anon_batch[i:i+1, :orig_len]
            save_audio(normalize_audio(wav_anon.cpu()), out_wav_path)
            new_scp_lines.append(f"{utid} {out_wav_path.absolute()}\n")
            success += 1
            
    # ───────── 3. 写入新的 wav.scp ─────────
    with open(out_dataset_dir / "wav.scp", 'w', encoding='utf-8') as f:
        f.writelines(new_scp_lines)
        
    print(f"✅ {dataset_name} 完成 | 成功: {success} | 失败: {fail}\n")

def main():
    parser = argparse.ArgumentParser(description="生成 VPC 2024 评测所需的匿名化音频")
    parser.add_argument('--ckpt', required=True, help='训练检查点路径 (.ckpt)')
    parser.add_argument('--config', required=True, help='训练/推理配置文件路径 (.yaml)')
    parser.add_argument('--pool', required=True, help='说话人特征池路径 (.pt 文件)')
    parser.add_argument('--vpc_data_dir', default='data', 
                        help='VPC2024 data 目录的路径 (包含 libri_dev, libri_test 等)')
    parser.add_argument('--out_dir',
                        default='../Voice-Privacy-Challenge-2024/data', 
                        help='生成音频的输出根目录')
    parser.add_argument('--anon_suffix', required=True, 
                        help='匿名化数据集后缀名，例如 _ours 将生成 libri_dev_ours')
    parser.add_argument('--datasets', nargs='+', 
                        default=[
                            'libri_dev_enrolls', 
                            'libri_dev_trials_f', 
                            'libri_dev_trials_m', 
                            'libri_test_enrolls', 
                            'libri_test_trials_f', 
                            'libri_test_trials_m', 
                            'IEMOCAP_dev', 
                            'IEMOCAP_test', 
                            'train-clean-360'
                        ], 
                        help='需要处理的 VPC 数据集名称列表')
    parser.add_argument('--condition', type=int, choices=[3, 4], default=3, help='匿名化条件: 3(α=0.9) 或 4(α=0.8)')
    parser.add_argument('--num_candidates', type=int, default=None, help='匿名化候选说话人数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子；不传则使用配置文件中的 random_seed')
    parser.add_argument('--batch_size', type=int, default=32, help='推理 batch size，用于提升 GPU 利用率')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='推理设备')
    args = parser.parse_args()

    # ───────── 1. 模型与配置加载 ─────────
    print(f"🚀 初始化 VPC 2024 评测生成管道")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get('random_seed', 42) if args.seed is None else args.seed
    setup_seed(seed)
    print(f"🎲 随机种子: {seed}")

    print(f"🔹 加载检查点: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    num_speakers = ckpt['state_dict']['l_spk.clf.weight'].shape[0]
    
    model = AnonSystem.load_from_checkpoint(
        args.ckpt, cfg=cfg, num_speakers=num_speakers, strict=False
    ).to(args.device)
    model.eval()

    # 🔧 移除权重归一化以加速推理
    if hasattr(model, 'remove_weight_norm'):
        model.remove_weight_norm()
        print("⚡ 已移除权重归一化，提升推理速度")

    print(f"🔹 加载说话人池: {args.pool}")
    
    print(f"🔹 加载说话人池: {args.pool}")
    if not Path(args.pool).exists():
        raise FileNotFoundError(f"❌ 找不到说话人池文件: {args.pool}")
    vctk_pool = torch.load(args.pool, map_location=args.device)
        
    alpha = cfg['anonymization'][f'alpha_cond{args.condition}']
    num_candidates = args.num_candidates if args.num_candidates is not None else cfg['anonymization'].get('num_candidates', 20)
    
    print(f"✅ 模型就绪 | Condition {args.condition} (α={alpha}) | Candidates: {num_candidates} | Device: {args.device}")
    
    # ───────── 2. 遍历并处理各个数据集 ─────────
    os.chdir("../Voice-Privacy-Challenge-2024")
    vpc_data_dir = Path(args.vpc_data_dir)
    
    for dataset_name in args.datasets:
        dataset_path = vpc_data_dir / dataset_name
        if not dataset_path.exists():
            print(f"⚠️ 跳过 {dataset_name}: 路径不存在 ({dataset_path})")
            continue
            
        process_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            out_dir=args.out_dir,
            anon_suffix=args.anon_suffix,
            model=model,
            cfg=cfg,
            vctk_pool=vctk_pool,
            alpha=alpha,
            num_candidates=num_candidates,
            device=args.device,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()
