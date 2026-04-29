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
from utils import save_audio, compute_mel, normalize_audio, get_stft_params

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

def load_wav_from_scp(wav_path_or_cmd):
    """处理 Kaldi wav.scp 中的路径或管道命令并返回张量 [1, T]"""
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
        
    return sample

def generate_anon_output(model, wav, alpha, vctk_pool, device, num_candidates: int):
    """
    仅生成匿名化音频的精简推理管道
    """
    with torch.no_grad():
        # ───────── 1. 提取特征与原始身份 ─────────
        mel_params = get_stft_params(model.cfg, prefix='mel')
        mel = compute_mel(wav, model.cfg['model']['n_mels'], 
                          model.cfg['model']['sample_rate'], **mel_params)
        feat = model.enc(wav)                     # [1, T_feat, 512]
        s_orig = model.spk_enc(mel).view(1, -1)   # [1, 512]
        
        # ───────── 2. 串行解耦 ─────────
        r1 = feat - s_orig.unsqueeze(1)           # [1, T_feat, 512]
        recon, _, _, _ = model.bottleneck(r1)     # recon: [1, T_feat, 512]
        
        # ───────── 3. 匿名化输出 ─────────
        n_select = min(num_candidates, vctk_pool.size(0))
        pool_idx = torch.randperm(vctk_pool.size(0), device=device)[:n_select]
        s_bar = vctk_pool[pool_idx].mean(dim=0, keepdim=True).view(1, -1)
        
        # 高斯扰动
        pool_mean = vctk_pool.mean(dim=0, keepdim=True)
        pool_std = vctk_pool.std(dim=0, keepdim=True)
        s_hat = torch.randn(1, model.cfg['model']['speaker']['dim'], device=device)
        s_hat = s_hat * pool_std + pool_mean
        
        s_anon = alpha * s_bar + (1.0 - alpha) * s_hat  # [1, 512]
        
        # --- 缩放 s_anon 使其与 s_orig 的模长(L2 Norm)一致 ---
        # 避免因为 alpha 混合和采样导致向量能量漂移，影响解码器的分布假设
        s_orig_norm = torch.linalg.vector_norm(s_orig, dim=-1, keepdim=True)
        s_anon_norm = torch.linalg.vector_norm(s_anon, dim=-1, keepdim=True)
        s_anon = s_anon * (s_orig_norm / (s_anon_norm + 1e-8))
        
        recon_anon = recon + s_anon.unsqueeze(1)
        wav_anon = model.dec(recon_anon)
        
        return wav_anon

def process_dataset(dataset_name, dataset_path, out_dir, anon_suffix, model, cfg, vctk_pool, alpha, num_candidates, device):
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
    
    for utid, wav_path_or_cmd in tqdm(scp_dict.items(), desc=f"Processing {dataset_name}"):
        try:
            # 匿名化后的绝对保存路径
            out_wav_path = out_wav_dir / f"{utid}.wav"
            
            # 使用官方类似的处理方式加载音频
            wav = load_wav_from_scp(wav_path_or_cmd).to(device)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            
            # 长度保护
            orig_len = wav.shape[-1]
            pad_len = (hop_length - orig_len % hop_length) % hop_length
            if pad_len > 0:
                wav = F.pad(wav, (0, pad_len), mode='reflect')
                
            wav_anon = generate_anon_output(model, wav, alpha, vctk_pool, device, num_candidates)
            
            # 裁剪回原始长度
            wav_anon = wav_anon[..., :orig_len]
            
            save_audio(normalize_audio(wav_anon.cpu()), out_wav_path)
            
            # 记录新的 wav.scp 条目
            new_scp_lines.append(f"{utid} {out_wav_path.absolute()}\n")
            success += 1
        except Exception as e:
            tqdm.write(f"❌ 失败 {utid}: {e}")
            fail += 1
            
    # ───────── 3. 写入新的 wav.scp ─────────
    with open(out_dataset_dir / "wav.scp", 'w', encoding='utf-8') as f:
        f.writelines(new_scp_lines)
        
    print(f"✅ {dataset_name} 完成 | 成功: {success} | 失败: {fail}\n")

def main():
    parser = argparse.ArgumentParser(description="生成 VPC 2024 评测所需的匿名化音频")
    parser.add_argument('--ckpt', required=True, help='训练检查点路径 (.ckpt)')
    parser.add_argument('--pool', required=True, help='说话人特征池路径 (.pt 文件)')
    parser.add_argument('--vpc_data_dir', default='data', 
                        help='VPC2024 data 目录的路径 (包含 libri_dev, libri_test 等)')
    parser.add_argument('--out_dir',
                        default='outputs/',  
                        # default='../Voice-Privacy-Challenge-2024/data', 
                        help='生成音频的输出根目录')
    parser.add_argument('--anon_suffix', default='_ours', 
                        help='匿名化数据集后缀名，例如 _ours 将生成 libri_dev_ours')
    parser.add_argument('--datasets', nargs='+', 
                        default=[
                            'libri_dev_trials_f', 
                            'libri_dev_trials_m', 
                            'libri_test_trials_f', 
                            'libri_test_trials_m', 
                            'IEMOCAP_dev', 
                            'IEMOCAP_test'
                        ], 
                        help='需要处理的 VPC 数据集名称列表')
    parser.add_argument('--condition', type=int, choices=[3, 4], default=3, help='匿名化条件: 3(α=0.9) 或 4(α=0.8)')
    parser.add_argument('--num_candidates', type=int, default=None, help='匿名化候选说话人数')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='推理设备')
    args = parser.parse_args()

    # ───────── 1. 模型与配置加载 ─────────
    print(f"🚀 初始化 VPC 2024 评测生成管道")
    with open("configs.yaml") as f: 
        cfg = yaml.safe_load(f)
        
    print(f"🔹 加载检查点: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    num_speakers = ckpt['state_dict']['l_spk.clf.weight'].shape[0]
    
    model = AnonSystem.load_from_checkpoint(
        args.ckpt, cfg=cfg, num_speakers=num_speakers, strict=False
    ).to(args.device)
    model.eval()
    
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
            device=args.device
        )

if __name__ == "__main__":
    main()
