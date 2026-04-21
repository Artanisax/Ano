# anonymize.py
import yaml, torch, argparse, os, glob
from pathlib import Path
from tqdm import tqdm
from system import AnonSystem
from utils import load_audio, save_audio, compute_mel
import torch.nn.functional as F

def generate_anonymized_audio(model, wav, alpha, vctk_pool, device):
    """VPC 2024 标准匿名化推理管线 (严格对齐 Eq.7)"""
    with torch.no_grad():
        mel = compute_mel(wav, model.cfg['model']['n_mels'], 
                          model.cfg['model']['sample_rate'], 
                          model.cfg['model']['mel_hop_length'])
        feat = model.enc(wav)
        
        # 1. 随机抽取 20 个候选说话人并平均 (s̄)
        pool_idx = torch.randperm(vctk_pool.size(0), device=device)[:20]
        s_bar = vctk_pool[pool_idx].mean(dim=0, keepdim=True)
        
        # 2. 生成高斯随机身份 (ŝ)
        s_hat = torch.randn(1, model.cfg['model']['speaker']['dim'], device=device)
        
        # 3. 加权融合 (Eq.7)
        s_anon = alpha * s_bar + (1.0 - alpha) * s_hat
        
        # 4. 串行解耦与重建
        r1 = feat - s_anon.unsqueeze(1)
        recon, _, _, _ = model.bottleneck(r1)
        wav_anon = model.dec(recon.transpose(1, 2))
        return wav_anon

def main():
    parser = argparse.ArgumentParser(description="VPC 2024 语音匿名化推理脚本")
    parser.add_argument('--ckpt', required=True, help='训练检查点路径 (.ckpt)')
    parser.add_argument('--input', required=True, help='输入音频文件 或 包含音频的目录')
    parser.add_argument('--output', default=None, help='输出路径 (文件/目录)。若为None则自动生成')
    parser.add_argument('--condition', type=int, choices=[3, 4], default=3, help='匿名化条件: 3(α=0.9) 或 4(α=0.8)')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='推理设备')
    parser.add_argument('--ext', nargs='+', default=['.wav', '.flac'], help='支持的音频扩展名')
    args = parser.parse_args()

    # ───────── 1. 路径解析与自动补全 ─────────
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ 输入路径不存在: {args.input}")

    if args.output is None:
        if input_path.is_file():
            base = input_path.stem
            args.output = str(input_path.parent / f"{base}_anon_cond{args.condition}.wav")
        elif input_path.is_dir():
            args.output = str(input_path / f"anon_cond{args.condition}")
        else:
            raise ValueError("输入必须是文件或目录")

    # ───────── 2. 模型与配置加载 (仅执行一次) ─────────
    with open("configs.yaml") as f: 
        cfg = yaml.safe_load(f)
        
    print(f"🔹 加载检查点: {args.ckpt}")
    model = AnonSystem.load_from_checkpoint(args.ckpt, cfg=cfg, strict=False)
    model.to(args.device)
    model.eval()  # 锁定 BN/Dropout 行为
    
    print(f"🔹 加载说话人池: {cfg['anonymization']['vctk_pool_path']}")
    vctk_pool = torch.load(cfg['anonymization']['vctk_pool_path'], map_location=args.device)
    if vctk_pool.dim() == 2:
        vctk_pool = F.normalize(vctk_pool, dim=1, p=2)  # 防御性归一化
        
    alpha = cfg['anonymization'][f'alpha_cond{args.condition}']
    print(f"✅ 环境就绪 | Condition {args.condition} (α={alpha}) | Device: {args.device}\n")

    # ───────── 3. 推理执行 ─────────
    if input_path.is_file():
        # 单文件模式
        print(f"🎧 处理单文件: {input_path.name}")
        wav = load_audio(str(input_path)).to(args.device).unsqueeze(0)
        anon_wav = generate_anonymized_audio(model, wav, alpha, vctk_pool, args.device)
        save_audio(anon_wav.squeeze().cpu(), args.output)
        print(f"✅ 已保存: {args.output}\n")
        
    elif input_path.is_dir():
        # 目录批量模式
        os.makedirs(args.output, exist_ok=True)
        audio_files = []
        for ext in args.ext:
            audio_files.extend(glob.glob(str(input_path / f"*{ext}")))
            
        if not audio_files:
            print(f"⚠️ 目录中未找到支持的音频格式 {args.ext}")
            return
            
        print(f"📁 发现 {len(audio_files)} 个音频文件，开始批量处理...\n")
        success, fail = 0, 0
        for f_path in tqdm(audio_files, desc="Anonymizing", unit="file"):
            try:
                fname = Path(f_path).stem
                out_path = os.path.join(args.output, f"{fname}_anon.wav")
                
                wav = load_audio(f_path).to(args.device).unsqueeze(0)
                anon_wav = generate_anonymized_audio(model, wav, alpha, vctk_pool, args.device)
                save_audio(anon_wav.squeeze().cpu(), out_path)
                success += 1
            except Exception as e:
                tqdm.write(f"❌ 失败 {Path(f_path).name}: {e}")
                fail += 1
                
        print(f"\n🎉 批量完成 | 成功: {success} | 失败: {fail} | 输出目录: {args.output}")

if __name__ == "__main__":
    main()