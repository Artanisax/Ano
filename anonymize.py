# anonymize.py
import yaml, torch, argparse, os, glob, json
from pathlib import Path
from tqdm import tqdm
from system import AnonSystem
from utils import load_audio, save_audio, compute_mel, normalize_audio, get_stft_params
import torch.nn.functional as F

def generate_dual_outputs(model, wav, alpha, vctk_pool, device):
    """
    VPC 2024 双输出推理管线 (严格对齐论文 §3.1, §3.4, Figure 1)
    共享 Encoder & Bottleneck，分别加回 s_orig 与 s_anon 进行解码
    wav: [1, 1, T]
    返回: (wav_rec, wav_anon) 均为 [1, T]
    """
    with torch.no_grad():
        # ───────── 1. 提取特征与原始身份 ─────────
        mel_params = get_stft_params(model.cfg, prefix='mel')
        mel = compute_mel(wav, model.cfg['model']['n_mels'], 
                          model.cfg['model']['sample_rate'], **mel_params)
        feat = model.enc(wav)                     # [1, T_feat, 512]
        s_orig = model.spk_enc(mel).view(1, -1)   # [1, 512]
        
        # ───────── 2. 串行解耦 (共享路径) ─────────
        r1 = feat - s_orig.unsqueeze(1)           # [1, T_feat, 512]
        recon, _, _, _ = model.bottleneck(r1)     # recon: [1, T_feat, 512]
        
        # ───────── 3. 重建输出：加回原始身份 ─────────
        recon_rec = recon + s_orig.unsqueeze(1)
        wav_rec = model.dec(recon_rec)
        
        # ───────── 4. 匿名化输出：加回匿名身份 (论文 Eq.7) ─────────
        pool_idx = torch.randperm(vctk_pool.size(0), device=device)[:20]
        s_bar = vctk_pool[pool_idx].mean(dim=0, keepdim=True).view(1, -1)
        s_hat = torch.randn(1, model.cfg['model']['speaker']['dim'], device=device)
        s_anon = alpha * s_bar + (1.0 - alpha) * s_hat  # [1, 512]
        
        recon_anon = recon + s_anon.unsqueeze(1)
        wav_anon = model.dec(recon_anon)
        
        return wav_rec, wav_anon

def main():
    parser = argparse.ArgumentParser(description="VPC 2024 语音匿名化推理脚本 (双输出: 重建 + 匿名)")
    parser.add_argument('--ckpt', required=True, help='训练检查点路径 (.ckpt)')
    parser.add_argument('--input', default='data/raw/LibriSpeech/test-clean', help='输入音频文件 或 包含音频的目录')
    parser.add_argument('--output', default='outputs', help='输出目录路径')
    parser.add_argument('--condition', type=int, choices=[3, 4], default=3, help='匿名化条件: 3(α=0.9) 或 4(α=0.8)')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", help='推理设备')
    parser.add_argument('--ext', nargs='+', default=['.wav', '.flac'], help='支持的音频扩展名')
    args = parser.parse_args()

    # ───────── 1. 路径解析 ─────────
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ 输入路径不存在：{args.input}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ───────── 2. 模型与配置加载 ─────────
    with open("configs.yaml") as f: 
        cfg = yaml.safe_load(f)
        
    print(f"🔹 加载检查点: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    num_speakers = ckpt['state_dict']['l_spk.clf.weight'].shape[0]
    print(f"🔍 检测到训练期说话人数量: {num_speakers}")
    
    model = AnonSystem.load_from_checkpoint(
        args.ckpt, cfg=cfg, num_speakers=num_speakers, strict=False
    ).to(args.device)
    model.eval()
    
    print(f"🔹 加载说话人池: {cfg['anonymization']['vctk_pool_path']}")
    vctk_pool = torch.load(cfg['anonymization']['vctk_pool_path'], map_location=args.device)
    # 防御性归一化
    if vctk_pool.dim() == 2:
        vctk_pool = F.normalize(vctk_pool, dim=1, p=2)
        
    alpha = cfg['anonymization'][f'alpha_cond{args.condition}']
    print(f"✅ 环境就绪 | Condition {args.condition} (α={alpha}) | Device: {args.device}\n")

    # ───────── 3. 推理执行 ─────────
    if input_path.is_file():
        print(f"🎧 处理单文件: {input_path.name}")
        wav = load_audio(str(input_path)).to(args.device).unsqueeze(0)
        
        # 🔧 长度保护：补齐至 hop_length 倍数，防止边界伪影
        hop_length = cfg['model'].get('hop_length', 320)
        orig_len = wav.shape[-1]
        pad_len = (hop_length - orig_len % hop_length) % hop_length
        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len), mode='reflect')
        
        wav_rec, wav_anon = generate_dual_outputs(model, wav, alpha, vctk_pool, args.device)
        
        # 裁剪回原始长度
        wav_rec = wav_rec[..., :orig_len]
        wav_anon = wav_anon[..., :orig_len]
        
        base_name = input_path.stem
        save_audio(normalize_audio(wav_rec.cpu()), out_dir / f"{base_name}_rec.wav")
        save_audio(normalize_audio(wav_anon.cpu()), out_dir / f"{base_name}_anon.wav")
        print(f"✅ 已保存: {out_dir / f'{base_name}_rec.wav'} & {out_dir / f'{base_name}_anon.wav'}\n")
        
    elif input_path.is_dir():
        audio_files = []
        for ext in args.ext:
            audio_files.extend(glob.glob(str(input_path / f"**/*{ext}"), recursive=True))
        audio_files = sorted(list(set(audio_files)))
            
        if not audio_files:
            print(f"⚠️ 目录中未找到支持的音频格式 {args.ext}")
            return
            
        print(f"📁 发现 {len(audio_files)} 个音频文件，开始批量处理...\n")
        success, fail = 0, 0
        hop_length = cfg['model'].get('hop_length', 320)
        
        for f_path in tqdm(audio_files, desc="Processing", unit="file"):
            try:
                fname = Path(f_path).stem
                out_rec_path = out_dir / f"{fname}_rec.wav"
                out_anon_path = out_dir / f"{fname}_anon.wav"
                
                wav = load_audio(f_path).to(args.device).unsqueeze(0)
                # 🔧 长度保护
                orig_len = wav.shape[-1]
                pad_len = (hop_length - orig_len % hop_length) % hop_length
                if pad_len > 0:
                    wav = F.pad(wav, (0, pad_len), mode='reflect')
                
                wav_rec, wav_anon = generate_dual_outputs(model, wav, alpha, vctk_pool, args.device)
                
                # 裁剪回原始长度
                wav_rec = wav_rec[..., :orig_len]
                wav_anon = wav_anon[..., :orig_len]
                
                save_audio(normalize_audio(wav_rec.cpu()), out_rec_path)
                save_audio(normalize_audio(wav_anon.cpu()), out_anon_path)
                success += 1
            except Exception as e:
                tqdm.write(f"❌ 失败 {Path(f_path).name}: {e}")
                fail += 1
                
        print(f"\n🎉 批量完成 | 成功: {success} | 失败: {fail} | 输出目录: {out_dir}")

if __name__ == "__main__":
    main()
