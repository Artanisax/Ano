# anonymize.py
import yaml, torch, argparse, os, glob, json
import torchaudio
from pathlib import Path
from tqdm import tqdm
from system import AnonSystem
from utils import load_audio, save_audio, compute_mel, normalize_audio, get_stft_params, setup_seed
import torch.nn.functional as F

def generate_dual_outputs(model, wav, alpha, vctk_pool, device, num_candidates: int):
    """批量双输出推理，wav: [B, 1, T]"""
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

        recon_rec = recon + s_orig.unsqueeze(1)
        wav_rec = model.dec(recon_rec)

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
        wav_anon = model.dec(recon_anon)
        return wav_rec, wav_anon

def _estimate_num_frames(path: str) -> int:
    try:
        return int(torchaudio.info(path).num_frames)
    except Exception:
        return -1


def main():
    parser = argparse.ArgumentParser(description="VPC 2024 语音匿名化推理脚本 (双输出: 重建 + 匿名)")
    parser.add_argument('--ckpt', required=True, help='训练检查点路径 (.ckpt)')
    parser.add_argument('--config', required=True, help='训练/推理配置文件路径 (.yaml)')
    parser.add_argument('--pool', required=True, help='说话人特征池路径 (.pt 文件)')
    parser.add_argument('--input', default='data/raw/LibriSpeech/test-clean', help='输入音频文件 或 包含音频的目录')
    parser.add_argument('--output', required=True, help='输出目录路径')
    parser.add_argument('--condition', type=int, choices=[3, 4], default=3, help='匿名化条件: 3(α=0.9) 或 4(α=0.8)')
    parser.add_argument('--num_candidates', type=int, default=None, help='匿名化候选说话人数；不传则使用配置文件')
    parser.add_argument('--seed', type=int, default=None, help='随机种子；不传则使用配置文件中的 random_seed')
    parser.add_argument('--batch_size', type=int, default=8, help='目录推理时的 batch size，用于提升 GPU 利用率')
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
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get('random_seed', 42) if args.seed is None else args.seed
    setup_seed(seed)
    print(f"🎲 随机种子: {seed}")

    print(f"🔹 加载检查点: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    num_speakers = ckpt['state_dict']['l_spk.clf.weight'].shape[0]
    print(f"🔍 检测到训练期说话人数量: {num_speakers}")
    
    model = AnonSystem.load_from_checkpoint(
        args.ckpt, cfg=cfg, num_speakers=num_speakers, strict=False
    ).to(args.device)
    model.eval()
    
    # 🔧 移除权重归一化以加速推理
    if hasattr(model, 'remove_weight_norm'):
        model.remove_weight_norm()
        print("⚡ 已移除权重归一化，提升推理速度")
    
    print(f"🔹 加载说话人池: {args.pool}")
    if not Path(args.pool).exists():
        raise FileNotFoundError(f"❌ 找不到说话人池文件: {args.pool}")
    vctk_pool = torch.load(args.pool, map_location=args.device)
        
    alpha = cfg['anonymization'][f'alpha_cond{args.condition}']
    num_candidates = args.num_candidates
    if num_candidates is None:
        num_candidates = cfg['anonymization'].get('num_candidates', 20)
    if num_candidates <= 0:
        raise ValueError(f"num_candidates 必须 > 0，当前为 {num_candidates}")
    print(f"✅ 环境就绪 | Condition {args.condition} (α={alpha}) | Candidates: {num_candidates} | Device: {args.device}\n")

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
        
        wav_rec, wav_anon = generate_dual_outputs(model, wav, alpha, vctk_pool, args.device, num_candidates)
        
        # 恢复 float32 进行保存
        wav_rec = wav_rec.float()
        wav_anon = wav_anon.float()
        
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

        audio_files = sorted(audio_files, key=_estimate_num_frames, reverse=True)
        print(f"📁 发现 {len(audio_files)} 个音频文件，已按长度排序并开始批量处理...\n")
        success, fail = 0, 0
        hop_length = cfg['model'].get('hop_length', 320)
        
        for start in tqdm(range(0, len(audio_files), args.batch_size), desc="Processing", unit="batch"):
            batch_files = audio_files[start:start + args.batch_size]
            batch_wavs, batch_meta = [], []

            for f_path in batch_files:
                fname = Path(f_path).stem
                out_rec_path = out_dir / f"{fname}_rec.wav"
                out_anon_path = out_dir / f"{fname}_anon.wav"
                wav = load_audio(f_path).to(args.device)
                orig_len = wav.shape[-1]
                pad_len = (hop_length - orig_len % hop_length) % hop_length
                if pad_len > 0:
                    wav = F.pad(wav, (0, pad_len), mode='reflect')
                batch_wavs.append(wav.squeeze(0))
                batch_meta.append((out_rec_path, out_anon_path, orig_len, f_path))

            wav_batch = torch.nn.utils.rnn.pad_sequence(batch_wavs, batch_first=True).unsqueeze(1)
            wav_rec_batch, wav_anon_batch = generate_dual_outputs(model, wav_batch, alpha, vctk_pool, args.device, num_candidates)
            wav_rec_batch = wav_rec_batch.float()
            wav_anon_batch = wav_anon_batch.float()

            for i, (out_rec_path, out_anon_path, orig_len, f_path) in enumerate(batch_meta):
                wav_rec = wav_rec_batch[i:i+1, :orig_len]
                wav_anon = wav_anon_batch[i:i+1, :orig_len]
                save_audio(normalize_audio(wav_rec.cpu()), out_rec_path)
                save_audio(normalize_audio(wav_anon.cpu()), out_anon_path)
                success += 1
                
        print(f"\n🎉 批量完成 | 成功: {success} | 失败: {fail} | 输出目录: {out_dir}")

if __name__ == "__main__":
    main()
