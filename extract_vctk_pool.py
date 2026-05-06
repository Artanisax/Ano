# extract_vctk_pool.py
import os, yaml, torch, torchaudio, glob, argparse
from tqdm import tqdm
from modules import SpeakerEncoder
from utils import compute_mel, get_stft_params
import torch.nn.functional as F

def load_speaker_encoder(ckpt_path, cfg):
    encoder = SpeakerEncoder(cfg['model']).eval()
    state_dict = torch.load(ckpt_path, map_location='cpu')
    raw_dict = state_dict.get('state_dict', state_dict)
    # 严格过滤 spk_enc. 前缀
    filtered_dict = {k[len('spk_enc.'):]: v for k, v in raw_dict.items() if k.startswith('spk_enc.')}
    encoder.load_state_dict(filtered_dict, strict=True)
    return encoder


def encode_mel_batches(encoder, mel_list, device, batch_size: int):
    embs = []
    for start in range(0, len(mel_list), batch_size):
        batch = mel_list[start:start + batch_size]
        lengths = [m.shape[-1] for m in batch]
        max_len = max(lengths)
        mel_batch = torch.stack([F.pad(m, (0, max_len - m.shape[-1])) for m in batch], dim=0).to(device)
        mask = torch.arange(max_len, device=device).unsqueeze(0) >= torch.tensor(lengths, device=device).unsqueeze(1)
        batch_embs = encoder(mel_batch, mask=mask).cpu()
        embs.extend(batch_embs.unbind(0))
    return embs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--vctk_dir', default='data/raw/VCTK/VCTK-Corpus-0.92/wav48_silence_trimmed')
    parser.add_argument('--output', default='data/vctk_speaker_vectors.pt')
    parser.add_argument('--batch_size', type=int, default=32, help='SpeakerEncoder 提取说话人向量时的 batch size')
    args = parser.parse_args()
    
    with open(args.config) as f: cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")
    
    encoder = load_speaker_encoder(args.ckpt, cfg).to(device)
    vctk_dir = args.vctk_dir
    
    # 兼容 VCTK 常见目录结构
    spk_dirs = sorted([d for d in os.listdir(vctk_dir) 
                       if os.path.isdir(os.path.join(vctk_dir, d)) and d.startswith('p')])
    print(f"📊 发现 {len(spk_dirs)} 个 VCTK 说话人")
    
    speaker_vectors = []
    mel_params = get_stft_params(cfg, prefix='mel')  # ✅ 从配置读取 STFT 参数
    
    for spk_id in tqdm(spk_dirs, desc="VCTK Pool Extraction"):
        audios = glob.glob(os.path.join(vctk_dir, spk_id, "*.wav")) + \
                 glob.glob(os.path.join(vctk_dir, spk_id, "*.flac"))
        mel_list = []

        with torch.inference_mode():
            for p in audios:
                try:
                    w, sr = torchaudio.load(p)
                    if w.dim() > 1:
                        w = w.mean(dim=0, keepdim=True)
                    if sr != 16000:
                        w = torchaudio.functional.resample(w, sr, 16000)

                    w = w.to(device)
                    mel = compute_mel(w, cfg['model']['n_mels'], 16000, **mel_params).unsqueeze(1).cpu()
                    mel_list.append(mel.squeeze(0))  # [1, 80, T]
                except Exception as e:
                    tqdm.write(f"⚠️ 跳过 {p}: {e}")
                    continue

        if mel_list:
            embs = encode_mel_batches(encoder, mel_list, device, args.batch_size)
            speaker_vectors.append(torch.stack(embs).mean(dim=0))
            
    if not speaker_vectors:
        raise RuntimeError("❌ 未提取到任何说话人向量，请检查 VCTK 目录结构或音频格式")
        
    pool = torch.stack(speaker_vectors)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(pool, args.output)
    print(f"💾 说话人池已保存: {pool.shape}")

if __name__ == "__main__":
    main()
