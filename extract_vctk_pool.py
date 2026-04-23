# extract_vctk_pool.py
import os, yaml, torch, torchaudio, glob, argparse
from tqdm import tqdm
from modules import SpeakerEncoder
from utils import compute_mel, get_stft_params
import torch.nn.functional as F

def load_speaker_encoder(ckpt_path, cfg):
    spk_cfg = {**cfg['model']['speaker'], 'n_mels': cfg['model']['n_mels']}
    encoder = SpeakerEncoder(spk_cfg).eval()
    state_dict = torch.load(ckpt_path, map_location='cpu')
    raw_dict = state_dict.get('state_dict', state_dict)
    # 严格过滤 spk_enc. 前缀
    filtered_dict = {k[len('spk_enc.'):]: v for k, v in raw_dict.items() if k.startswith('spk_enc.')}
    encoder.load_state_dict(filtered_dict, strict=True)
    return encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--vctk_dir', default='data/raw/VCTK/VCTK-Corpus-0.92/wav48_silence_trimmed')
    parser.add_argument('--output', default='data/vctk_speaker_vectors.pt')
    args = parser.parse_args()
    
    with open(args.config) as f: cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        embs = []
        
        with torch.no_grad():
            for p in audios:
                try:
                    w, sr = torchaudio.load(p)
                    if w.dim() > 1: w = w.mean(dim=0, keepdim=True)
                    if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
                    
                    w = w.to(device)
                    # ✅ 参数化计算 Mel
                    mel = compute_mel(w, cfg['model']['n_mels'], 16000, **mel_params)
                    mel = mel.unsqueeze(1)  # [1, 1, 80, T] 匹配 2D CNN 输入
                    emb = encoder(mel).squeeze(0).cpu()  # [D]
                    embs.append(emb)
                except Exception as e:
                    tqdm.write(f"⚠️ 跳过 {p}: {e}")
                    continue
                    
        if embs:
            speaker_vectors.append(torch.stack(embs).mean(dim=0))
            
    if not speaker_vectors:
        raise RuntimeError("❌ 未提取到任何说话人向量，请检查 VCTK 目录结构或音频格式")
        
    pool = torch.stack(speaker_vectors)
    # ✅ L2 归一化确保匿名化插值尺度稳定（论文 §3.4）
    pool = F.normalize(pool, dim=1, p=2)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(pool, args.output)
    print(f"💾 说话人池已保存: {pool.shape} | 已 L2 归一化")

if __name__ == "__main__":
    main()