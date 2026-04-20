# extract_vctk_pool.py
import os, yaml, torch, torchaudio, glob, argparse, numpy as np
from tqdm import tqdm
from models import SpeakerEncoder
from utils import compute_mel, setup_logger

def load_speaker_encoder(ckpt_path, cfg):
    spk_cfg = {**cfg['model']['speaker'], 'n_mels': cfg['model']['n_mels']}
    encoder = SpeakerEncoder(spk_cfg).eval()
    state_dict = torch.load(ckpt_path, map_location='cpu')
    raw_dict = state_dict.get('state_dict', state_dict)
    filtered_dict = {k[len('spk_enc.'):]: v for k, v in raw_dict.items() if k.startswith('spk_enc.')}
    encoder.load_state_dict(filtered_dict, strict=False)
    return encoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--vctk_dir', required=True)
    parser.add_argument('--output', default='data/vctk_speaker_vectors.pt')
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    
    encoder = load_speaker_encoder(args.ckpt, cfg).cuda()
    spk_dirs = sorted([d for d in os.listdir(args.vctk_dir) if os.path.isdir(os.path.join(args.vctk_dir, d)) and d.startswith('p')])
    print(f"📊 发现 {len(spk_dirs)} 个 VCTK 说话人")
    
    speaker_vectors = []
    for spk_id in tqdm(spk_dirs, desc="VCTK Pool Extraction"):
        audios = glob.glob(os.path.join(args.vctk_dir, spk_id, "*.wav")) + glob.glob(os.path.join(args.vctk_dir, spk_id, "*.flac"))
        embs = []
        with torch.no_grad():
            for p in audios:
                try:
                    w, sr = torchaudio.load(p)
                    if w.dim()>1: w=w.mean(dim=0, keepdim=True)
                    if sr!=16000: w=torchaudio.functional.resample(w, sr, 16000)
                    mel = compute_mel(w, cfg['model']['n_mels'], 16000, cfg['model']['mel_hop_length']).cuda()
                    embs.append(encoder(mel).cpu())
                except: continue
        if embs: speaker_vectors.append(torch.stack(embs).mean(0))
            
    pool = torch.stack(speaker_vectors)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(pool, args.output)
    print(f"💾 说话人池已保存: {pool.shape}")

if __name__ == "__main__":
    main()