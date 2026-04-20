# evaluate.py
import yaml, torch, json, os
from datasets import VPDataset
from system import AnonSystem
from utils import compute_mel, save_audio

def generate_anonymized_audio(model, wav, alpha, vctk_pool, device):
    model.eval()
    with torch.no_grad():
        feat = model.enc(wav)
        mel = compute_mel(wav, model.cfg['model']['n_mels'], 
                          model.cfg['model']['sample_rate'], 
                          model.cfg['model']['mel_hop_length'])
        spk = model.spk_enc(mel)
        r1 = feat - spk.unsqueeze(1)
        recon, q1, q2, _ = model.bottleneck(r1)
        
        pool = vctk_pool.to(device)
        idx = torch.randperm(pool.shape[0])[:model.cfg['anonymization']['num_candidates']]
        s_avg = pool[idx].mean(0)
        s_rand = torch.randn(model.cfg['anonymization']['anon_dim'], device=device)
        s_anon = alpha * s_avg + (1 - alpha) * s_rand
        
        # ✅ 修复：显式对齐广播维度 [B, T, D] - [1, 1, D]
        s_anon = s_anon.unsqueeze(0).unsqueeze(1)
        r1_anon = feat - s_anon
        recon_anon, _, _, _ = model.bottleneck(r1_anon)
        return model.dec(recon_anon.transpose(1,2))

def main():
    with open("configs.yaml") as f: cfg = yaml.safe_load(f)
    test_mf = os.path.join(cfg['paths']['manifest_dir'], "test_manifest.txt")
    if not os.path.exists(test_mf):
        import preprocess; preprocess.run_manifest(cfg)
        
    ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], "last.ckpt")
    if not os.path.exists(ckpt_path):
        ckpts = [f for f in os.listdir(cfg['paths']['checkpoint_dir']) if f.endswith('.ckpt')]
        ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], sorted(ckpts)[-1])
        
    num_spk = len(json.load(open(cfg['paths']['spk_map_path'])))
    model = AnonSystem.load_from_checkpoint(ckpt_path, cfg=cfg, num_speakers=num_spk, strict=False).cuda()
    vctk_pool = torch.load(cfg['anonymization']['vctk_pool_path'], map_location='cuda')
    ds = VPDataset(test_mf, cfg, False)
    out_dir = cfg['paths']['test_out_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    print("🔍 Generating anonymized audio for test set...")
    for item in ds:
        wav = item['wav'].cuda().unsqueeze(0)
        uid = item['uid']
        wav_c3 = generate_anonymized_audio(model, wav, cfg['anonymization']['alpha_cond3'], vctk_pool, 'cuda')
        wav_c4 = generate_anonymized_audio(model, wav, cfg['anonymization']['alpha_cond4'], vctk_pool, 'cuda')
        save_audio(wav_c3.squeeze().cpu(), os.path.join(out_dir, f"{uid}_cond3.wav"))
        save_audio(wav_c4.squeeze().cpu(), os.path.join(out_dir, f"{uid}_cond4.wav"))
    print("✅ Done. Output:", out_dir)

if __name__ == "__main__":
    main()