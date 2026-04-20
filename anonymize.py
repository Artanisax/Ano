# anonymize.py
import yaml, torch, argparse, os, json
from system import AnonSystem
from utils import load_audio, save_audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--condition', type=int, choices=[3,4], default=3)
    parser.add_argument('--output', default="anon.wav")
    args = parser.parse_args()
    
    with open("configs.yaml") as f: cfg = yaml.safe_load(f)
    spk_map_path = os.path.join(cfg['paths']['manifest_dir'], "train_manifest_spk_map.json")
    num_spk = len(json.load(open(spk_map_path)))
    model = AnonSystem.load_from_checkpoint(args.ckpt, cfg=cfg, num_speakers=num_spk, strict=False).cuda()
    wav = load_audio(args.input).cuda().unsqueeze(0)
    
    from evaluate import generate_anonymized_audio
    vctk_pool = torch.load(cfg['anonymization']['vctk_pool_path']).cuda()
    alpha = cfg['anonymization'][f'alpha_cond{args.condition}']
    anon = generate_anonymized_audio(model, wav, alpha, vctk_pool, 'cuda')
    save_audio(anon.squeeze().cpu(), args.output)
    print(f"🎧 已保存匿名化音频至 {args.output} (Cond{args.condition})")

if __name__ == "__main__":
    main()