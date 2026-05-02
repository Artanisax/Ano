import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import List

import joblib
import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

from utils import load_audio


@dataclass
class UtteranceCheck:
    uid: str
    wav_path: str
    cached_path: str
    wav_samples: int
    pred_frames: int
    cached_frames: int
    exact_match: bool
    mismatch_frames: int
    mismatch_ratio: float
    first_mismatch_index: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="校验缓存 token 是否与当前 WavLM 指定层 + KMeans 重新聚类结果一致"
    )
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="要校验的数据划分",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="可选，直接指定 manifest 文件；传入后将覆盖 --split",
    )
    parser.add_argument(
        "--token_dir",
        default=None,
        help="可选，覆盖配置中的 data.token_dir",
    )
    parser.add_argument(
        "--kmeans_path",
        default=None,
        help="可选，覆盖配置中的 paths.kmeans_path",
    )
    parser.add_argument(
        "--wavlm_model",
        default=None,
        help="可选，覆盖配置中的 preprocess.wavlm_model",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="可选，覆盖配置中的 preprocess.wavlm_layer_idx",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="重新提取特征时的 batch size",
    )
    parser.add_argument(
        "--max_utts",
        type=int,
        default=None,
        help="最多抽查多少条；不传则检查全部",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="抽样随机种子",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备",
    )
    parser.add_argument(
        "--report_json",
        default=None,
        help="可选，将详细结果保存到 JSON 文件",
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=10,
        help="最多展示多少条不一致样例",
    )
    return parser.parse_args()


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_manifests(cfg: dict, split: str, manifest: str | None) -> List[str]:
    if manifest is not None:
        return [manifest]

    manifest_dir = cfg["paths"]["manifest_dir"]
    if split == "all":
        return [
            os.path.join(manifest_dir, "train_manifest.txt"),
            os.path.join(manifest_dir, "val_manifest.txt"),
            os.path.join(manifest_dir, "test_manifest.txt"),
        ]
    return [os.path.join(manifest_dir, f"{split}_manifest.txt")]


def read_manifest(manifest_path: str) -> List[str]:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest 不存在: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return [line.strip().split("|")[0] for line in f if line.strip()]


def sample_paths(paths: List[str], max_utts: int | None, seed: int) -> List[str]:
    if max_utts is None or max_utts <= 0 or len(paths) <= max_utts:
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, max_utts)


def compare_tokens(pred: np.ndarray, cached: np.ndarray) -> tuple[bool, int, float, int]:
    pred = np.asarray(pred).astype(np.int64)
    cached = np.asarray(cached).astype(np.int64)

    if pred.shape[0] != cached.shape[0]:
        min_len = min(pred.shape[0], cached.shape[0])
        mismatch_frames = abs(pred.shape[0] - cached.shape[0])
        if min_len > 0:
            frame_mismatch = int(np.count_nonzero(pred[:min_len] != cached[:min_len]))
            mismatch_frames += frame_mismatch
            mismatch_ratio = mismatch_frames / max(pred.shape[0], cached.shape[0])
            mismatch_idx = int(np.flatnonzero(pred[:min_len] != cached[:min_len])[0]) if frame_mismatch > 0 else min_len
        else:
            mismatch_ratio = 1.0
            mismatch_idx = 0
        return False, mismatch_frames, float(mismatch_ratio), mismatch_idx

    diff = pred != cached
    mismatch_frames = int(np.count_nonzero(diff))
    if mismatch_frames == 0:
        return True, 0, 0.0, -1
    first_mismatch = int(np.flatnonzero(diff)[0])
    mismatch_ratio = mismatch_frames / pred.shape[0]
    return False, mismatch_frames, float(mismatch_ratio), first_mismatch


def verify_manifest(
    manifest_path: str,
    token_dir: str,
    model: WavLMModel,
    extractor: AutoFeatureExtractor,
    km,
    layer_idx: int,
    hop_length: int,
    sample_rate: int,
    batch_size: int,
    max_utts: int | None,
    seed: int,
    device: str,
) -> tuple[dict, List[UtteranceCheck]]:
    wav_paths = read_manifest(manifest_path)
    wav_paths = sample_paths(wav_paths, max_utts, seed)

    items = []
    missing_cache = []
    for wav_path in wav_paths:
        uid = os.path.splitext(os.path.basename(wav_path))[0]
        cached_path = os.path.join(token_dir, f"{uid}.npy")
        if not os.path.exists(cached_path):
            missing_cache.append((uid, cached_path))
            continue
        items.append((uid, wav_path, cached_path))

    print(f"\n[Check] manifest={manifest_path}")
    print(f"  总条数: {len(wav_paths)} | 可校验: {len(items)} | 缺失缓存: {len(missing_cache)}")

    checks: List[UtteranceCheck] = []
    exact_matches = 0
    mismatch_utts = 0
    total_frames = 0
    total_mismatch_frames = 0

    with torch.no_grad():
        for start in tqdm(
            range(0, len(items), batch_size),
            desc=f"Verifying {os.path.basename(manifest_path)}",
            unit="batch",
            leave=False,
        ):
            batch_items = items[start:start + batch_size]
            batch_wavs = []
            batch_lengths = []

            for _, wav_path, _ in batch_items:
                wav = load_audio(wav_path, sr=sample_rate).squeeze(0)
                batch_wavs.append(wav)
                batch_lengths.append(int(wav.shape[0]))

            if not batch_wavs:
                continue

            pad = torch.nn.utils.rnn.pad_sequence(batch_wavs, batch_first=True).to(device)
            inp = extractor(
                pad.cpu().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            ).to(device)
            feats = model(**inp, output_hidden_states=True).hidden_states[layer_idx]

            for i, (uid, wav_path, cached_path) in enumerate(batch_items):
                pred_frames = min(feats.shape[1], int(math.ceil(batch_lengths[i] / hop_length)))
                pred = km.predict(feats[i, :pred_frames].detach().cpu().numpy())
                cached = np.load(cached_path)
                exact, mismatch_frames, mismatch_ratio, first_idx = compare_tokens(pred, cached)

                checks.append(
                    UtteranceCheck(
                        uid=uid,
                        wav_path=wav_path,
                        cached_path=cached_path,
                        wav_samples=batch_lengths[i],
                        pred_frames=int(pred.shape[0]),
                        cached_frames=int(cached.shape[0]),
                        exact_match=exact,
                        mismatch_frames=mismatch_frames,
                        mismatch_ratio=mismatch_ratio,
                        first_mismatch_index=first_idx,
                    )
                )

                total_frames += max(int(pred.shape[0]), int(cached.shape[0]))
                total_mismatch_frames += mismatch_frames
                if exact:
                    exact_matches += 1
                else:
                    mismatch_utts += 1

    summary = {
        "manifest": manifest_path,
        "checked_utts": len(checks),
        "missing_cache_utts": len(missing_cache),
        "exact_match_utts": exact_matches,
        "mismatch_utts": mismatch_utts,
        "utt_exact_match_rate": exact_matches / len(checks) if checks else 0.0,
        "total_frames_compared": total_frames,
        "total_mismatch_frames": total_mismatch_frames,
        "frame_exact_match_rate": 1.0 - (total_mismatch_frames / total_frames) if total_frames > 0 else 0.0,
        "missing_cache_examples": [
            {"uid": uid, "cached_path": cached_path}
            for uid, cached_path in missing_cache[:10]
        ],
    }
    return summary, checks


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    token_dir = args.token_dir or cfg["data"]["token_dir"]
    kmeans_path = args.kmeans_path or cfg["paths"]["kmeans_path"]
    wavlm_model = args.wavlm_model or cfg["preprocess"]["wavlm_model"]
    layer_idx = args.layer if args.layer is not None else cfg["preprocess"]["wavlm_layer_idx"]
    hop_length = cfg["model"].get("hop_length", 320)
    sample_rate = cfg["model"].get("sample_rate", 16000)
    manifests = resolve_manifests(cfg, args.split, args.manifest)

    print("[Config]")
    print(f"  config      : {args.config}")
    print(f"  token_dir   : {token_dir}")
    print(f"  kmeans_path : {kmeans_path}")
    print(f"  wavlm_model : {wavlm_model}")
    print(f"  wavlm_layer : {layer_idx}")
    print(f"  hop_length  : {hop_length}")
    print(f"  sample_rate : {sample_rate}")
    print(f"  device      : {args.device}")

    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(f"KMeans 模型不存在: {kmeans_path}")
    if not os.path.isdir(token_dir):
        raise FileNotFoundError(f"token 缓存目录不存在: {token_dir}")

    km = joblib.load(kmeans_path)
    extractor = AutoFeatureExtractor.from_pretrained(wavlm_model)
    model = WavLMModel.from_pretrained(wavlm_model).to(args.device).eval()

    all_summaries = []
    all_checks: List[UtteranceCheck] = []

    for manifest_path in manifests:
        summary, checks = verify_manifest(
            manifest_path=manifest_path,
            token_dir=token_dir,
            model=model,
            extractor=extractor,
            km=km,
            layer_idx=layer_idx,
            hop_length=hop_length,
            sample_rate=sample_rate,
            batch_size=args.batch_size,
            max_utts=args.max_utts,
            seed=args.seed,
            device=args.device,
        )
        all_summaries.append(summary)
        all_checks.extend(checks)

        print("[Summary]")
        print(f"  checked_utts          : {summary['checked_utts']}")
        print(f"  missing_cache_utts    : {summary['missing_cache_utts']}")
        print(f"  exact_match_utts      : {summary['exact_match_utts']}")
        print(f"  mismatch_utts         : {summary['mismatch_utts']}")
        print(f"  utt_exact_match_rate  : {summary['utt_exact_match_rate']:.6f}")
        print(f"  total_frames_compared : {summary['total_frames_compared']}")
        print(f"  total_mismatch_frames : {summary['total_mismatch_frames']}")
        print(f"  frame_exact_match_rate: {summary['frame_exact_match_rate']:.6f}")

        mismatch_examples = [c for c in checks if not c.exact_match][:args.show_examples]
        if mismatch_examples:
            print("[Mismatch Examples]")
            for item in mismatch_examples:
                print(
                    f"  uid={item.uid} | pred_frames={item.pred_frames} | "
                    f"cached_frames={item.cached_frames} | mismatch_frames={item.mismatch_frames} | "
                    f"mismatch_ratio={item.mismatch_ratio:.6f} | first_mismatch={item.first_mismatch_index}"
                )

    if args.report_json is not None:
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        payload = {
            "config": args.config,
            "token_dir": token_dir,
            "kmeans_path": kmeans_path,
            "wavlm_model": wavlm_model,
            "wavlm_layer": layer_idx,
            "summaries": all_summaries,
            "checks": [asdict(c) for c in all_checks],
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"\n[Saved] 详细报告已写入: {args.report_json}")


if __name__ == "__main__":
    main()
