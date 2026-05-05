# train.py
import yaml
import json
import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datasets import VPDataset, collate_fn
from system import AnonSystem
from utils import setup_seed

torch.backends.nnpack.enabled = False

def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, (list, tuple)):
            items[new_key] = str(v)
        else:
            items[new_key] = v
    return items

def _count_parameters(module) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())

def _format_params(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)

def _print_model_size_summary(model: AnonSystem):
    gen_modules = [model.enc, model.spk_enc, model.bottleneck, model.dec]
    gen_params = sum(_count_parameters(m) for m in gen_modules)

    disc_total = _count_parameters(model.disc)
    mpd_params = _count_parameters(getattr(model.disc, "mpd", None))
    msd_params = _count_parameters(getattr(model.disc, "msd", None))
    mstftd_params = _count_parameters(getattr(model.disc, "mstftd", None))

    print("\n[Model Size Summary]")
    print(f"  Generator total : {_format_params(gen_params)} ({gen_params:,})")
    print(f"  Discriminator   : {_format_params(disc_total)} ({disc_total:,})")
    print(f"    MPD           : {_format_params(mpd_params)} ({mpd_params:,})")
    print(f"    MSD           : {_format_params(msd_params)} ({msd_params:,})")
    print(f"    MSTFTD        : {_format_params(mstftd_params)} ({mstftd_params:,})")
    print("")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the anonymization system")
    parser.add_argument(
        "--config",
        default="configs.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--resume_ckpt",
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_seed(cfg.get('random_seed', 42))

    if args.resume_ckpt is not None and not os.path.exists(args.resume_ckpt):
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_ckpt}")
    train_workers = cfg['training'].get('train_num_workers', cfg['training'].get('num_workers', 0))
    val_workers = cfg['training'].get('val_num_workers', cfg['training'].get('num_workers', 0))
    
    os.makedirs(cfg['paths']['manifest_dir'], exist_ok=True)
    train_mf = os.path.join(cfg['paths']['manifest_dir'], "train_manifest.txt")
    val_mf = os.path.join(cfg['paths']['manifest_dir'], "val_manifest.txt")
    train_spk_map_path = os.path.join(cfg['paths']['manifest_dir'], "train_manifest_spk_map.json")
    with open(train_spk_map_path) as f:
        num_spk = len(json.load(f))
        
    train_ds = VPDataset(train_mf, cfg, training=True)
    val_ds = VPDataset(val_mf, cfg, training=False)
    
    tr_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True, 
        num_workers=train_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=train_workers > 0,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False, 
        num_workers=val_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=val_workers > 0,
    )
    
    tb_logger = TensorBoardLogger(cfg['paths']['log_dir'], name="Ano")
    ckpt_dir = os.path.join(tb_logger.log_dir, "checkpoints")
    model = AnonSystem(cfg, num_speakers=num_spk)
    # _print_model_size_summary(model)

    # Ensure full and flattened hyperparameters are persisted at training start.
    os.makedirs(tb_logger.log_dir, exist_ok=True)
    with open(os.path.join(tb_logger.log_dir, "hparams_full.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    with open(os.path.join(tb_logger.log_dir, "hparams_full.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=True, indent=2)
    flat_hparams = _flatten_dict(cfg)
    flat_hparams["num_speakers"] = int(num_spk)
    tb_logger.log_hyperparams(flat_hparams)
    
    trainer = pl.Trainer(
        max_steps=cfg['training']['max_steps'],
        max_epochs=cfg['training'].get('max_epochs', None),
        # accumulate_grad_batches=cfg['training'].get('accumulate_grad_batches', 1), # Automatic gradient accumulation is not supported for manual optimization
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir,
                save_last=True,
                save_top_k=5,
                monitor=cfg['training']['early_stop_monitor'],
                mode="min",
            ),
            EarlyStopping(
                monitor=cfg['training']['early_stop_monitor'],
                patience=cfg['training']['early_stop_patience'], 
                mode=cfg['training']['early_stop_mode'],
                check_finite=True,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=tb_logger,
        precision=cfg['training']['precision'],
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        val_check_interval=cfg['training']['val_check_interval'],
        log_every_n_steps=cfg['training']['log_every_n_steps'],
        enable_progress_bar=True,
        enable_autolog_hparams=True,
    )
    
    if args.resume_ckpt is not None:
        print(f"🔁 Resuming training from checkpoint: {args.resume_ckpt}")

    trainer.fit(model, tr_dl, val_dl, ckpt_path=args.resume_ckpt)
    print("🎉 Training completed.")

if __name__ == "__main__":
    main()
