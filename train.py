# train.py
import yaml
import json
import os
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

def main():
    with open("configs.yaml") as f: cfg = yaml.safe_load(f)
    setup_seed(cfg.get('random_seed', 42))
    
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
        num_workers=cfg['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False, 
        num_workers=cfg['training']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    tb_logger = TensorBoardLogger(cfg['paths']['log_dir'], name="Ano")
    ckpt_dir = os.path.join(tb_logger.log_dir, "checkpoints")
    model = AnonSystem(cfg, num_speakers=num_spk)

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
    
    trainer.fit(model, tr_dl, val_dl)
    print("🎉 Training completed.")

if __name__ == "__main__":
    main()
