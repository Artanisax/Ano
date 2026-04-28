# system.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from modules import *
from losses import *
from utils import setup_logger, compute_mel, save_audio, setup_seed, normalize_audio, get_stft_params
from transformers import WavLMModel

class KMeansPredictor:
    def __init__(self, path: str):
        self.centers = torch.tensor(joblib.load(path).cluster_centers_, dtype=torch.float32)  # [K, D_code]
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmin(torch.cdist(x, self.centers.to(x.device)), dim=-1)  # [B*, T_feat]

class AnonSystem(pl.LightningModule):
    def __init__(self, cfg: dict, num_speakers: int):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['num_speakers', 'cfg'])
        self.num_speakers = num_speakers
        
        # 核心编解码模块
        self.enc = SpeechEncoder(cfg['model']['encoder_strides'], cfg['model']['hidden_dim'], cfg['model'].get('lstm_layers', 2))
        self.spk_enc = SpeakerEncoder({**cfg['model']['speaker'], 'n_mels': cfg['model']['n_mels']})
        self.bottleneck = ResidualBottleneck(cfg)
        self.dec = Decoder(cfg['model']['encoder_strides'], cfg['model']['hidden_dim'], cfg['model'].get('lstm_layers', 2))
        self.disc = Discriminator(cfg.get('discriminator', {}))
        
        # 教师模型按需加载（缓存开启时跳过，节省 ~1.5GB 显存）
        self.use_cache = cfg['data']['use_cache']
        if not self.use_cache:
            self.wavlm = WavLMExtractor(cfg['preprocess']['wavlm_model'], cfg['preprocess']['wavlm_layer_idx'])
            self.kmeans = KMeansPredictor(cfg['paths']['kmeans_path'])
        else:
            self.wavlm = None
            self.kmeans = None
            
        # 损失函数
        self.l_spk = SpkDistillLoss(cfg['model']['speaker']['dim'], num_speakers)
        self.l_lin = LinDistillLoss(cfg['model']['bottleneck']['codebook_size'], cfg['model']['bottleneck']['codebook_dim'])
        
        self.f0_type = cfg['losses'].get('f0_type', 'log')
        self.l_emo = EmoDistillLoss(cfg['model']['bottleneck']['codebook_dim'], f0_type=self.f0_type)
        
        self.enable_chroma = cfg['losses'].get('enable_chroma', True)
        if self.enable_chroma:
            self.l_chroma = ChromaDistillLoss(cfg['model']['bottleneck']['codebook_dim'], n_chroma=24)
        
        # ✅ 从 cfg 读取 MR-STFT 配置（支持消融实验）
        self.enable_mrstft = cfg['losses'].get('enable_mrstft', True)
        if self.enable_mrstft:
            mrstft_resolutions = cfg['losses'].get('mrstft_resolutions')
            self.l_mrstft = MultiResolutionSTFTLoss(resolutions=mrstft_resolutions)
        self.l_adv = AdvLoss()
        
        self.automatic_optimization = False

    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad_(flag)

    def get_tokens_dynamic(self, wav_flat: torch.Tensor) -> torch.Tensor:
        if self.wavlm is None or self.kmeans is None:
            raise RuntimeError("教师模型未加载。请确保 data.use_cache=False 或预缓存文件完整。")
        with torch.no_grad():
            feats = self.wavlm(wav_flat)  # [B*, T_wavlm, D_wavlm]
            hop_length = self.cfg['model'].get('hop_length', 320)
            target = wav_flat.shape[-1] // hop_length
            if feats.shape[1] != target:
                feats = F.interpolate(feats.transpose(1, 2), size=target, mode='linear', align_corners=False).transpose(1, 2)
        return self.kmeans.predict(feats)  # [B*, T_feat]

    def _compute_mel_3d(self, wav: torch.Tensor) -> torch.Tensor:
        """辅助函数：计算 Mel 并统一输出 [B, F, T] 维度"""
        mel_params = get_stft_params(self.cfg, prefix='mel')
        return compute_mel(wav, n_mels=self.cfg['model']['n_mels'], 
                          sr=self.cfg['model']['sample_rate'], **mel_params).squeeze(1)

    def forward(self, wav: torch.Tensor) -> tuple:
        # 训练期: [B, 3, 1, T] | 验证期: [B, 1, T]
        is_train = wav.dim() == 4
        
        if is_train:
            wav_main = wav[:, 0]   # [B, 1, T] 主重建路径
            wav_s1   = wav[:, 1]   # [B, 1, T] 蒸馏参考1
            wav_s2   = wav[:, 2]   # [B, 1, T] 蒸馏参考2
        else:
            wav_main = wav         # [B, 1, T] 验证期直接保留
            wav_s1 = wav_s2 = None
        
        # ───────── 主重建路径 ─────────
        mel_params = get_stft_params(self.cfg, prefix='mel')
        mel_main_4d = compute_mel(wav_main, self.cfg['model']['n_mels'], 
                                  self.cfg['model']['sample_rate'], **mel_params)
        feat_main = self.enc(wav_main)                  # [B, T_feat, 512]
        spk_main = self.spk_enc(mel_main_4d)            # [B, 512]
        
        # ✅ 串行解耦：减去原始说话人身份（论文 §3.1, Figure 1）
        r1 = feat_main - spk_main.unsqueeze(1)          # [B, T_feat, 512]
        recon, q1, q2, com = self.bottleneck(r1)        # recon:[B, T_feat, 512]
        
        # 🔑 核心修复：加回原始身份用于重建（严格对齐论文 §3.4 & Eq.7）
        recon_with_spk = recon + spk_main.unsqueeze(1)  # [B, T_feat, 512]
        wav_rec = self.dec(recon_with_spk)  # [B, T_feat, hidden] -> [B, T]
        
        # 🔧 长度保护：裁剪至原始输入长度，防止转置卷积边界伪影
        if wav_rec.shape[-1] != wav_main.shape[-1]:
            wav_rec = wav_rec[..., :wav_main.shape[-1]]

        if is_train:
            # ───────── 蒸馏路径：仅提取 s1, s2 用于说话人一致性约束 ─────────
            mel_s1_4d = compute_mel(wav_s1, self.cfg['model']['n_mels'],
                                    self.cfg['model']['sample_rate'], **mel_params)
            mel_s2_4d = compute_mel(wav_s2, self.cfg['model']['n_mels'],
                                    self.cfg['model']['sample_rate'], **mel_params)
            spk1 = self.spk_enc(mel_s1_4d)  # [B, 512]
            spk2 = self.spk_enc(mel_s2_4d)  # [B, 512]
            return wav_rec, spk1, spk2, q1, q2, com
        return wav_rec, spk_main, spk_main, q1, q2, com

    def training_step(self, batch: dict, batch_idx: int):
        opt_g, opt_d = self.optimizers()
        wav = batch['wav']  # [B, 3, 1, T]
        f0_main = batch['f0'][:, 0]  # [B, T_frames]
        tok_main = batch.get('tok')
        tokens = tok_main[:, 0] if tok_main is not None else None  # [B, T_frames]
        spk_ids = batch['spk_ids']  # [B]
        
        if not self.use_cache:
            tokens = self.get_tokens_dynamic(wav[:, 0])
            
        # 前向传播
        wav_rec, spk1, spk2, q1, q2, com = self(wav)
        
        # ───────── 1. 重建损失 ─────────
        mel_gt = self._compute_mel_3d(wav[:, 0])
        mel_rec = self._compute_mel_3d(wav_rec.unsqueeze(1))
        
        T_min = min(mel_rec.shape[-1], mel_gt.shape[-1])
        mel_rec, mel_gt = mel_rec[..., :T_min], mel_gt[..., :T_min]
 
        l_rec = F.l1_loss(mel_rec, mel_gt) + F.mse_loss(mel_rec, mel_gt)
        l_mrstft = self.l_mrstft(wav_rec.squeeze(1), wav[:, 0].squeeze(1)) if self.enable_mrstft else torch.tensor(0.0, device=wav.device)
        
        # ───────── 2. 蒸馏损失 ─────────
        l_spk = self.l_spk(spk1, spk2, spk_ids)
        l_lin = self.l_lin(q1, tokens) if tokens is not None else torch.tensor(0.0, device=wav.device)
        l_emo_f0 = self.l_emo(q2, f0_main)
        chroma_batch = batch.get('chroma')
        chroma_main = chroma_batch[:, 0] if chroma_batch is not None else None
        l_emo_chroma = self.l_chroma(q2, chroma_main) if (self.enable_chroma and chroma_main is not None) else torch.tensor(0.0, device=wav.device)
        
        # ───────── 3. 对抗损失（显存优化版） ─────────
        # 3.1 生成器步：冻结判别器参数，仅保留到 wav_rec 的梯度链路
        self._set_requires_grad(self.disc, False)
        y_dr, y_dg, f_r, f_g = self.disc(wav[:, 0], wav_rec.unsqueeze(1), return_fmaps=True)
        l_adv_g, l_adv_g_adv, l_adv_g_fm = self.l_adv(
            y_dg, y_dr, f_g, f_r, 'gen', return_components=True
        )
        
        # ───────── 4. 总 Loss 与优化步进 ─────────
        total = (self.cfg['losses']['lambda_r'] * l_rec + 
                 self.cfg['losses']['lambda_a'] * l_adv_g +
                 self.cfg['losses']['lambda_c'] * com + 
                 self.cfg['losses']['lambda_s'] * l_spk +
                 self.cfg['losses']['lambda_l'] * l_lin + 
                 self.cfg['losses']['lambda_e_f0'] * l_emo_f0 +
                 self.cfg['losses']['lambda_e_chroma'] * l_emo_chroma +
                 self.cfg['losses']['lambda_mrstft'] * l_mrstft)
                 
        # 优化生成器
        opt_g.zero_grad()
        self.manual_backward(total)
        torch.nn.utils.clip_grad_norm_(opt_g.param_groups[0]['params'], max_norm=1.0)
        opt_g.step()

        # 3.2 判别器步：放在 G step 之后，避免两张计算图同时驻留；且不返回 fmaps
        self._set_requires_grad(self.disc, True)
        wav_rec_det = wav_rec.detach()
        y_dr_d, y_dg_d, _, _ = self.disc(wav[:, 0], wav_rec_det.unsqueeze(1), return_fmaps=False)
        l_adv_d, l_adv_d_real, l_adv_d_fake = self.l_adv(
            y_dg_d, y_dr_d, [], [], 'disc', return_components=True
        )

        opt_d.zero_grad()
        self.manual_backward(l_adv_d)
        torch.nn.utils.clip_grad_norm_(opt_d.param_groups[0]['params'], max_norm=1.0)
        opt_d.step()
        
        # 日志记录
        self.log_dict(
            {
                'train/loss': total,
                'train/rec': l_rec,
                'train/mrstft': l_mrstft,
                'train/adv_g': l_adv_g,
                'train/adv_g_adv': l_adv_g_adv,
                'train/adv_g_fm': l_adv_g_fm,
                'train/adv_d': l_adv_d,
                'train/adv_d_real': l_adv_d_real,
                'train/adv_d_fake': l_adv_d_fake,
                'train/com': com,
                'train/spk': l_spk,
                'train/lin': l_lin,
                'train/emo_f0': l_emo_f0,
                'train/emo_chroma': l_emo_chroma,
            },
            prog_bar=True,
            batch_size=self.cfg['training']['batch_size'],
        )

    def validation_step(self, batch: dict, batch_idx: int):
        wav = batch['wav']  # [B, 1, T_max]
        wav_rec, _, _, _, _, _ = self(wav)
        
        # ───────── 1. Mel 重建损失 ─────────
        mel_gt = self._compute_mel_3d(wav)
        mel_rec = self._compute_mel_3d(wav_rec.unsqueeze(1))
        if mel_rec.shape[-1] != mel_gt.shape[-1]:
            raise RuntimeError(
                f"Validation mel 帧数不一致: mel_rec={mel_rec.shape[-1]}, mel_gt={mel_gt.shape[-1]}"
            )
        l_rec = F.l1_loss(mel_rec, mel_gt) + F.mse_loss(mel_rec, mel_gt)
        
        # ───────── 2. MR-STFT 损失（剔除 Padding 区域） ─────────
        if self.enable_mrstft:
            if 'lengths' in batch:
                valid_len = int(batch['lengths'].min().item())
            else:
                valid_len = min(wav_rec.shape[-1], wav.shape[-1])
            l_mrstft = self.l_mrstft(
                wav_rec[:, :valid_len],
                wav[:, 0, :valid_len]
            )
        else:
            l_mrstft = torch.tensor(0.0, device=wav.device)
        
        # ───────── 3. 日志记录 ─────────
        bs = self.cfg['training']['batch_size']
        self.log('val/rec', l_rec, prog_bar=True, batch_size=bs)
        self.log('val/mrstft', l_mrstft, prog_bar=False, batch_size=bs)
        
        # ───────── 4. 音频日志 ─────────
        if self.global_rank == 0 and batch_idx == 0:
            sr = self.cfg['model']['sample_rate']
            step = self.global_step
            orig = normalize_audio(wav[0, 0].detach().cpu().unsqueeze(0))
            rec = normalize_audio(wav_rec[0].detach().cpu().unsqueeze(0))
            self.logger.experiment.add_audio('val/original', orig, step, sr)
            self.logger.experiment.add_audio('val/reconstructed', rec, step, sr)

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()
    
    def configure_optimizers(self):
        g_p = [p for n, p in self.named_parameters() if 'disc' not in n]
        
        lr_g = self.cfg['training'].get('lr_g', 1.25e-4)
        lr_d = self.cfg['training'].get('lr_d', 1.25e-4)
        
        opt_g = torch.optim.AdamW(
            g_p, 
            lr=lr_g, 
            betas=self.cfg['training']['betas'],
            weight_decay=self.cfg['training']['weight_decay']
        )
        
        opt_d = torch.optim.AdamW(
            self.disc.parameters(), 
            lr=lr_d,
            betas=self.cfg['training']['betas'],
            weight_decay=self.cfg['training']['weight_decay']
        )
        
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.cfg['training']['gamma'])
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.cfg['training']['gamma'])
        
        return [opt_g, opt_d], [scheduler_g, scheduler_d]
