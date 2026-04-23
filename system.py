# system.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from modules import *
from losses import *
from utils import setup_logger, compute_mel, save_audio, setup_seed, normalize_audio
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
        self.enc = SpeechEncoder(cfg['model']['encoder_strides'], cfg['model']['hidden_dim'])
        self.spk_enc = SpeakerEncoder({**cfg['model']['speaker'], 'n_mels': cfg['model']['n_mels']})
        self.bottleneck = ResidualBottleneck(cfg)
        self.dec = Decoder(cfg['model']['encoder_strides'], cfg['model']['hidden_dim'])
        self.disc = HiFiGANDiscriminator()
        
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
        self.l_emo = EmoDistillLoss(cfg['model']['bottleneck']['codebook_dim'])  # ✅ F0 MSE
        self.l_chroma = ChromaDistillLoss(cfg['model']['bottleneck']['codebook_dim'], n_chroma=24)  # ✅ 新增
        self.l_mrstft = MultiResolutionSTFTLoss()  # ✅ 新增
        self.l_adv = AdvLoss()
        
        self.automatic_optimization = False

    def get_tokens_dynamic(self, wav_flat: torch.Tensor) -> torch.Tensor:
        if self.wavlm is None or self.kmeans is None:
            raise RuntimeError("教师模型未加载。请确保 data.use_cache=False 或预缓存文件完整。")
        with torch.no_grad():
            feats = self.wavlm(wav_flat)  # [B*, T_wavlm, D_wavlm]
            target = wav_flat.shape[-1] // 320
            if feats.shape[1] != target:
                feats = F.interpolate(feats.transpose(1, 2), size=target, mode='linear', align_corners=False).transpose(1, 2)  # [B*, T_feat, D_wavlm]
        return self.kmeans.predict(feats)  # [B*, T_feat]

    def _compute_mel_3d(self, wav: torch.Tensor) -> torch.Tensor:
        """辅助函数：计算 Mel 并统一输出 [B, F, T] 维度，避免重复 .squeeze(1)"""
        return compute_mel(wav, self.cfg['model']['n_mels'], 
                          self.cfg['model']['sample_rate'], 
                          self.cfg['model']['mel_hop_length']).squeeze(1)  # [B, F, T_mel]

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
        mel_main_4d = compute_mel(wav_main, self.cfg['model']['n_mels'], 
                                  self.cfg['model']['sample_rate'], self.cfg['model']['mel_hop_length'])
        feat_main = self.enc(wav_main)                  # [B, T_feat, 512]
        spk_main = self.spk_enc(mel_main_4d)            # [B, 256]
        
        # ✅ 串行解耦：减去说话人身份
        r1 = feat_main - spk_main.unsqueeze(1)          # [B, T_feat, 512]
        recon, q1, q2, com = self.bottleneck(r1)        # recon:[B, T_feat, 512]
        
        # 🔑 核心修复：加回说话人身份用于重建（严格对齐论文 Eq.7 & Figure 1）
        recon_with_spk = recon + spk_main.unsqueeze(1)  # [B, T_feat, 512]
        wav_rec = self.dec(recon_with_spk.transpose(1, 2))  # [B, C, T] -> [B, T]

        if is_train:
            # ───────── 蒸馏路径：仅提取 s1, s2 用于说话人一致性约束 ─────────
            mel_s1_4d = compute_mel(
                wav_s1,
                self.cfg['model']['n_mels'],
                self.cfg['model']['sample_rate'],
                self.cfg['model']['mel_hop_length'],
            )  # [B, 1, F, T_mel]
            mel_s2_4d = compute_mel(
                wav_s2,
                self.cfg['model']['n_mels'],
                self.cfg['model']['sample_rate'],
                self.cfg['model']['mel_hop_length'],
            )  # [B, 1, F, T_mel]
            spk1 = self.spk_enc(mel_s1_4d)  # [B, 256]
            spk2 = self.spk_enc(mel_s2_4d)  # [B, 256]
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
        l_mrstft = self.l_mrstft(wav_rec.squeeze(1), wav[:, 0].squeeze(1))
        
        # ───────── 2. 蒸馏损失 ─────────
        l_spk = self.l_spk(spk1, spk2, spk_ids)
        l_lin = self.l_lin(q1, tokens) if tokens is not None else torch.tensor(0.0, device=wav.device)
        l_emo_f0 = self.l_emo(q2, f0_main)
        chroma_batch = batch.get('chroma')
        chroma_main = chroma_batch[:, 0] if chroma_batch is not None else None
        l_emo_chroma = self.l_chroma(q2, chroma_main) if chroma_main is not None else torch.tensor(0.0, device=wav.device)
        
        # ───────── 3. 对抗损失（✅ 补全判别器步） ─────────
        # 3.1 生成器步
        y_dr, y_dg, f_r, f_g = self.disc(wav[:, 0], wav_rec.unsqueeze(1))
        l_adv_g = self.l_adv(y_dg, y_dr, f_g, f_r, 'gen')
        
        # ✅ 3.2 判别器步：必须 detach 切断生成器梯度，防止二次反向传播报错
        wav_rec_det = wav_rec.detach()
        y_dr_d, y_dg_d, _, _ = self.disc(wav[:, 0], wav_rec_det.unsqueeze(1))
        l_adv_d = self.l_adv(y_dg_d, y_dr_d, [], [], 'disc')  # 判别器无需 fmap，传 [] 节省显存
        
        # ───────── 4. 总 Loss 与优化步进 ─────────
        total = (self.cfg['losses']['lambda_r'] * l_rec + 
                 self.cfg['losses']['lambda_a'] * l_adv_g +
                 self.cfg['losses']['lambda_c'] * com + 
                 self.cfg['losses']['lambda_s'] * l_spk +
                 self.cfg['losses']['lambda_l'] * l_lin + 
                 self.cfg['losses']['lambda_e'] * l_emo_f0 +
                 self.cfg['losses']['lambda_e_chroma'] * l_emo_chroma +
                 self.cfg['losses']['lambda_mrstft'] * l_mrstft)
                 
        # 优化生成器
        opt_g.zero_grad()
        self.manual_backward(total)
        torch.nn.utils.clip_grad_norm_(opt_g.param_groups[0]['params'], max_norm=1.0)
        opt_g.step()
        
        # 优化判别器
        opt_d.zero_grad()
        self.manual_backward(l_adv_d)  # ✅ 现在 l_adv_d 已定义
        torch.nn.utils.clip_grad_norm_(opt_d.param_groups[0]['params'], max_norm=1.0)
        opt_d.step()
        
        # 日志记录
        self.log_dict(
            {
                'train/loss': total,
                'train/rec': l_rec,
                'train/mrstft': l_mrstft,
                'train/adv_g': l_adv_g,      # ✅ 拆分 adv_g
                'train/adv_d': l_adv_d,      # ✅ 新增 adv_d 监控
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
        wav_rec, _, _, _, _, _ = self(wav)  # wav_rec: [B, T_max]
        
        # ───────── 1. Mel 重建损失 ─────────
        mel_gt = self._compute_mel_3d(wav)
        mel_rec = self._compute_mel_3d(wav_rec.unsqueeze(1))
        T_min_mel = min(mel_rec.shape[-1], mel_gt.shape[-1])
        l_rec = F.l1_loss(mel_rec[..., :T_min_mel], mel_gt[..., :T_min_mel]) + \
                F.mse_loss(mel_rec[..., :T_min_mel], mel_gt[..., :T_min_mel])
        
        # ───────── 2. MR-STFT 损失（时域多分辨率谱监督） ─────────
        # 对齐波形长度，防止编解码 stride 导致的 1~2 帧偏差
        T_min_wave = min(wav_rec.shape[-1], wav.shape[-1])
        l_mrstft = self.l_mrstft(
            wav_rec[:, :T_min_wave],          # [B, T]
            wav[:, 0, :T_min_wave]            # [B, T]
        )
        
        # ───────── 3. 日志记录 ─────────
        bs = self.cfg['training']['batch_size']
        self.log('val/rec', l_rec, prog_bar=True, batch_size=bs)
        self.log('val/mrstft', l_mrstft, prog_bar=False, batch_size=bs)  # ✅ 新增监控
        
        # ───────── 4. 音频日志（仅 rank 0 且 batch_idx=0 时记录） ─────────
        if self.global_rank == 0 and batch_idx == 0:
            sr = self.cfg['model']['sample_rate']
            step = self.global_step
            orig = normalize_audio(wav[0, 0].detach().cpu().unsqueeze(0))
            rec = normalize_audio(wav_rec[0].detach().cpu().unsqueeze(0))
            self.logger.experiment.add_audio('val/original', orig, step, sr)
            self.logger.experiment.add_audio('val/reconstructed', rec, step, sr)

    def on_train_epoch_end(self):
        """手动优化模式下，需在 Epoch 结束时手动触发学习率调度器步进"""
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()
    
    def configure_optimizers(self):
        # 生成器参数（排除判别器）
        g_p = [p for n, p in self.named_parameters() if 'disc' not in n]
        
        # ✅ 生成器优化器：论文指定 β1=0.8, β2=0.99, weight_decay=1e-5
        opt_g = torch.optim.AdamW(
            g_p, 
            lr=self.cfg['training']['lr'], 
            betas=self.cfg['training']['betas'],  # [0.8, 0.99]
            weight_decay=self.cfg['training']['weight_decay']  # 1e-5
        )
        
        # ✅ 判别器优化器：标准动量 + 显式关闭权重衰减（防决策边界模糊）
        opt_d = torch.optim.AdamW(
            self.disc.parameters(), 
            lr=self.cfg['training']['lr'],
            betas=(0.9, 0.999),     # 默认值，判别器无需低动量
            weight_decay=0.0        # 🔑 关键：判别器必须关闭 weight_decay
        )
        
        # ✅ 论文指定：学习率每 epoch 衰减 0.99 倍
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.cfg['training']['gamma'])
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.cfg['training']['gamma'])
        
        # PyTorch Lightning 标准返回格式：优化器列表 + 调度器列表
        # 默认在每 epoch 结束时自动调用 scheduler.step()
        return [opt_g, opt_d], [scheduler_g, scheduler_d]