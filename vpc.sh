# CONFIG=runs/Ano/Baseline/hparams_full.yaml
# CKPT=runs/Ano/Baseline/checkpoints/best/epoch=3-step=60248.ckpt
# POOL=runs/Ano/Baseline/vctk_speaker_vectors.pt
# SUFFIX=_Baseline_TEST

# python extract_vctk_pool.py --config $CONFIG --ckpt $CKPT --output $POOL
# python generate_vpc.py --config $CONFIG --ckpt $CKPT --pool $POOL --anon_suffix $SUFFIX

================================================================

CONFIG=runs/Ano/Ablation_17/hparams_full.yaml
CKPT=runs/Ano/Ablation_17/checkpoints/best/epoch=5-step=95130.ckpt
POOL=runs/Ano/Ablation_17/vctk_speaker_vectors.pt
SUFFIX=_Ablation_17

python extract_vctk_pool.py --config $CONFIG --ckpt $CKPT --output $POOL
python generate_vpc.py --config $CONFIG --ckpt $CKPT --pool $POOL --anon_suffix $SUFFIX

================================================================

CONFIG=runs/Ano/Ablation_Extra_Losses/hparams_full.yaml
CKPT=runs/Ano/Ablation_Extra_Losses/checkpoints/best/epoch=6-step=104646.ckpt
POOL=runs/Ano/Ablation_Extra_Losses/vctk_speaker_vectors.pt
SUFFIX=_Ablation_Extra_Losses

python extract_vctk_pool.py --config $CONFIG --ckpt $CKPT --output $POOL
python generate_vpc.py --config $CONFIG --ckpt $CKPT --pool $POOL --anon_suffix $SUFFIX

================================================================

CONFIG=runs/Ano/Ablation_Log_F0_with_Chroma/hparams_full.yaml
CKPT=runs/Ano/Ablation_Log_F0_with_Chroma/checkpoints/best/epoch=6-step=110986.ckpt
POOL=runs/Ano/Ablation_Log_F0_with_Chroma/vctk_speaker_vectors.pt
SUFFIX=_Ablation_Log_F0_with_Chroma

python extract_vctk_pool.py --config $CONFIG --ckpt $CKPT --output $POOL
python generate_vpc.py --config $CONFIG --ckpt $CKPT --pool $POOL --anon_suffix $SUFFIX

================================================================
