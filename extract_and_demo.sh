CONFIG=runs/Ano/Baseline/hparams_full.yaml
CKPT=runs/Ano/Baseline/checkpoints/best/epoch=3-step=60248.ckpt
POOL=runs/Ano/Baseline/vctk_speaker_vectors.pt
OUTPUT=runs/Ano/Baseline/outputs

# python extract_vctk_pool.py --config $CONFIG --ckpt $CKPT --output $POOL
python anonymize.py --config $CONFIG --ckpt $CKPT --pool $POOL --output $OUTPUT