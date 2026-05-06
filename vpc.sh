CONFIG=runs/Ano/Baseline/hparams_full.yaml
CKPT=runs/Ano/Baseline/checkpoints/best/epoch=6-step=98306.ckpt
POOL=runs/Ano/Baseline/vctk_speaker_vectors.pt
SUFFIX=_Baseline_50k

python extract_vctk_pool.py --config $CONFIG --ckpt $CKPT --output $POOL
python generate_vpc.py --config $CONFIG --ckpt $CKPT --pool $POOL --anon_suffix $SUFFIX

cd ../Voice-Privacy-Challenge-2024

bash 05_evaluations.sh
