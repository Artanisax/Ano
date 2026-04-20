srun \
    --pty \
    -p a100 \
    --qos a100 \
    -c20 \
    --mem=64G \
    --gres=gpu:1 \
    -n1 \
    bash
# --nodelist l40gpu001 \

srun \
    --pty \
    -p titan \
    --qos titan \
    --nodelist titanrtx01 \
    -c20 \
    --mem=64G \
    --gres=gpu:1 \
    -n1 \
    bash

srun \
    --pty \
    -p rtx2080ti \
    --qos rtx2080ti \
    -c20 \
    --mem=24G \
    --gres=gpu:1 \
    -n1 \
    bash

srun \
    --pty \
    -p titan \
    --qos titan \
    --nodelist titanrtx01 \
    -c50 \
    --mem=64G \
    --gres=gpu:4 \
    -n1 \
    bash

srun \
    --pty \
    -p rtx2080ti \
    --qos rtx2080ti \
    -c20 \
    --mem=64G \
    --gres=gpu:3 \
    -n1 \
    bash