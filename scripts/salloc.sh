salloc \
    -p titan \
    --qos titan \
    --nodelist titanrtx01 \
    -c25 \
    --mem=64G \
    --gres=gpu:4 \
    -n1

salloc \
    -p a100 \
    --qos a100 \
    -c14 \
    --mem=48G \
    --gres=gpu:1 \
    -n1

salloc \
    -p rtx2080ti \
    --qos rtx2080ti \
    -c20 \
    --mem=64G \
    --gres=gpu:1 \
    -n1

salloc \
    -p l40s \
    --qos dcgpu \
    -c16 \
    --mem=64G \
    --gres=gpu:1 \
    -n1