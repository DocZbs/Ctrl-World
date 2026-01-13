SVD="/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e"
DATA="dataset_example/droid_new_setup"

for i in $(seq 1 2); do
sid=$(printf "%04d" "$i")
python scripts/gen_latent_simple.py \
    --scene-id "$sid" \
    --data-root "$DATA" \
    --split val \
    --svd-path "$SVD" \
    --device cuda:1
done