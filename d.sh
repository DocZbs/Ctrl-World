HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Youyi-Kou/ctrl-world \
    --local-dir-use-symlinks False

# gg huggingface-cli download Youyi-Kou/ctrl-world \
#     --local-dir /data1/zbs_files/data/HF/hub/models--Youyi-Kou--ctrl-world --local-dir-use-symlinks False
HF_HUB_OFFLINE=0 HF_HUB_CACHE=$TOK_CACHE TRANSFORMERS_CACHE=$TOK_CACHE python
- <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("physical-intelligence/fast", cache_dir="/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi0_fast_droid")
PY