  CKPT_PATH="/data1/zbs_files/data/HF/hub/models--yjguo--Ctrl-World/snapshots/8cf814693f411962dc866a2ddb5b785afd17a93a/checkpoint-10000.pt"

  CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py \
      --dataset_root_path dataset_example \
      --dataset_meta_info_path dataset_meta_info \
      --dataset_names droid_subset \
      --svd_model_path stabilityai/stable-video-diffusion-img2vid \
      --clip_model_path openai/clip-vit-base-patch32 \
      --ckpt_path ${CKPT_PATH} \
      --task_type close_laptop \
      # --keyboard llrr