#!/bin/bash
# Run Pi0-FAST experiments on DROID validation scenarios

python scripts/run_all_droid_new_setup.py \
    --config omni_ctrl/configs/omni_ctrl_pi0_fast_droid.yaml \
    --ann-dir dataset_example/droid_new_setup/annotation/val \
    --droid-root dataset_example/droid_new_setup \
    --out-base experiments/omni_ctrl_fixed_scene_batch_pi0_fast \
    --iterations 1 \
    --skip-existing
