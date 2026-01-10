python scripts/generate_synthetic_trajectories.py --annotation-file dataset_example/droid_new_setup/annotation/val/0002.json --num-rollouts 1 --output-dir synthetic_data/pickplace_0002 \
    --instruction-variants "pick the blue block and place it in plate" "pick up the blue block and put it inside the plate" "grasp the blue cube and move it to the plate" "take the blue object and place it in the dish" \
          --wm-device cuda:0 \
      --policy-device cuda:1 \