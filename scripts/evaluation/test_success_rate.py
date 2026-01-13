#!/usr/bin/env python3
"""
Test success rate on droid_new_setup dataset using rollout_interact_pi.py interface

Usage:
    python scripts/test_success_rate.py \
        --wm_ckpt /path/to/checkpoint-10000.pt \
        --policy_type pi05 \
        --dataset_dir dataset_example/droid_new_setup \
        --output_dir experiments/test_success_rate
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapy
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import wm_args
from scripts.rollout_interact_pi import agent


def load_test_scenarios(dataset_dir):
    """Load all test scenarios from droid_new_setup."""
    dataset_dir = Path(dataset_dir)
    annotation_dir = dataset_dir / "annotation" / "val"

    scenarios = []
    for ann_file in sorted(annotation_dir.glob("*.json")):
        with open(ann_file) as f:
            anno = json.load(f)

        episode_id = ann_file.stem
        scenarios.append({
            'episode_id': episode_id,
            'instruction': anno['texts'][0] if 'texts' in anno and len(anno['texts']) > 0 else 'No instruction',
            'annotation_file': str(ann_file),
        })

    return scenarios


def run_single_test(agent_obj, episode_id, instruction, dataset_dir, output_dir, args):
    """Run a single test episode using rollout_interact_pi.py interface."""

    print(f"\n{'='*70}")
    print(f"Testing Episode: {episode_id}")
    print(f"Instruction: {instruction}")
    print(f"{'='*70}")

    try:
        # Get trajectory info (using rollout_interact_pi interface)
        eef_gt, joint_pos_gt, video_dict, video_latents, _ = agent_obj.get_traj_info(
            episode_id,
            start_idx=0,
            steps=int(args.pred_step * args.interact_num + 8)
        )

        print(f"Initial state:")
        print(f"  EEF pose: {eef_gt[0]}")
        print(f"  Joint pos: {joint_pos_gt[0]}")

        # Initialize history buffers (following rollout_interact_pi.py lines 335-343)
        video_to_save, info_to_save = [], []
        his_cond, his_joint, his_eef = [], [], []

        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"

        for i in range(args.num_history * 4):
            his_cond.append(first_latent)  # (1, 4, 72, 40)
            his_joint.append(joint_pos_gt[0:1])  # (1, 7)
            his_eef.append(eef_gt[0:1])  # (1, 7)

        video_dict_pred = [v[0:1] for v in video_dict]

        # Run rollout (following rollout_interact_pi.py lines 346-398)
        print(f"\nRunning rollout for {args.interact_num} interactions...")
        for step_i in tqdm(range(args.interact_num), desc="Rollout"):
            # Get ground truth video latents for this step (lines 350-352)
            start_id = int(step_i * (args.pred_step - 1))
            end_id = start_id + args.pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            # Policy forward (lines 356-360) - using forward_policy interface
            current_joint = his_joint[-1][0]
            current_pose = his_eef[-1][0]
            current_obs = [video_dict_pred[i][-1] for i in range(3)]
            policy_in_out, joint_pos_skip, state_fk_skip = agent_obj.forward_policy(
                current_obs,
                current_pose,
                current_joint,
                text=instruction,
                time_step=step_i
            )

            # Prepare action condition (lines 369-370)
            action_cond = np.concatenate([his_eef[idx] for idx in args.history_idx], axis=0)
            action_cond = np.concatenate([action_cond, state_fk_skip], axis=0)

            # Prepare history latents (line 371)
            his_latent = torch.cat([his_cond[idx] for idx in args.history_idx], dim=0).unsqueeze(0)

            # Current latent (line 372)
            current_latent = his_cond[-1]

            # World model forward (lines 374)
            video_cat, true_video, videos_pred, latents_pred = agent_obj.forward_wm(
                action_cond,
                video_latent_true,
                current_latent,
                his_cond=his_latent,
                text=instruction
            )

            # Extract predicted latent for next step (line 380)
            pred_step = args.pred_step
            latent_pred = torch.cat([latents_pred[v, pred_step-1] for v in range(3)], dim=1).unsqueeze(0)

            # Update history (lines 378-380)
            his_cond.append(latent_pred)
            his_joint.append(joint_pos_skip[pred_step-1][None, :])
            his_eef.append(state_fk_skip[pred_step-1][None, :])

            # Update video dict (lines 389-391)
            for j in range(3):
                video_dict_pred[j] = np.concatenate([video_dict_pred[j], videos_pred[j]], axis=0)

        # Save results
        episode_output_dir = Path(output_dir) / episode_id
        episode_output_dir.mkdir(parents=True, exist_ok=True)

        # Save video
        video_path = episode_output_dir / "rollout.mp4"
        # Concatenate 3 camera views horizontally
        frames_concat = np.concatenate([video_dict_pred[i] for i in range(3)], axis=2)
        mediapy.write_video(str(video_path), frames_concat, fps=5)

        print(f"\n✓ Test completed successfully")
        print(f"  Video saved to: {video_path}")
        print(f"  Generated {len(frames_concat)} frames")

        return {
            'episode_id': episode_id,
            'instruction': instruction,
            'status': 'success',
            'video_path': str(video_path),
            'num_frames': len(frames_concat),
        }

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

        return {
            'episode_id': episode_id,
            'instruction': instruction,
            'status': 'failed',
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='Test success rate on droid_new_setup')

    # Model paths
    parser.add_argument('--wm_ckpt', type=str, required=True,
                       help='Path to world model checkpoint')
    parser.add_argument('--svd_model_path', type=str,
                       default='/mnt/nvme-fast/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e',
                       help='Path to SVD model')
    parser.add_argument('--clip_model_path', type=str,
                       default='/mnt/nvme-fast/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
                       help='Path to CLIP model')
    parser.add_argument('--pi_ckpt', type=str,
                       default='/mnt/nvme-fast/huggingface/hub/openpi-assets/checkpoints/pi05_droid',
                       help='Path to policy checkpoint')
    parser.add_argument('--action_adapter', type=str,
                       default='models/action_adapter/model2_15_9.pth',
                       help='Path to action adapter')

    # Dataset
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Path to droid_new_setup dataset')
    parser.add_argument('--data_stat_path', type=str,
                       default='dataset_meta_info/droid/stat.json',
                       help='Path to data statistics')

    # Policy
    parser.add_argument('--policy_type', type=str, default='pi05',
                       choices=['pi05', 'pi0', 'pi0fast'],
                       help='Policy type')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')

    # Test configuration
    parser.add_argument('--episode_ids', type=str, nargs='+', default=None,
                       help='Specific episode IDs to test (e.g., 0001 0002). If not specified, test all.')

    args_cli = parser.parse_args()

    # Create wm_args
    args = wm_args()
    args.ckpt_path = args_cli.wm_ckpt
    args.svd_model_path = args_cli.svd_model_path
    args.clip_model_path = args_cli.clip_model_path
    args.pi_ckpt = args_cli.pi_ckpt
    args.action_adapter = args_cli.action_adapter
    args.val_dataset_dir = args_cli.dataset_dir
    args.data_stat_path = args_cli.data_stat_path
    args.policy_type = args_cli.policy_type
    args.dtype = torch.bfloat16

    # Create output directory
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Testing Success Rate on DROID New Setup")
    print(f"{'='*70}")
    print(f"World Model: {args.ckpt_path}")
    print(f"Policy: {args.policy_type} ({args.pi_ckpt})")
    print(f"Dataset: {args.val_dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    # Load scenarios
    print("Loading test scenarios...")
    all_scenarios = load_test_scenarios(args.val_dataset_dir)

    # Filter scenarios if specific episode IDs are provided
    if args_cli.episode_ids:
        scenarios = [s for s in all_scenarios if s['episode_id'] in args_cli.episode_ids]
        print(f"Testing {len(scenarios)} specified episodes: {args_cli.episode_ids}")
    else:
        scenarios = all_scenarios
        print(f"Testing all {len(scenarios)} episodes")

    # Create agent
    print("\nInitializing agent...")
    Agent = agent(args)
    print("✓ Agent initialized")

    # Run tests
    results = []
    for scenario in scenarios:
        result = run_single_test(
            agent_obj=Agent,
            episode_id=scenario['episode_id'],
            instruction=scenario['instruction'],
            dataset_dir=args.val_dataset_dir,
            output_dir=output_dir,
            args=args,
        )
        results.append(result)

    # Compute statistics
    print(f"\n{'='*70}")
    print("Test Results Summary")
    print(f"{'='*70}")

    total = len(results)
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    success_rate = (success / total * 100) if total > 0 else 0

    print(f"Total tests: {total}")
    print(f"Successful:  {success}")
    print(f"Failed:      {failed}")
    print(f"Success rate: {success_rate:.2f}%")

    # Save results
    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'wm_ckpt': args.ckpt_path,
                'policy_type': args.policy_type,
                'dataset_dir': args.val_dataset_dir,
            },
            'summary': {
                'total': total,
                'success': success,
                'failed': failed,
                'success_rate': success_rate,
            },
            'results': results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
