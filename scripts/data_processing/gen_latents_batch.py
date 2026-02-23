"""Batch generate latents for DROID dataset."""

import torch
import numpy as np
from pathlib import Path
from diffusers.models import AutoencoderKLTemporalDecoder
from tqdm import tqdm
import json
import argparse
import subprocess
import tempfile


def read_video_ffmpeg(video_path):
    """Read video using ffmpeg (supports AV1).

    Returns:
        numpy array of shape (T, H, W, 3) in RGB format
    """
    # Use ffmpeg to decode video to raw RGB frames
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-v', 'quiet',
        'pipe:1'
    ]

    # Get video info first
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=width,height,nb_read_packets',
        '-of', 'csv=p=0',
        str(video_path)
    ]

    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    info = result.stdout.strip().split(',')
    if len(info) < 3:
        return None

    width, height, num_frames = int(info[0]), int(info[1]), int(info[2])

    # Read raw video data
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return None

    # Convert bytes to numpy array
    video_data = np.frombuffer(result.stdout, dtype=np.uint8)
    expected_size = num_frames * height * width * 3

    if len(video_data) != expected_size:
        # Try to reshape with available data
        num_frames = len(video_data) // (height * width * 3)
        if num_frames == 0:
            return None
        video_data = video_data[:num_frames * height * width * 3]

    frames = video_data.reshape((num_frames, height, width, 3))
    return frames


def generate_latents_for_episode(
    episode_id,
    droid_root,
    svd_path,
    device="cuda:0",
    output_root=None
):
    """Generate latents for a specific episode.

    Args:
        episode_id: Episode ID (e.g., "074572")
        droid_root: Root path of DROID dataset
        svd_path: Path to SVD model
        device: Device to use
        output_root: Output root (default: droid_root/latents)
    """
    droid_root = Path(droid_root)
    if output_root is None:
        output_root = droid_root / "latents"

    # Determine chunk
    chunk_num = int(episode_id) // 1000
    chunk_name = f"chunk-{chunk_num:03d}"

    # Check if videos exist
    video_dir = droid_root / "videos" / chunk_name
    camera_names = ["exterior_1_left", "exterior_2_left", "wrist_left"]

    video_paths = []
    for cam in camera_names:
        video_path = video_dir / f"observation.images.{cam}" / f"episode_{episode_id}.mp4"
        if not video_path.exists():
            print(f"Warning: {video_path} not found")
            return False
        video_paths.append(video_path)

    # Load VAE
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)
    vae.eval()

    # Output directory
    output_dir = output_root / chunk_name / f"episode_{episode_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each camera
    for cam_idx, video_path in enumerate(video_paths):
        print(f"  Processing camera {cam_idx}: {video_path.name}")

        # Read video using ffmpeg (supports AV1)
        frames = read_video_ffmpeg(video_path)

        if frames is None or len(frames) == 0:
            print(f"    Warning: Failed to read {video_path}")
            continue

        print(f"    Loaded {len(frames)} frames, shape: {frames.shape}")

        # Convert to tensor: (T, H, W, C) -> (T, C, H, W), normalize to [-1, 1]
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        frames_tensor = frames_tensor / 255.0 * 2.0 - 1.0
        frames_tensor = frames_tensor.to(device)

        # Encode in batches
        with torch.no_grad():
            batch_size = 32
            latents = []
            for i in range(0, len(frames_tensor), batch_size):
                batch = frames_tensor[i:i+batch_size]
                latent = vae.encode(batch).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
                latents.append(latent.cpu())

            latent_tensor = torch.cat(latents, dim=0)

        print(f"    Latent shape: {latent_tensor.shape}")

        # Save
        output_path = output_dir / f"{cam_idx}.pt"
        torch.save(latent_tensor, output_path)
        print(f"    Saved to {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--droid-root", type=str, required=True, help="DROID dataset root")
    parser.add_argument("--svd-path", type=str, required=True, help="SVD model path")
    parser.add_argument("--episodes", type=str, nargs="+", help="Episode IDs to process")
    parser.add_argument("--episode-file", type=str, help="File containing episode IDs (one per line)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to process")

    args = parser.parse_args()

    # Get episode list
    if args.episode_file:
        with open(args.episode_file) as f:
            episodes = [line.strip() for line in f if line.strip()]
    elif args.episodes:
        episodes = args.episodes
    else:
        # Process all episodes from annotations
        anno_dir = Path(args.droid_root) / "annotation" / "train"
        episodes = [f.stem for f in sorted(anno_dir.glob("*.json"))]

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"Processing {len(episodes)} episodes...")

    success_count = 0
    for episode_id in tqdm(episodes, desc="Generating latents"):
        try:
            success = generate_latents_for_episode(
                episode_id=episode_id,
                droid_root=args.droid_root,
                svd_path=args.svd_path,
                device=args.device
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"Error processing {episode_id}: {e}")
            continue

    print(f"\nDone! Successfully processed {success_count}/{len(episodes)} episodes")


if __name__ == "__main__":
    main()
