"""Generate latents for DROID chunk videos (raw format)."""

import torch
import mediapy
import numpy as np
from pathlib import Path
from diffusers.models import AutoencoderKLTemporalDecoder
from tqdm import tqdm
import argparse


def generate_latent_for_episode(episode_id, chunk_dir, output_dir, vae, device="cuda:1"):
    """Generate latents for a specific episode (3 camera views).

    Args:
        episode_id: Episode number (e.g., 0, 1, 2, ...)
        chunk_dir: Path to chunk-000 directory
        output_dir: Output directory for latents
        vae: VAE model
        device: Device to use
    """
    chunk_dir = Path(chunk_dir)
    output_dir = Path(output_dir)

    # Camera folders in DROID raw format
    camera_folders = [
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
        "observation.images.wrist_left"
    ]

    episode_str = f"episode_{episode_id:06d}"
    episode_output_dir = output_dir / episode_str
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Episode {episode_id:06d}] Processing...")

    for cam_id, cam_folder in enumerate(camera_folders):
        video_path = chunk_dir / cam_folder / f"{episode_str}.mp4"

        if not video_path.exists():
            print(f"  Warning: {video_path} not found, skipping camera {cam_id}...")
            continue

        print(f"  [Camera {cam_id}] Processing {video_path.name}...")

        # Load video
        video = mediapy.read_video(str(video_path))
        print(f"    Original video shape: {video.shape}")

        # Resize to 192x320 if needed (DROID standard resolution)
        target_height, target_width = 192, 320
        if video.shape[1] != target_height or video.shape[2] != target_width:
            print(f"    Resizing to {target_height}x{target_width}...")
            import cv2
            resized_frames = []
            for frame in video:
                resized = cv2.resize(frame, (target_width, target_height))
                resized_frames.append(resized)
            video = np.array(resized_frames)
            print(f"    Resized video shape: {video.shape}")

        # Convert to tensor: (T, H, W, C) -> (T, C, H, W), normalize to [-1, 1]
        frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
        frames = frames.to(device)

        # Encode to latent in batches
        with torch.no_grad():
            batch_size = 64
            latents = []
            for i in tqdm(range(0, len(frames), batch_size),
                         desc=f"    Encoding", leave=False):
                batch = frames[i:i+batch_size]
                latent = vae.encode(batch).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
                latents.append(latent.cpu())

            latent_tensor = torch.cat(latents, dim=0)

        print(f"    Latent shape: {latent_tensor.shape}")

        # Save latent
        output_path = episode_output_dir / f"{cam_id}.pt"
        torch.save(latent_tensor, output_path)
        print(f"    Saved to {output_path}")

    print(f"  ✓ Episode {episode_id:06d} done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=str, required=True,
                       help="Path to chunk directory (e.g., data/videos/chunk-000)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for latents")
    parser.add_argument("--svd-path", type=str, required=True,
                       help="SVD model path")
    parser.add_argument("--start-id", type=int, default=0,
                       help="Start episode ID")
    parser.add_argument("--end-id", type=int, default=999,
                       help="End episode ID (inclusive)")
    parser.add_argument("--device", type=str, default="cuda:1")

    args = parser.parse_args()

    # Load VAE
    print(f"Loading VAE from {args.svd_path}...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.svd_path, subfolder="vae"
    ).to(args.device)
    print(f"VAE loaded on {args.device}\n")

    # Process episodes
    for episode_id in range(args.start_id, args.end_id + 1):
        try:
            generate_latent_for_episode(
                episode_id=episode_id,
                chunk_dir=args.chunk_dir,
                output_dir=args.output_dir,
                vae=vae,
                device=args.device
            )
        except Exception as e:
            print(f"  ✗ Error processing episode {episode_id:06d}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"All done! Latents saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
