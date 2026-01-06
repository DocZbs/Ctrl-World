"""Simple script to generate latents for existing videos."""

import torch
import mediapy
import numpy as np
from pathlib import Path
from diffusers.models import AutoencoderKLTemporalDecoder
from tqdm import tqdm
import json


def generate_latents_for_scene(scene_id, data_root, split="val", svd_path=None, device="cuda:1"):
    """Generate latents for a specific scene.

    Args:
        scene_id: Scene ID (e.g., "0001")
        data_root: Root path of DROID dataset (e.g., "dataset_example/droid_new_setup")
        split: "train" or "val"
        svd_path: Path to SVD model
        device: Device to use
    """
    data_root = Path(data_root)

    # Load annotation
    anno_path = data_root / "annotation" / split / f"{scene_id}.json"
    print(f"Loading annotation from {anno_path}...")
    with open(anno_path) as f:
        anno = json.load(f)

    instruction = anno["texts"][0] if anno["texts"] else "N/A"
    print(f"Scene: {scene_id}")
    print(f"Instruction: {instruction}")
    print(f"Video length: {anno['video_length']} frames")

    # Load VAE
    print(f"\nLoading VAE from {svd_path}...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)
    print(f"VAE loaded on {device}")

    # Process each camera view
    output_dir = data_root / "latent_videos" / split / scene_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_id in range(3):
        video_path = data_root / "videos" / split / scene_id / f"{video_id}.mp4"

        if not video_path.exists():
            print(f"Warning: {video_path} not found, skipping...")
            continue

        print(f"\n[Camera {video_id}] Processing {video_path}...")

        # Load video
        video = mediapy.read_video(str(video_path))
        print(f"  Original video shape: {video.shape}")

        # Convert to tensor: (T, H, W, C) -> (T, C, H, W), normalize to [-1, 1]
        frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
        frames = frames.to(device)
        print(f"  Tensor shape: {frames.shape}")

        # Encode to latent in batches
        with torch.no_grad():
            batch_size = 64
            latents = []
            for i in tqdm(range(0, len(frames), batch_size), desc=f"  Encoding"):
                batch = frames[i:i+batch_size]
                latent = vae.encode(batch).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
                latents.append(latent.cpu())

            latent_tensor = torch.cat(latents, dim=0)

        print(f"  Latent shape: {latent_tensor.shape}")

        # Save latent
        output_path = output_dir / f"{video_id}.pt"
        torch.save(latent_tensor, output_path)
        print(f"  Saved to {output_path}")

    print(f"\n✓ Done! Latents saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, required=True, help="Scene ID (e.g., 0001)")
    parser.add_argument("--data-root", type=str, required=True, help="DROID dataset root")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--svd-path", type=str, required=True, help="SVD model path")
    parser.add_argument("--device", type=str, default="cuda:1")

    args = parser.parse_args()

    generate_latents_for_scene(
        scene_id=args.scene_id,
        data_root=args.data_root,
        split=args.split,
        svd_path=args.svd_path,
        device=args.device
    )
