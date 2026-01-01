"""FAISS index for DROID scenario retrieval."""

import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List

try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss-cpu or faiss-gpu: pip install faiss-cpu")

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")


@dataclass
class DROIDScenario:
    """Retrieved scenario from DROID dataset.

    Attributes:
        episode_id: Unique identifier for the episode
        initial_frames: Initial video frames (T, H, W, 3)
        initial_state: Initial cartesian pose (7,)
        initial_joints: Initial joint positions (7,)
        instruction: Original task instruction from DROID
        video_path: Path to original video
        latent_path: Path to pre-encoded latent
    """

    episode_id: str
    initial_frames: np.ndarray
    initial_state: np.ndarray
    initial_joints: np.ndarray
    instruction: str
    video_path: str
    latent_path: str


class DROIDIndex:
    """FAISS index for semantic search over DROID scenarios."""

    def __init__(self, index, metadata, text_encoder):
        """Initialize DROID index.

        Args:
            index: FAISS index for similarity search
            metadata: List of metadata dicts for each indexed item
            text_encoder: Function to encode text to embeddings
        """
        self.index = index
        self.metadata = metadata
        self.text_encoder = text_encoder

    @classmethod
    def build(cls, droid_path: str, clip_model_path: str):
        """Build FAISS index from DROID annotations.

        Args:
            droid_path: Path to DROID dataset root
            clip_model_path: Path or name of CLIP model

        Returns:
            DROIDIndex instance
        """
        print(f"Loading CLIP model from {clip_model_path}...")
        model = CLIPModel.from_pretrained(clip_model_path)
        processor = CLIPProcessor.from_pretrained(clip_model_path)

        def text_encoder(text):
            """Encode text using CLIP."""
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
            return features.detach().numpy()[0]

        # Load DROID annotations
        annotation_dir = Path(droid_path) / "annotation" / "train"
        if not annotation_dir.exists():
            raise FileNotFoundError(f"DROID annotation directory not found: {annotation_dir}")

        metadata = []
        embeddings = []

        print(f"Building index from {annotation_dir}...")
        annotation_files = list(annotation_dir.glob("*.json"))
        if not annotation_files:
            raise ValueError(f"No annotation files found in {annotation_dir}")

        for json_path in annotation_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)

                instruction = data.get("language_instruction", "")
                if not instruction:
                    continue

                # Encode instruction
                emb = text_encoder(instruction)
                embeddings.append(emb)

                # Store metadata
                metadata.append({
                    "episode_id": json_path.stem,
                    "instruction": instruction,
                    "cartesian_position": data.get("cartesian_position", []),
                    "joint_position": data.get("joint_position", []),
                    "video_path": str(Path(droid_path) / "videos" / json_path.stem),
                    "latent_path": str(Path(droid_path) / "latent_videos" / f"{json_path.stem}.pt"),
                })
            except Exception as e:
                print(f"Warning: Failed to process {json_path.name}: {e}")
                continue

        if not embeddings:
            raise ValueError("No valid episodes found in DROID dataset")

        # Build FAISS index
        embeddings = np.array(embeddings).astype("float32")
        print(f"Building FAISS index with {len(embeddings)} episodes...")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        print(f"Index built successfully with {index.ntotal} vectors")
        return cls(index, metadata, text_encoder)

    def search(self, query_text: str, top_k: int = 5) -> List[DROIDScenario]:
        """Search for relevant scenarios using text query.

        Args:
            query_text: Natural language query
            top_k: Number of top results to return

        Returns:
            List of DROIDScenario objects
        """
        # Encode query
        query_emb = self.text_encoder(query_text).astype("float32").reshape(1, -1)

        # Search in FAISS
        distances, indices = self.index.search(query_emb, top_k)

        scenarios = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            try:
                # Load initial state from latent if available
                latent_path = Path(meta["latent_path"])
                if latent_path.exists():
                    latent = torch.load(latent_path)
                    if isinstance(latent, torch.Tensor):
                        initial_frames = latent[:6].numpy()  # First 6 frames as history
                    else:
                        initial_frames = np.zeros((6, 4, 24, 40))  # Placeholder latent shape
                else:
                    initial_frames = np.zeros((6, 4, 24, 40))  # Placeholder

                # Extract initial state
                cart_pos = meta.get("cartesian_position", [])
                joint_pos = meta.get("joint_position", [])

                initial_state = np.array(cart_pos[0] if cart_pos else [0.0] * 7)
                initial_joints = np.array(joint_pos[0] if joint_pos else [0.0] * 7)

                scenario = DROIDScenario(
                    episode_id=meta["episode_id"],
                    initial_frames=initial_frames,
                    initial_state=initial_state,
                    initial_joints=initial_joints,
                    instruction=meta["instruction"],
                    video_path=meta["video_path"],
                    latent_path=meta["latent_path"],
                )
                scenarios.append(scenario)

            except Exception as e:
                print(f"Warning: Failed to load scenario {meta['episode_id']}: {e}")
                continue

        return scenarios

    def save(self, save_path: str):
        """Save index to disk.

        Args:
            save_path: Directory to save index and metadata
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))

        # Save metadata
        with open(save_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Index saved to {save_path}")

    @classmethod
    def load(cls, load_path: str, clip_model_path: str):
        """Load index from disk.

        Args:
            load_path: Directory containing saved index
            clip_model_path: Path or name of CLIP model

        Returns:
            DROIDIndex instance
        """
        load_path = Path(load_path)

        # Load FAISS index
        index = faiss.read_index(str(load_path / "index.faiss"))

        # Load metadata
        with open(load_path / "metadata.json") as f:
            metadata = json.load(f)

        # Recreate text encoder
        print(f"Loading CLIP model from {clip_model_path}...")
        model = CLIPModel.from_pretrained(clip_model_path)
        processor = CLIPProcessor.from_pretrained(clip_model_path)

        def text_encoder(text):
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
            return features.detach().numpy()[0]

        print(f"Index loaded with {index.ntotal} vectors")
        return cls(index, metadata, text_encoder)
