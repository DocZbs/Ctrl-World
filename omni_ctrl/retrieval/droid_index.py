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
    import transformers
    # Use getattr to avoid import-time errors with lazy loading
    CLIPProcessor = getattr(transformers, 'CLIPProcessor')
    CLIPModel = getattr(transformers, 'CLIPModel')
except (ImportError, AttributeError) as e:
    raise ImportError(f"Please install transformers: pip install transformers. Error: {e}")


@dataclass
class DROIDScenario:
    """Retrieved scenario from DROID dataset.

    Attributes:
        episode_id: Unique identifier for the episode
        initial_frames: Initial video latent frames (T, 4, 72, 40)
        initial_state: Initial cartesian pose (7,)
        initial_joints: Initial joint positions (7,)
        instruction: Original task instruction from DROID
        video_path: Path to original video
        latent_path: Path to pre-encoded latent
        real_initial_frames: Real RGB frames from video (3, H, W, 3), optional
    """

    episode_id: str
    initial_frames: np.ndarray
    initial_state: np.ndarray
    initial_joints: np.ndarray
    instruction: str
    video_path: str
    latent_path: str
    real_initial_frames: list = None  # List of 3 numpy arrays (H, W, 3)


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
        # Move CLIP model to CPU to save GPU memory for world model
        model = model.to('cpu')

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

                # Try multiple possible instruction field names
                instruction = data.get("language_instruction", "")
                if not instruction:
                    # Try 'texts' field (DROID format)
                    texts = data.get("texts", [])
                    if texts and len(texts) > 0:
                        instruction = texts[0]

                # Skip if still no instruction
                if not instruction or instruction.strip() == "":
                    continue

                # Encode instruction
                emb = text_encoder(instruction)
                embeddings.append(emb)

                # Store metadata
                # Try to get cartesian/joint positions from different possible fields
                cart_pos = data.get("states", [])  # DROID uses 'states' for cartesian positions
                if not cart_pos:
                    cart_pos = data.get("cartesian_position", [])
                if not cart_pos:
                    cart_pos = data.get("observation.state.cartesian_position", [])

                joint_pos = data.get("observation.state.joint_position", [])
                if not joint_pos:
                    joint_pos = data.get("joint_position", [])

                # Construct paths
                video_dir = Path(droid_path) / "videos" / "train" / json_path.stem
                latent_dir = Path(droid_path) / "latent_videos" / "train" / json_path.stem

                metadata.append({
                    "episode_id": json_path.stem,
                    "instruction": instruction,
                    "cartesian_position": cart_pos,
                    "joint_position": joint_pos,
                    "video_path": str(video_dir),
                    "latent_path": str(latent_dir),  # This is a directory containing latent .pt files
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
                if latent_path.exists() and latent_path.is_dir():
                    # Load all 3 camera views and concatenate (DROID has 3 camera views)
                    latent_files = [latent_path / f"{i}.pt" for i in range(3)]
                    if all(f.exists() for f in latent_files):
                        latents = [torch.load(f) for f in latent_files]
                        # Concatenate along height dimension (dim=2): (89, 4, 24, 40) x3 -> (89, 4, 72, 40)
                        combined_latent = torch.cat(latents, dim=2)
                        initial_frames = combined_latent[:6].numpy()  # First 6 frames as history (6, 4, 72, 40)
                    else:
                        initial_frames = np.zeros((6, 4, 72, 40))  # Placeholder with correct shape
                elif latent_path.exists() and latent_path.is_file():
                    # Handle case where it's a single file (legacy support)
                    latent = torch.load(latent_path)
                    if isinstance(latent, torch.Tensor):
                        # Assume already concatenated
                        initial_frames = latent[:6].numpy()
                    else:
                        initial_frames = np.zeros((6, 4, 72, 40))
                else:
                    initial_frames = np.zeros((6, 4, 72, 40))  # Placeholder with correct shape

                # Extract initial state
                cart_pos = meta.get("cartesian_position", [])
                joint_pos = meta.get("joint_position", [])

                # Ensure we have 7-dim cartesian state and joint positions
                if cart_pos and len(cart_pos) > 0:
                    initial_state = np.array(cart_pos[0])
                    # Ensure it's 7-dim (xyz, rpy, gripper)
                    if len(initial_state) < 7:
                        initial_state = np.pad(initial_state, (0, 7 - len(initial_state)))
                    elif len(initial_state) > 7:
                        initial_state = initial_state[:7]
                else:
                    initial_state = np.array([0.0] * 7)

                if joint_pos and len(joint_pos) > 0:
                    initial_joints = np.array(joint_pos[0])
                    # Ensure it's 7-dim
                    if len(initial_joints) < 7:
                        initial_joints = np.pad(initial_joints, (0, 7 - len(initial_joints)))
                    elif len(initial_joints) > 7:
                        initial_joints = initial_joints[:7]
                else:
                    initial_joints = np.array([0.0] * 7)

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
        # Move CLIP model to CPU to save GPU memory for world model
        model = model.to('cpu')

        def text_encoder(text):
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
            return features.detach().numpy()[0]

        print(f"Index loaded with {index.ntotal} vectors")
        return cls(index, metadata, text_encoder)
