"""FAISS index for DROID scenario retrieval."""

import json
import numpy as np
import torch
import cv2
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

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
        reference_latents: Optional latent sequence for GT video visualization,
            shape (T, 4, 72, 40)
    """

    episode_id: str
    initial_frames: np.ndarray
    initial_state: np.ndarray
    initial_joints: np.ndarray
    instruction: str
    video_path: str
    latent_path: str
    real_initial_frames: list = None  # List of 3 numpy arrays (H, W, 3)
    reference_latents: Optional[np.ndarray] = None


class DROIDIndex:
    """FAISS index for semantic search over DROID scenarios."""

    _INDEX_CAMERA_PRIORITY = (
        "exterior_1_left",
        "exterior_2_left",
        "wrist_left",
    )

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

    @staticmethod
    def _read_first_frame_from_video(cam_video: Path) -> Optional[np.ndarray]:
        """Read first RGB frame from a video file.

        Returns normalized float32 frame (H, W, 3) in [0, 1], or None on failure.
        """
        read_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-i",
            str(cam_video),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        read_out = subprocess.run(read_cmd, capture_output=True)
        if read_out.returncode != 0 or not read_out.stdout:
            print(f"Warning: Failed to decode first frame: {cam_video}")
            return None

        frame_data = np.frombuffer(read_out.stdout, dtype=np.uint8)
        frame_bgr = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"Warning: Failed to parse first frame image: {cam_video}")
            return None

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32) / 255.0

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @classmethod
    def _load_clip_components(cls, clip_model_path: str):
        print(f"Loading CLIP model from {clip_model_path}...")
        model = CLIPModel.from_pretrained(clip_model_path)
        processor = CLIPProcessor.from_pretrained(clip_model_path)
        model = model.to('cpu')
        model.eval()

        def text_encoder(text: str) -> np.ndarray:
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
                features = cls._l2_normalize(features)
            return features.detach().cpu().numpy()[0].astype(np.float32)

        def image_encoder(image: np.ndarray) -> np.ndarray:
            inputs = processor(images=[image], return_tensors="pt", do_rescale=False)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = cls._l2_normalize(features)
            return features.detach().cpu().numpy()[0].astype(np.float32)

        return text_encoder, image_encoder

    @classmethod
    def _read_anchor_frame(cls, video_dir: Path, episode_id: int) -> tuple[Optional[np.ndarray], Optional[str]]:
        for cam in cls._INDEX_CAMERA_PRIORITY:
            cam_video = video_dir / f"observation.images.{cam}" / f"episode_{episode_id:06d}.mp4"
            if not cam_video.exists():
                continue
            frame = cls._read_first_frame_from_video(cam_video)
            if frame is not None:
                return frame, cam
        return None, None

    @staticmethod
    def _resolve_parquet_path(meta: dict) -> Optional[Path]:
        parquet_str = meta.get("parquet_path", "")
        if parquet_str:
            parquet_path = Path(parquet_str)
            if parquet_path.exists():
                return parquet_path

        episode_id = str(meta.get("episode_id", ""))
        if not episode_id.isdigit():
            return None

        video_path = Path(meta.get("video_path", ""))
        if not video_path.exists():
            return None

        droid_root = video_path.parent.parent
        chunk_name = video_path.name
        parquet_path = droid_root / "data" / chunk_name / f"episode_{int(episode_id):06d}.parquet"
        if parquet_path.exists():
            return parquet_path
        return None

    @staticmethod
    def _load_initial_state_from_parquet(parquet_path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            import pandas as pd
        except ImportError:
            return None, None

        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            return None, None
        if df is None or len(df) == 0:
            return None, None

        row0 = df.iloc[0]

        joint8 = None
        if "observation.state" in df.columns:
            state = np.asarray(row0["observation.state"], dtype=np.float32).reshape(-1)
            if state.size >= 8:
                joint8 = state[:8]
            elif state.size == 7:
                joint8 = np.concatenate([state, np.zeros((1,), dtype=np.float32)], axis=0)

        if joint8 is None and "observation.state.joint_position" in df.columns:
            joints = np.asarray(row0["observation.state.joint_position"], dtype=np.float32).reshape(-1)
            if joints.size >= 7:
                joints = joints[:7]
                if "observation.state.gripper_position" in df.columns:
                    gripper_arr = np.asarray(row0["observation.state.gripper_position"], dtype=np.float32).reshape(-1)
                    gripper = float(gripper_arr[0]) if gripper_arr.size > 0 else 0.0
                else:
                    gripper = 0.0
                joint8 = np.concatenate([joints, np.array([gripper], dtype=np.float32)], axis=0)

        cart7 = None
        if "observation.state.cartesian_position" in df.columns:
            cart = np.asarray(row0["observation.state.cartesian_position"], dtype=np.float32).reshape(-1)
            if cart.size >= 6:
                gripper = float(joint8[7]) if joint8 is not None and joint8.size >= 8 else 0.0
                cart7 = np.concatenate([cart[:6], np.array([gripper], dtype=np.float32)], axis=0)

        if joint8 is not None:
            joint8 = joint8.astype(np.float32)
        if cart7 is not None:
            cart7 = cart7.astype(np.float32)

        return cart7, joint8

    @classmethod
    def build(cls, droid_path: str, clip_model_path: str, max_episodes: int = None):
        """Build FAISS index from DROID annotations.

        Args:
            droid_path: Path to DROID dataset root
            clip_model_path: Path or name of CLIP model
            max_episodes: Maximum number of episodes to index (None for all)

        Returns:
            DROIDIndex instance
        """
        text_encoder, image_encoder = cls._load_clip_components(clip_model_path)

        # Load DROID annotations
        annotation_dir = Path(droid_path) / "annotation" / "train"
        if not annotation_dir.exists():
            raise FileNotFoundError(f"DROID annotation directory not found: {annotation_dir}")

        metadata = []
        embeddings = []

        print(f"Building index from {annotation_dir}...")
        annotation_files = sorted(annotation_dir.glob("*.json"), key=lambda x: int(x.stem))
        if not annotation_files:
            raise ValueError(f"No annotation files found in {annotation_dir}")

        if max_episodes is not None and len(annotation_files) > max_episodes:
            print(f"Limiting index to {max_episodes} episodes (out of {len(annotation_files)})")
            annotation_files = annotation_files[:max_episodes]

        print(f"Found {len(annotation_files)} annotation files. Encoding visual anchors with CLIP...")
        from tqdm import tqdm
        for json_path in tqdm(annotation_files, desc="Building CLIP index"):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Failed to read {json_path.name}: {e}")
                continue

            # Try multiple possible instruction field names (kept as metadata only)
            instruction = data.get("language_instruction", "")
            if not instruction:
                texts = data.get("texts", [])
                if texts and len(texts) > 0:
                    instruction = texts[0]

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

            # Construct paths - DROID uses chunk-based structure
            episode_id = json_path.stem
            if not episode_id.isdigit():
                print(f"Warning: Non-numeric episode id in {json_path.name}, skipping")
                continue

            # Determine chunk number from episode_id (e.g., 074572 -> chunk-074)
            chunk_num = int(episode_id) // 1000
            chunk_name = f"chunk-{chunk_num:03d}"

            video_dir = Path(droid_path) / "videos" / chunk_name

            anchor_frame, anchor_camera = cls._read_anchor_frame(video_dir, int(episode_id))
            if anchor_frame is None:
                continue

            emb = image_encoder(anchor_frame)
            embeddings.append(emb)

            # Latent dirs may be named either "<episode_id>" or "episode_<id:06d>".
            latent_root = Path(droid_path) / "latents" / chunk_name
            latent_dir_plain = latent_root / episode_id
            latent_dir_prefixed = latent_root / f"episode_{int(episode_id):06d}"
            if latent_dir_plain.exists():
                latent_dir = latent_dir_plain
            elif latent_dir_prefixed.exists():
                latent_dir = latent_dir_prefixed
            else:
                # Prefer canonical prefixed form in metadata; loader has fallback resolution.
                latent_dir = latent_dir_prefixed

            parquet_path = Path(droid_path) / "data" / chunk_name / f"episode_{int(episode_id):06d}.parquet"

            metadata.append({
                "episode_id": episode_id,
                "instruction": instruction,
                "cartesian_position": cart_pos,
                "joint_position": joint_pos,
                "video_path": str(video_dir),
                "latent_path": str(latent_dir),  # This is a directory containing latent .pt files
                "parquet_path": str(parquet_path),
                "retrieval_camera": anchor_camera,
            })

        if not embeddings:
            raise ValueError("No valid episodes found in DROID dataset")

        # Build FAISS index
        embeddings = np.array(embeddings).astype("float32")
        print(f"Building FAISS index with {len(embeddings)} visual anchors...")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        print(f"Index built successfully with {index.ntotal} vectors")
        return cls(index, metadata, text_encoder)

    def search_with_scores(self, query_text: str, top_k: int = 20):
        """Search and return scenarios with similarity scores.

        Args:
            query_text: Natural language query
            top_k: Number of candidates

        Returns:
            Tuple of (List[DROIDScenario], np.ndarray of similarity scores).
            Similarities are negated L2 distances (higher = more similar).
        """
        query_emb = self.text_encoder(query_text).astype("float32").reshape(1, -1)
        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_emb, actual_k)

        # Convert L2 distances to similarity (negate so higher = better)
        similarities = -distances[0]

        scenarios = []
        valid_sims = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            scenario = self._build_scenario(meta)
            scenarios.append(scenario)
            valid_sims.append(similarities[rank])

        return scenarios, np.array(valid_sims, dtype=np.float32)

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
            scenario = self._build_scenario(meta)
            scenarios.append(scenario)

        return scenarios

    def _build_scenario(self, meta: dict) -> DROIDScenario:
        """Build a DROIDScenario from metadata dict.

        Args:
            meta: Metadata dictionary for one episode

        Returns:
            DROIDScenario instance
        """
        # Load initial state from latent if available
        latent_path = Path(meta["latent_path"])

        # Backward compatibility for indexes that stored non-prefixed episode IDs.
        # Example: .../latents/chunk-000/1  -> .../latents/chunk-000/episode_000001
        if not latent_path.exists():
            episode_id_str = str(meta.get("episode_id", ""))
            if episode_id_str.isdigit():
                candidate = latent_path.parent / f"episode_{int(episode_id_str):06d}"
                if candidate.exists():
                    latent_path = candidate
        initial_frames = None
        real_initial_frames = None
        reference_latents = None

        if latent_path.exists() and latent_path.is_dir():
            latent_files = [latent_path / f"{i}.pt" for i in range(3)]
            if all(f.exists() for f in latent_files):
                latents = [torch.load(f) for f in latent_files]
                combined_latent = torch.cat(latents, dim=2)
                initial_frames = combined_latent[:6].numpy()
                reference_latents = combined_latent.numpy()
        elif latent_path.exists() and latent_path.is_file():
            latent = torch.load(latent_path)
            if isinstance(latent, torch.Tensor):
                initial_frames = latent[:6].numpy()
                reference_latents = latent.numpy()

        # If latent not found, load from video and encode on-the-fly
        if initial_frames is None:
            video_path = Path(meta["video_path"])
            episode_id = meta["episode_id"]

            # Try to load initial frames from video
            camera_names = ["exterior_1_left", "exterior_2_left", "wrist_left"]
            video_frames = []

            for cam in camera_names:
                cam_video = video_path / f"observation.images.{cam}" / f"episode_{int(episode_id):06d}.mp4"
                if cam_video.exists():
                    frame_normalized = self._read_first_frame_from_video(cam_video)
                    if frame_normalized is not None:
                        video_frames.append(frame_normalized)
                else:
                    print(f"Warning: Video not found: {cam_video}")

            if len(video_frames) == 3:
                # Store real frames for later use
                real_initial_frames = video_frames
                # Use zeros as placeholder for latent (will be encoded on first use)
                initial_frames = np.zeros((6, 4, 72, 40))
            else:
                # Fallback to zeros
                initial_frames = np.zeros((6, 4, 72, 40))

        # Extract initial state: prefer exact t=0 state from episode parquet
        initial_state = None
        initial_joints = None

        parquet_path = self._resolve_parquet_path(meta)
        if parquet_path is not None:
            parquet_state, parquet_joints = self._load_initial_state_from_parquet(parquet_path)
            if parquet_state is not None:
                initial_state = parquet_state
            if parquet_joints is not None:
                initial_joints = parquet_joints[:7]

        # Fallback to metadata fields if parquet is unavailable
        if initial_state is None:
            cart_pos = meta.get("cartesian_position", [])
            if cart_pos and len(cart_pos) > 0:
                initial_state = np.array(cart_pos[0], dtype=np.float32)
                if len(initial_state) < 7:
                    initial_state = np.pad(initial_state, (0, 7 - len(initial_state)))
                elif len(initial_state) > 7:
                    initial_state = initial_state[:7]
            else:
                initial_state = np.array([0.0] * 7, dtype=np.float32)

        if initial_joints is None:
            joint_pos = meta.get("joint_position", [])
            if joint_pos and len(joint_pos) > 0:
                initial_joints = np.array(joint_pos[0], dtype=np.float32)
                if len(initial_joints) < 7:
                    initial_joints = np.pad(initial_joints, (0, 7 - len(initial_joints)))
                elif len(initial_joints) > 7:
                    initial_joints = initial_joints[:7]
            else:
                initial_joints = np.array([0.0] * 7, dtype=np.float32)

        return DROIDScenario(
            episode_id=meta["episode_id"],
            initial_frames=initial_frames,
            initial_state=initial_state,
            initial_joints=initial_joints,
            instruction=meta["instruction"],
            video_path=meta["video_path"],
            latent_path=meta["latent_path"],
            real_initial_frames=real_initial_frames,
            reference_latents=reference_latents,
        )

    def save(self, save_path: str):
        """Save index to disk.

        Args:
            save_path: Directory to save index and metadata
        """
        save_path = Path(save_path)
        if save_path.suffix == ".faiss":
            save_path.parent.mkdir(parents=True, exist_ok=True)
            index_file = save_path
            meta_file = save_path.with_suffix(".json")
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            index_file = save_path / "index.faiss"
            meta_file = save_path / "metadata.json"

        # Save FAISS index
        faiss.write_index(self.index, str(index_file))

        # Save metadata
        with open(meta_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Index saved to {index_file}")

    @classmethod
    def load(cls, load_path: str, clip_model_path: str):
        """Load index from disk.

        Args:
            load_path: Directory or .faiss file containing saved index
            clip_model_path: Path or name of CLIP model

        Returns:
            DROIDIndex instance
        """
        load_path = Path(load_path)
        if load_path.suffix == ".faiss":
            index_file = load_path
            meta_file = load_path.with_suffix(".json")
        else:
            index_file = load_path / "index.faiss"
            meta_file = load_path / "metadata.json"

        # Load FAISS index
        index = faiss.read_index(str(index_file))

        # Load metadata
        with open(meta_file) as f:
            metadata = json.load(f)

        # Recreate CLIP text encoder (query text -> CLIP joint space)
        text_encoder, _ = cls._load_clip_components(clip_model_path)

        print(f"Index loaded with {index.ntotal} vectors")
        return cls(index, metadata, text_encoder)
