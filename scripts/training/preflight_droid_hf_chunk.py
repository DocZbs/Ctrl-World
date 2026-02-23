#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_droid_root_and_chunk(dataset_path: Path) -> tuple[Path, str]:
    # Supported:
    #  - /path/to/droid_data
    #  - /path/to/droid_data/data/chunk-000
    if (dataset_path / "data").exists() and (dataset_path / "videos").exists():
        return dataset_path, "chunk-000"
    if dataset_path.name.startswith("chunk-") and dataset_path.parent.name == "data":
        root = dataset_path.parent.parent
        return root, dataset_path.name
    raise ValueError(
        "dataset_path must be either DROID root (contains data/ and videos/) "
        "or a chunk dir like .../data/chunk-000; got: "
        f"{dataset_path}"
    )


def _read_episode_columns(parquet_path: Path):
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "action.joint_position",
                "action.joint_velocity",
                "action.gripper_position",
            ],
        )
        def col(name):
            return np.asarray(table[name].to_pylist(), dtype=np.float32)
        return {k: col(k) for k in table.column_names}
    except Exception:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "action.joint_position",
                "action.joint_velocity",
                "action.gripper_position",
            ],
        )
        out = {}
        for k in df.columns:
            out[k] = np.asarray(df[k].tolist(), dtype=np.float32)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-path",
        required=True,
        type=str,
        help="DROID root (/.../droid_data) or chunk dir (/.../droid_data/data/chunk-000)",
    )
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--action-horizon", type=int, default=10)
    ap.add_argument("--require-joint-velocity", action="store_true")
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    root, chunk = _resolve_droid_root_and_chunk(dataset_path)
    parquet_dir = root / "data" / chunk
    videos_dir = root / "videos" / chunk

    if not parquet_dir.exists():
        raise FileNotFoundError(parquet_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(videos_dir)

    parquet_paths = sorted(parquet_dir.glob("episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episodes found under {parquet_dir}")

    if args.require_joint_velocity:
        os.environ["DROID_REQUIRE_JOINT_VELOCITY"] = "1"

    print("== DROID HF preflight ==")
    print(f"root={root}")
    print(f"chunk={chunk}")
    print(f"episodes={len(parquet_paths)}")
    print(f"require_joint_velocity={os.environ.get('DROID_REQUIRE_JOINT_VELOCITY','0')}")
    print("")

    # Quick schema sanity on a few episodes.
    n = min(args.num_episodes, len(parquet_paths))
    for p in parquet_paths[:n]:
        cols = _read_episode_columns(p)
        state = cols["observation.state"]
        action = cols["action"]
        joint_pos = cols["action.joint_position"]
        joint_vel = cols["action.joint_velocity"]
        grip = cols["action.gripper_position"].reshape(-1, 1)

        max_abs_action_minus_jointpos = float(np.max(np.abs(action[:, :7] - joint_pos)))
        max_abs_action_minus_grip = float(np.max(np.abs(action[:, 7:8] - grip)))
        vel_min = float(np.min(joint_vel))
        vel_max = float(np.max(joint_vel))

        print(
            f"{p.name}: "
            f"T={len(state)} | "
            f"max|action[:7]-joint_pos|={max_abs_action_minus_jointpos:.6f} | "
            f"max|action[7]-grip|={max_abs_action_minus_grip:.6f} | "
            f"joint_vel_range=[{vel_min:.3f},{vel_max:.3f}]"
        )

    print("")

    # Verify the actual training dataset path chooses the expected action source.
    from openpi.training.droid_hf_parquet_dataset import DroidHFParquetChunkDataset

    ds = DroidHFParquetChunkDataset(droid_root=root, chunk=chunk, action_horizon=args.action_horizon)
    source_counts = {}
    for ep in ds._episodes:  # noqa: SLF001 (debug-only)
        source_counts[ep.action_source] = source_counts.get(ep.action_source, 0) + 1
    print(f"action_source_breakdown={source_counts}")

    if args.require_joint_velocity and any(k != "action.joint_velocity+action.gripper_position" for k in source_counts):
        raise SystemExit(
            "Preflight failed: required joint_velocity but some episodes fell back to `action`.\n"
            f"breakdown={source_counts}"
        )

    # Small stat sanity: actions should look velocity-like (not absolute positions).
    # Use a few episodes to avoid big memory.
    sample_actions = []
    for ep in ds._episodes[: min(10, len(ds._episodes))]:  # noqa: SLF001
        sample_actions.append(ep.actions[:, :7])
    if sample_actions:
        a = np.concatenate(sample_actions, axis=0)
        mean = np.mean(a, axis=0)
        std = np.std(a, axis=0)
        abs_p99 = np.quantile(np.abs(a), 0.99, axis=0)
        print(f"joint_action_mean={mean.round(4).tolist()}")
        print(f"joint_action_std={std.round(4).tolist()}")
        print(f"joint_action_abs_p99={abs_p99.round(4).tolist()}")

    print("OK")


if __name__ == "__main__":
    main()

