#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_KEYS = [
    "pick_place",
    "reorientation",
    "articulation_manipulation",
    "tool_use",
    "deformable_object_manipulation",
]


@dataclass(frozen=True)
class EpisodeRef:
    chunk: str
    episode_index: int
    instruction: str
    num_frames: int
    num_samples: int


def _resolve_root_and_chunks(dataset_path: Path, chunks: list[str] | None) -> tuple[Path, list[str]]:
    dataset_path = dataset_path.resolve()

    if dataset_path.name.startswith("chunk-") and dataset_path.parent.name == "data":
        return dataset_path.parent.parent, [dataset_path.name]

    if not (dataset_path / "data").exists():
        raise ValueError(
            "dataset_path must be either DROID root (contains data/) or a chunk dir like .../data/chunk-000; "
            f"got {dataset_path}"
        )

    root = dataset_path
    if chunks:
        return root, chunks

    chunk_dirs = sorted(p.name for p in (root / "data").iterdir() if p.is_dir() and p.name.startswith("chunk-"))
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk-* dirs found under {root / 'data'}")
    return root, chunk_dirs


def _episode_index_from_name(p: Path) -> int:
    m = re.match(r"episode_(\d+)\.parquet$", p.name)
    if not m:
        raise ValueError(f"Unexpected episode parquet name: {p.name}")
    return int(m.group(1))


def _read_instruction_and_rows(parquet_path: Path) -> tuple[str, int]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(parquet_path)
        num_rows = int(pf.metadata.num_rows)
        try:
            table = pq.read_table(parquet_path, columns=["language_instruction"])
            if table.num_rows > 0:
                val = table["language_instruction"][0].as_py()
                inst = str(val or "").strip()
            else:
                inst = ""
        except Exception:
            inst = ""
        return inst, num_rows
    except Exception:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(parquet_path, columns=["language_instruction"])
        inst = str((df["language_instruction"].iloc[0] if len(df) > 0 else "") or "").strip()
        return inst, int(len(df))


def _norm(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def _is_tool_use(text: str) -> bool:
    if re.search(r"\buse\b.+\bto\b", text):
        return True

    tool_verbs = [
        "wipe",
        "clean",
        "scrub",
        "mop",
        "vacuum",
        "brush",
        "sweep",
        "stir",
        "scoop",
        "cut",
        "write",
        "draw",
        "plug",
        "unplug",
        "charge",
        "hang",
    ]
    tool_objects = [
        "cloth",
        "towel",
        "sponge",
        "brush",
        "mop",
        "vacuum",
        "charger",
        "cable",
        "rag",
        "paper towel",
    ]

    has_tool_verb = any(re.search(rf"\b{re.escape(v)}\b", text) for v in tool_verbs)
    has_tool_obj = any(o in text for o in tool_objects)
    return has_tool_verb and has_tool_obj


def classify_instruction(instruction: str) -> str | None:
    t = _norm(instruction)
    if not t:
        return None

    deformable_kw = [
        "cloth",
        "towel",
        "paper towel",
        "napkin",
        "tissue",
        "rag",
        "fabric",
        "bag",
        "carrier bag",
        "laundry",
        "rope",
        "string",
    ]

    articulation_obj = [
        "drawer",
        "door",
        "cabinet",
        "microwave",
        "fridge",
        "refrigerator",
        "oven",
        "dishwasher",
        "lid",
        "knob",
        "button",
        "switch",
        "lever",
        "handle",
        "faucet",
        "tap",
        "bin",
    ]
    articulation_verb = ["open", "close", "push", "pull", "press", "turn", "twist"]

    reorientation_kw = [
        "rotate",
        "reorient",
        "flip",
        "upright",
        "upside down",
        "turn over",
        "orientation",
    ]

    pick_place_verb = [
        "pick",
        "place",
        "put",
        "move",
        "take",
        "transfer",
        "stack",
        "insert",
        "remove",
    ]

    # Priority: deformable > tool use > articulation > reorientation > pick-place
    if _contains_any(t, deformable_kw):
        return "deformable_object_manipulation"

    if _is_tool_use(t):
        return "tool_use"

    if _contains_any(t, articulation_obj) and any(re.search(rf"\b{re.escape(v)}\b", t) for v in articulation_verb):
        return "articulation_manipulation"

    if _contains_any(t, reorientation_kw):
        return "reorientation"

    if any(re.search(rf"\b{re.escape(v)}\b", t) for v in pick_place_verb):
        return "pick_place"

    return None


def _sample_or_all(items: list[EpisodeRef], max_episodes: int | None, seed: int) -> list[EpisodeRef]:
    if max_episodes is None or len(items) <= max_episodes:
        return sorted(items, key=lambda x: (x.chunk, x.episode_index))
    rng = random.Random(seed)
    sampled = rng.sample(items, max_episodes)
    return sorted(sampled, key=lambda x: (x.chunk, x.episode_index))


def main() -> None:
    ap = argparse.ArgumentParser(description="Index DROID episodes into 5 instruction-driven task categories.")
    ap.add_argument("--dataset-path", required=True, type=str)
    ap.add_argument("--chunks", nargs="*", default=None)
    ap.add_argument("--action-horizon", type=int, default=15)
    ap.add_argument("--max-episodes-per-category", type=int, default=None)
    ap.add_argument("--min-episodes-per-category", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-instruction-k", type=int, default=15)
    ap.add_argument("--out", type=str, default="droid_instruction_task_index_000_009.json")
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    root, chunks = _resolve_root_and_chunks(Path(args.dataset_path), args.chunks)

    worklist: list[tuple[str, Path]] = []
    for c in chunks:
        parquet_dir = root / "data" / c
        worklist.extend((c, p) for p in sorted(parquet_dir.glob("episode_*.parquet")))

    if not worklist:
        raise FileNotFoundError(f"No episode_*.parquet found for chunks={chunks}")

    iterator: Any = worklist
    if args.progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(worklist, total=len(worklist), unit="ep", dynamic_ncols=True, leave=True)
        except Exception:
            iterator = worklist

    per_task_refs: dict[str, list[EpisodeRef]] = {k: [] for k in TASK_KEYS}
    per_task_instruction_counter: dict[str, Counter[str]] = {k: Counter() for k in TASK_KEYS}

    total_eps = 0
    labeled_eps = 0
    empty_instruction_eps = 0

    for chunk, parquet_path in iterator:
        total_eps += 1
        episode_index = _episode_index_from_name(parquet_path)
        instruction, num_rows = _read_instruction_and_rows(parquet_path)
        if not instruction.strip():
            empty_instruction_eps += 1
            continue

        task_key = classify_instruction(instruction)
        if task_key is None:
            continue

        labeled_eps += 1
        num_samples = max(int(num_rows) - int(args.action_horizon) + 1, 0)
        ref = EpisodeRef(
            chunk=chunk,
            episode_index=episode_index,
            instruction=instruction,
            num_frames=int(num_rows),
            num_samples=int(num_samples),
        )
        per_task_refs[task_key].append(ref)
        per_task_instruction_counter[task_key][instruction.strip()] += 1

    selected_refs: dict[str, list[EpisodeRef]] = {}
    for k, refs in per_task_refs.items():
        selected_refs[k] = _sample_or_all(refs, args.max_episodes_per_category, seed=args.seed)

    scenes: dict[str, dict[str, Any]] = {}
    for task_key in TASK_KEYS:
        refs = selected_refs[task_key]
        by_chunk: dict[str, list[int]] = defaultdict(list)
        total_frames = 0
        total_samples = 0
        for r in refs:
            by_chunk[r.chunk].append(r.episode_index)
            total_frames += r.num_frames
            total_samples += r.num_samples
        scenes[task_key] = {
            "num_episodes": len(refs),
            "total_frames": int(total_frames),
            "total_samples": int(total_samples),
            "by_chunk": {c: sorted(v) for c, v in sorted(by_chunk.items())},
            "top_instructions": [
                {"instruction": inst, "count": int(cnt)}
                for inst, cnt in per_task_instruction_counter[task_key].most_common(max(args.top_instruction_k, 0))
            ],
        }

    index = {
        "dataset_root": str(root),
        "chunks": chunks,
        "scene_col": "instruction_task_category",
        "action_horizon": int(args.action_horizon),
        "total_episodes": int(total_eps),
        "labeled_episodes": int(labeled_eps),
        "empty_instruction_episodes": int(empty_instruction_eps),
        "max_episodes_per_category": args.max_episodes_per_category,
        "task_keys": TASK_KEYS,
        # Keep `scenes` for direct compatibility with openpi.data_loader filtering.
        "scenes": scenes,
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("== DROID instruction task index ==")
    print(f"root={root}")
    print(f"chunks={chunks}")
    print(f"episodes_scanned={total_eps}")
    print(f"episodes_labeled={labeled_eps}")
    print(f"empty_instruction_episodes={empty_instruction_eps}")
    print(f"wrote={out_path}")
    print("")

    for task_key in TASK_KEYS:
        ne = scenes[task_key]["num_episodes"]
        ns = scenes[task_key]["total_samples"]
        mark = "OK" if ne >= int(args.min_episodes_per_category) else "LOW"
        print(f"  {task_key:30s} episodes={ne:5d} samples≈{ns:8d} [{mark}]")


if __name__ == "__main__":
    main()
