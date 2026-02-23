#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict


@dataclass(frozen=True)
class EpisodeRef:
    chunk: str
    episode_index: int
    num_frames: int
    num_samples: int


def _resolve_root_and_chunks(dataset_path: Path, chunks: list[str] | None) -> tuple[Path, list[str]]:
    dataset_path = dataset_path.resolve()

    # Supported:
    #  - /path/to/droid_data
    #  - /path/to/droid_data/data/chunk-000
    if dataset_path.name.startswith("chunk-") and dataset_path.parent.name == "data":
        root = dataset_path.parent.parent
        return root, [dataset_path.name]

    if not (dataset_path / "data").exists():
        raise ValueError(
            "dataset_path must be either DROID root (contains data/) "
            "or a chunk dir like .../data/chunk-000; got: "
            f"{dataset_path}"
        )

    root = dataset_path
    if chunks:
        return root, chunks

    chunk_dirs = sorted(p.name for p in (root / "data").iterdir() if p.is_dir() and p.name.startswith("chunk-"))
    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk-* dirs found under {root / 'data'}")
    return root, chunk_dirs


def _choose_scene_col(sample_parquet: Path, scene_col: str | None) -> str:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "pyarrow is required to read parquet schemas.\n"
            "Try running with the OpenPI venv python:\n"
            "  ./openpi/.venv/bin/python scripts/training/index_droid_scenes.py ..."
        ) from e

    pf = pq.ParquetFile(sample_parquet)
    names = set(pf.schema_arrow.names)

    if scene_col is not None:
        if scene_col not in names:
            raise ValueError(
                f"scene_col={scene_col!r} not found in parquet schema for {sample_parquet}.\n"
                f"Available columns include: {sorted(list(names))[:50]} ..."
            )
        return scene_col

    # Heuristic defaults (DROID HF dump often uses `building` as a scene/environment identifier).
    preferred = [
        "scene",
        "scene_id",
        "scene_name",
        "building",
        "room",
        "environment",
        "env",
        "task_category",
    ]
    for c in preferred:
        if c in names:
            return c

    # Fallback: pick the first column containing a scene-like keyword.
    for n in pf.schema_arrow.names:
        if re.search(r"(scene|building|room|environment|env)", n, flags=re.IGNORECASE):
            return n

    raise ValueError(
        "Could not auto-detect a scene column. Please pass --scene-col explicitly.\n"
        f"Available columns include: {sorted(list(names))[:50]} ..."
    )


def _normalize_key(value: Any) -> str:
    if value is None:
        return "__MISSING__"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return str(value)


def _episode_index_from_name(p: Path) -> int:
    m = re.match(r"episode_(\d+)\.parquet$", p.name)
    if not m:
        raise ValueError(f"Unexpected episode parquet name: {p.name}")
    return int(m.group(1))


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = slug.strip("._-")
    return slug or "scene"


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Idempotent: skip if already present.
        return

    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unknown link mode: {mode}")


def _materialize_subset(
    *,
    root: Path,
    out_root: Path,
    episode_refs: list[EpisodeRef],
    link_mode: str,
    videos_dirname: str,
    video_subdirs: list[str],
    merge_to_chunk: str | None,
    mapping_out: Path | None,
    skip_missing: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    out_root = out_root.resolve()
    if out_root.exists():
        if overwrite:
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(f"Output root already exists: {out_root} (pass --overwrite to replace)")

    mapping_rows: list[dict[str, Any]] = []
    skipped = 0
    written = 0
    if merge_to_chunk is None:
        for ref in episode_refs:
            ep = ref.episode_index
            parquet_src = root / "data" / ref.chunk / f"episode_{ep:06d}.parquet"
            parquet_dst = out_root / "data" / ref.chunk / parquet_src.name
            if not parquet_src.exists():
                if skip_missing:
                    skipped += 1
                    continue
                raise FileNotFoundError(parquet_src)

            ops: list[tuple[Path, Path]] = [(parquet_src, parquet_dst)]
            for sub in video_subdirs:
                mp4_src = root / videos_dirname / ref.chunk / sub / f"episode_{ep:06d}.mp4"
                mp4_dst = out_root / videos_dirname / ref.chunk / sub / mp4_src.name
                if not mp4_src.exists():
                    if skip_missing:
                        ops = []
                        skipped += 1
                        break
                    raise FileNotFoundError(mp4_src)
                ops.append((mp4_src, mp4_dst))

            if not ops:
                continue

            if dry_run:
                for s, d in ops:
                    print(f"[dry-run] {link_mode}: {d} -> {s}")
                continue

            for s, d in ops:
                _link_or_copy(s, d, link_mode)
            written += 1
        if skipped:
            print(f"Materialize summary: wrote_episodes={written} skipped_missing={skipped}")
        return

    # Merge all selected episodes into a single target chunk (renumber episodes to avoid collisions).
    target_chunk = merge_to_chunk
    new_ep = 0
    for ref in episode_refs:
        src_ep = ref.episode_index
        parquet_src = root / "data" / ref.chunk / f"episode_{src_ep:06d}.parquet"
        parquet_dst = out_root / "data" / target_chunk / f"episode_{new_ep:06d}.parquet"
        if not parquet_src.exists():
            if skip_missing:
                skipped += 1
                continue
            raise FileNotFoundError(parquet_src)

        ops: list[tuple[Path, Path]] = [(parquet_src, parquet_dst)]
        for sub in video_subdirs:
            mp4_src = root / videos_dirname / ref.chunk / sub / f"episode_{src_ep:06d}.mp4"
            mp4_dst = out_root / videos_dirname / target_chunk / sub / f"episode_{new_ep:06d}.mp4"
            if not mp4_src.exists():
                if skip_missing:
                    ops = []
                    skipped += 1
                    break
                raise FileNotFoundError(mp4_src)
            ops.append((mp4_src, mp4_dst))

        if not ops:
            continue

        mapping_rows.append(
            {
                "new_chunk": target_chunk,
                "new_episode_index": new_ep,
                "src_chunk": ref.chunk,
                "src_episode_index": src_ep,
                "num_frames": ref.num_frames,
                "num_samples": ref.num_samples,
            }
        )

        if dry_run:
            for s, d in ops:
                print(f"[dry-run] {link_mode}: {d} -> {s}")
            continue

        for s, d in ops:
            _link_or_copy(s, d, link_mode)
        written += 1
        new_ep += 1

    if mapping_out is not None:
        mapping_out = mapping_out.resolve()
        mapping_out.parent.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            mapping_out.write_text(json.dumps(mapping_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if skipped:
        print(f"Materialize summary: wrote_episodes={written} skipped_missing={skipped}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Index DROID episodes by a parquet scene/environment column.")
    ap.add_argument(
        "--dataset-path",
        required=True,
        type=str,
        help="DROID root (/.../droid_data) or chunk dir (/.../droid_data/data/chunk-000)",
    )
    ap.add_argument(
        "--chunks",
        nargs="*",
        default=None,
        help="Restrict to these chunks (e.g. chunk-000 chunk-001). Only applies when dataset-path is a root.",
    )
    ap.add_argument(
        "--scene-col",
        default=None,
        help="Parquet column name to group by (default: auto-detect; common: building).",
    )
    ap.add_argument("--action-horizon", type=int, default=10, help="Used to estimate sample count per episode (T-H+1).")
    ap.add_argument(
        "--out",
        type=str,
        default="droid_scene_index.json",
        help="Where to write the index JSON (file path). If a directory is provided, writes <scene_col>_index.json inside it.",
    )
    ap.add_argument("--top-k", type=int, default=20, help="Print top-K scenes by total_samples.")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a progress bar while scanning episodes.",
    )
    ap.add_argument("--scene", type=str, default=None, help="If set, print/export just this scene.")
    ap.add_argument(
        "--episodes-out",
        type=str,
        default=None,
        help="Write selected episode parquet paths (chunk/episode_XXXXXX.parquet) to this file.",
    )

    # Optional: create a subset dataset directory (symlink/hardlink/copy) for training.
    ap.add_argument(
        "--materialize-root",
        type=str,
        default=None,
        help="If set, create a new DROID root here containing only the selected --scene episodes.",
    )
    ap.add_argument(
        "--merge-to-chunk",
        type=str,
        default=None,
        help="If set, materialize ALL selected episodes into a single chunk (renumbered) with this name "
        "(e.g. chunk-000). Useful for training on a single merged chunk.",
    )
    ap.add_argument(
        "--mapping-out",
        type=str,
        default=None,
        help="When using --merge-to-chunk, write JSON mapping (new->src episode ids) to this path.",
    )
    ap.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--videos-dirname", type=str, default="videos", help="Videos directory name under the root.")
    ap.add_argument(
        "--video-subdirs",
        nargs="*",
        default=[
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
            "observation.images.wrist_left",
        ],
        help="Relative video subdirs under <root>/<videos-dirname>/<chunk>/ (HF DROID layout).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting --materialize-root if it exists.")
    ap.add_argument("--dry-run", action="store_true", help="Don't write files (use with --materialize-root).")
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="When materializing, skip episodes with missing parquet/video files instead of failing.",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    root, chunks = _resolve_root_and_chunks(dataset_path, args.chunks)

    # Find a sample parquet to infer the schema.
    sample_parquet = None
    for c in chunks:
        candidates = sorted((root / "data" / c).glob("episode_*.parquet"))
        if candidates:
            sample_parquet = candidates[0]
            break
    if sample_parquet is None:
        raise FileNotFoundError(f"No episode_*.parquet found under {root / 'data' / chunks[0]}")

    scene_col = _choose_scene_col(sample_parquet, args.scene_col)

    # Resolve output path early so we don't scan the full dataset and then fail at write time.
    out_path = Path(args.out).expanduser()
    out_arg = str(args.out).strip()
    if out_arg.endswith(("/", os.sep)) or (out_path.exists() and out_path.is_dir()):
        out_dir = out_path
        out_path = out_dir / f"{scene_col}_index.json"
        print(f"Info: --out is a directory; writing index to {out_path}")

    # scene -> chunk -> list[episode_index]
    scenes: DefaultDict[str, DefaultDict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    scene_frames: DefaultDict[str, int] = defaultdict(int)
    scene_samples: DefaultDict[str, int] = defaultdict(int)
    scene_episode_refs: DefaultDict[str, list[EpisodeRef]] = defaultdict(list)

    action_horizon = int(args.action_horizon)
    total_episodes = 0
    total_samples = 0

    import pyarrow.parquet as pq  # type: ignore

    # Build worklist so we can show an overall progress bar.
    worklist: list[tuple[str, Path]] = []
    for chunk in chunks:
        parquet_dir = root / "data" / chunk
        parquet_paths = sorted(parquet_dir.glob("episode_*.parquet"))
        if not parquet_paths:
            continue
        worklist.extend((chunk, p) for p in parquet_paths)

    if args.progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            it = tqdm(worklist, total=len(worklist), unit="ep", dynamic_ncols=True, leave=True)
        except Exception:
            it = worklist
    else:
        it = worklist

    for chunk, p in it:
        total_episodes += 1
        ep = _episode_index_from_name(p)

        try:
            pf = pq.ParquetFile(p)
            t = int(pf.metadata.num_rows)
            if pf.num_row_groups > 0:
                table = pf.read_row_group(0, columns=[scene_col])
                arr = table.column(0)
                scene_val = arr.chunk(0)[0].as_py() if arr.num_chunks and len(arr.chunk(0)) else None
            else:
                scene_val = None
        except Exception:
            t = 0
            scene_val = None

        scene_key = _normalize_key(scene_val)
        num_samples = max(t - action_horizon + 1, 0)

        scenes[scene_key][chunk].append(ep)
        scene_frames[scene_key] += t
        scene_samples[scene_key] += num_samples
        scene_episode_refs[scene_key].append(EpisodeRef(chunk=chunk, episode_index=ep, num_frames=t, num_samples=num_samples))
        total_samples += num_samples

    # Sort episode lists for stable output.
    scenes_sorted: dict[str, dict[str, list[int]]] = {}
    for scene_key in sorted(scenes.keys()):
        scenes_sorted[scene_key] = {c: sorted(eps) for c, eps in sorted(scenes[scene_key].items())}

    index = {
        "dataset_root": str(root),
        "chunks": chunks,
        "scene_col": scene_col,
        "action_horizon": action_horizon,
        "total_episodes": total_episodes,
        "total_samples": total_samples,
        "scenes": {
            scene_key: {
                "num_episodes": sum(len(eps) for eps in scenes_sorted[scene_key].values()),
                "total_frames": int(scene_frames[scene_key]),
                "total_samples": int(scene_samples[scene_key]),
                "by_chunk": scenes_sorted[scene_key],
            }
            for scene_key in scenes_sorted.keys()
        },
    }

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Print summary.
    print("== DROID scene index ==")
    print(f"root={root}")
    print(f"chunks={chunks}")
    print(f"scene_col={scene_col}")
    print(f"episodes={total_episodes}")
    print(f"samples≈{total_samples} (using action_horizon={action_horizon})")
    print(f"wrote={out_path}")
    print("")

    # Top-K scenes by samples.
    ranked = sorted(scene_samples.items(), key=lambda kv: kv[1], reverse=True)
    k = max(int(args.top_k), 0)
    if k > 0:
        print(f"Top-{min(k, len(ranked))} scenes by total_samples:")
        for scene_key, ns in ranked[:k]:
            ne = len(scene_episode_refs[scene_key])
            print(f"  {scene_key}: episodes={ne}, total_samples≈{ns}")
        print("")

    selected_scene = args.scene
    if selected_scene is not None:
        # Accept either raw scene string or its normalized representation.
        key = selected_scene if selected_scene in scene_episode_refs else _normalize_key(selected_scene)
        if key not in scene_episode_refs:
            raise SystemExit(f"Scene not found: {selected_scene!r}. See {out_path} for available keys.")

        refs = sorted(scene_episode_refs[key], key=lambda r: (r.chunk, r.episode_index))
        print(f"Selected scene: {key}")
        print(f"episodes={len(refs)} total_samples≈{scene_samples[key]}")

        if args.episodes_out is not None:
            eps_out = Path(args.episodes_out).resolve()
            eps_out.parent.mkdir(parents=True, exist_ok=True)
            eps_out.write_text(
                "".join(f"{r.chunk}/episode_{r.episode_index:06d}.parquet\n" for r in refs),
                encoding="utf-8",
            )
            print(f"wrote episode list: {eps_out}")

        if args.materialize_root is not None:
            out_root = Path(args.materialize_root)
            # If user passes a directory that looks like a file (no trailing slash), keep it.
            # Otherwise, if they pass "..." we don't auto-append scene name; they can choose.
            print(f"Materializing subset dataset under: {out_root.resolve()}")
            mapping_out = Path(args.mapping_out) if args.mapping_out is not None else None
            _materialize_subset(
                root=root,
                out_root=out_root,
                episode_refs=refs,
                link_mode=args.link_mode,
                videos_dirname=args.videos_dirname,
                video_subdirs=args.video_subdirs,
                merge_to_chunk=args.merge_to_chunk,
                mapping_out=mapping_out,
                skip_missing=args.skip_missing,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print("Dry-run complete.")
            else:
                print("Done.")


if __name__ == "__main__":
    main()
