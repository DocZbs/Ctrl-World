"""Runtime bootstrap helpers for EvoW.

These helpers keep path bootstrapping and environment initialization in one
place so the package entrypoint, orchestrator, and policy wrappers behave
consistently whether EvoW is run as `python -m evow` or via direct module
scripts such as `python evow/run_evow.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = PROJECT_ROOT / "openpi"
OPENPI_CLIENT_ROOT = OPENPI_ROOT / "packages" / "openpi-client" / "src"


def _prepend_sys_path(path: Path) -> Path:
    """Prepend a path to `sys.path` once and return it."""
    resolved = path.resolve()
    resolved_str = str(resolved)
    if resolved_str not in sys.path:
        sys.path.insert(0, resolved_str)
    return resolved


def ensure_project_root_on_path() -> Path:
    """Ensure the Ctrl-World repo root is importable."""
    return _prepend_sys_path(PROJECT_ROOT)


def ensure_openpi_on_path() -> Path | None:
    """Ensure the in-repo `openpi` package is importable when present."""
    if not OPENPI_ROOT.exists():
        return None
    return _prepend_sys_path(OPENPI_ROOT)


def ensure_openpi_client_on_path() -> Path | None:
    """Ensure the local `openpi_client` package is importable when present."""
    if not OPENPI_CLIENT_ROOT.exists():
        return None
    return _prepend_sys_path(OPENPI_CLIENT_ROOT)


def configure_jax_environment() -> dict[str, str | None]:
    """Apply EvoW JAX runtime defaults before any JAX import happens."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    return {
        "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
    }


__all__ = [
    "PROJECT_ROOT",
    "OPENPI_ROOT",
    "OPENPI_CLIENT_ROOT",
    "configure_jax_environment",
    "ensure_openpi_client_on_path",
    "ensure_openpi_on_path",
    "ensure_project_root_on_path",
]
