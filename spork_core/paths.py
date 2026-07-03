"""Shared filesystem paths for SPORK.

All roots default to locations relative to the repository and can be
overridden via environment variables so the release is portable:

  SPORK_REPO_ROOT   repository root (default: parent of this package)
  SPORK_MODEL_ROOT  directory holding model weights (default: $SPORK_REPO_ROOT/models)
  SPORK_TAU2_ROOT   tau2-bench checkout (default: $SPORK_REPO_ROOT/tau2-bench)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get("SPORK_REPO_ROOT", Path(__file__).resolve().parent.parent)
)
CODE_ROOT = REPO_ROOT.parent
RESULTS_ROOT = REPO_ROOT / "results"
LOG_ROOT = REPO_ROOT / "memory" / "logs"
MODEL_ROOT = Path(os.environ.get("SPORK_MODEL_ROOT", REPO_ROOT / "models"))
QWEN3_32B_MODEL = MODEL_ROOT / "Qwen3-32B"
TAU2_ROOT = Path(os.environ.get("SPORK_TAU2_ROOT", REPO_ROOT / "tau2-bench"))
TAU2_SRC = TAU2_ROOT / "src"


def add_runtime_paths() -> None:
    """Make coder scripts and tau2 source imports work."""
    for path in (REPO_ROOT / "coder" / "scripts", TAU2_SRC):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
