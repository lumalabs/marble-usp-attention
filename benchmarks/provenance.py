# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Environment stamps for benchmark rows; importable in isolated baseline venvs.

Unknown values stay null with an explanation. Never stamp archived measurements
with the environment of the machine that happens to be reading them.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch",
    "flash-attn-4",
    "flash-attn-3",
    "flash-attn-cute",
    "nvidia-cutlass",
    "nvidia-cutlass-dsl",
    "nvidia-nccl-cu12",
    "nvidia-nccl-cu13",
    "flashmask",
    "magi-attention",
    "transformer-engine",
    "yunchang",
    "quack-kernels",
)


def _command(args: list[str], *, cwd: Path | None = None) -> str | None:
    """Read a bounded diagnostic, returning null when the tool is unavailable."""
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(path: Path) -> dict[str, Any]:
    """Record source revision and tracked/untracked dirt without author metadata.

    Args:
        path: Directory in the source checkout to inspect.

    Returns:
        A dictionary containing the revision SHA, dirty flag, and
        unavailability reason. Inaccessible Git fields are None; the checkout
        is not modified.
    """
    sha = _command(["git", "rev-parse", "HEAD"], cwd=path)
    status = _command(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=path
    )
    return {
        "sha": sha,
        "dirty": bool(status) if status is not None else None,
        "unavailable_reason": None if sha else "source has no accessible Git revision",
    }


def source_revision() -> dict[str, Any]:
    """Resolve the submitted or locally available source revision.

    Returns:
        Revision metadata supplied by the launcher or the local Git checkout,
        in that order. Unavailable fields remain None with a reason.
    """
    launched = os.environ.get("MARBLE_GIT_COMMIT", "")
    if re.fullmatch(r"[0-9a-f]{40}(?:-dirty)?", launched):
        return {
            "sha": launched.removesuffix("-dirty"),
            "dirty": launched.endswith("-dirty"),
            "source": "submitted_tree",
            "unavailable_reason": None,
        }
    return git_state(PROJECT)


@lru_cache(maxsize=1)
def environment() -> dict[str, Any]:
    """Snapshot the emitting process's environment, once per process.

    The torch NCCL version is the loaded runtime's version; distribution
    versions are also retained to distinguish bundled NCCL from an LD_PRELOAD
    override. GPU inventory covers visible CUDA devices. Driver versions come
    from NVML's command-line client, not the CUDA version advertised by
    nvidia-smi.

    Returns:
        A cached dictionary of source, package, CUDA, NCCL, driver, GPU and
        process metadata. Unavailable components are marked explicitly. Callers
        should treat the shared dictionary as read-only.
    """
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    stamp: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": source_revision(),
        "python": sys.version.split()[0],
        "packages": packages,
        "torch": None,
        "cuda": None,
        "nccl": None,
        "gpus": [],
        "driver": _command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "nvcc": _command(["nvcc", "--version"]),
        "command": sys.argv,
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("USP_", "MAGI_ATTENTION_", "NVTE_", "NCCL_"))
            or key
            in ("LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "FLASHMASK_DIR", "MAGI_DIR")
        },
        "baseline_sources": {},
        "unavailable": {},
    }
    for variable in ("FLASHMASK_DIR", "MAGI_DIR"):
        if os.environ.get(variable):
            stamp["baseline_sources"][variable] = git_state(Path(os.environ[variable]))
    try:
        import torch  # optional: baseline/report environments can lack torch
    except ImportError:
        stamp["unavailable"]["torch"] = "torch is not importable"
    else:
        stamp["torch"] = str(torch.__version__)
        stamp["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                stamp["gpus"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "compute_capability": [props.major, props.minor],
                        "total_memory_bytes": props.total_memory,
                    }
                )
            try:
                stamp["nccl"] = list(torch.cuda.nccl.version())
            except (AttributeError, RuntimeError):
                stamp["unavailable"]["nccl"] = "NCCL runtime version is unavailable"
        else:
            stamp["unavailable"]["gpus"] = "CUDA is not available"
    for field in ("cuda", "nccl", "driver", "nvcc"):
        if stamp[field] is None:
            stamp["unavailable"].setdefault(field, "not available in this process")
    return stamp


def stamp_row(row: dict[str, Any]) -> dict[str, Any]:
    """Stamp a newly measured row, preserving stamps from worker processes.

    Args:
        row: New measurement or correctness row; not modified in place.

    Returns:
        The original row if it already has provenance; otherwise a new
        dictionary containing the row and the current process environment.
    """
    if "provenance" in row:
        return row
    return {"provenance": environment(), **row}


def result_json(row: dict[str, Any]) -> str:
    """Serialize a new measurement or correctness row with its environment.

    Args:
        row: New measurement or correctness row to stamp and serialize.

    Returns:
        A JSON string retaining existing worker provenance when present.

    Raises:
        TypeError: The row contains values that cannot be serialized as JSON.
    """
    return json.dumps(stamp_row(row))
