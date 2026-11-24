# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Loader for the compiled ``_C`` HIP extension.

Build it (once, in place) with a ROCm PyTorch + hipcc:

    cd src/usp_attention/online_flash_amd
    python setup.py build_ext --inplace
"""

from __future__ import annotations

try:
    from . import _C  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - build-environment dependent
    raise ImportError(
        "online_flash_4_amd: failed to import the compiled `_C` extension. Build it with a ROCm "
        "PyTorch + hipcc:\n"
        "    cd src/usp_attention/online_flash_amd && python setup.py build_ext --inplace"
    ) from exc

__all__ = ["_C"]
