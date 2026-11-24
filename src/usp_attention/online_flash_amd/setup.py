# SPDX-FileCopyrightText: 2026 The MoonMath.ai
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Build the online_flash_4_amd ``_C`` extension via torch CUDAExtension (ROCm / CDNA3 only).

Run in place (writes ``_C.<abi>.so`` next to this file):

    python setup.py build_ext --inplace

Requires a ROCm build of PyTorch and ``hipcc`` on PATH. Adapted from MoonMath's amd-kernels
build (https://github.com/moonmath-ai/amd-kernels, MIT).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from setuptools import setup
from setuptools.command.build import build as _build
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if torch.version.hip is None:
    raise RuntimeError(
        "online_flash_4_amd requires a ROCm build of PyTorch (torch.version.hip is None). "
        "Install ROCm PyTorch and ensure hipcc is available on PATH."
    )

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"
DIST = ROOT / "dist"

# Default gfx942 (MI300X/MI325X); override with CDNA3_ARCH.
ARCH = os.environ.get("CDNA3_ARCH", "gfx942")

# Stage csrc/ into dist/ and build from there so torch's hipify intermediates
# (`<stem>_hip.cpp`) stay out of csrc/.
DIST.mkdir(parents=True, exist_ok=True)
_compile_names = [
    "attention_api.cpp",
    "attention_rtna.hip",
    "attention_rtne.hip",
    "attention_rtz.hip",
]
_include_names = ["attention_kernel.hip", "opus.hpp", "sparse_args.h"]

# PyTorch's HIP ninja rule does not emit header dependency files. Include a
# content digest in each staged translation unit so changing a kernel/header
# reliably rebuilds every rounding-mode specialization (and the binding).
header_digest = hashlib.sha256(
    b"".join((CSRC / name).read_bytes() for name in _include_names)
).hexdigest()
sources = []
for name in _compile_names + _include_names:
    src, dst = CSRC / name, DIST / name
    content = src.read_bytes()
    if name in _compile_names:
        content += f"\n// Header dependency digest: {header_digest}\n".encode()
        sources.append(str(dst.relative_to(ROOT)))
    if not dst.exists() or dst.read_bytes() != content:
        dst.write_bytes(content)

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [  # BuildExtension routes "nvcc" to hipcc on ROCm
        "-O3",
        "-std=c++17",
        f"--offload-arch={ARCH}",
        "-ffast-math",
        "-fno-math-errno",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
    ],
}


class DistBuild(_build):
    """Route every build artifact under dist/ (object files, ninja temp, wheel staging)."""

    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(DIST)


setup(
    name="online_flash_4_amd",
    ext_modules=[
        CUDAExtension(name="_C", sources=sources, extra_compile_args=extra_compile_args)
    ],
    cmdclass={"build": DistBuild, "build_ext": BuildExtension},
)
