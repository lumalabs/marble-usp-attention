# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""usp.py passes only keywords its attention entry points accept.

The dense-gather and halo paths need multi-GPU runs, so a keyword added at one call
site and not the other only failed at 128 GPUs. This checks the call sites statically.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "usp_attention"
ENTRY_POINTS = {
    "dense_gather_attention": SRC / "ring" / "dense_gather.py",
    "halo_attention": SRC / "ring" / "halo.py",
}


def _params(path: Path, name: str) -> set[str]:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            a = node.args
            return {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    raise AssertionError(f"{name} not found in {path}")


def test_usp_call_sites_use_known_keywords() -> None:
    tree = ast.parse((SRC / "usp.py").read_text())
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in ENTRY_POINTS:
                seen.add(name)
                allowed = _params(ENTRY_POINTS[name], name)
                passed = {k.arg for k in node.keywords if k.arg is not None}
                assert passed <= allowed, f"{name}() got unknown {passed - allowed}"
    assert seen == set(ENTRY_POINTS)
