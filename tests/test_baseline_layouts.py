# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The baseline scripts' layout copy matches usp_attention.layouts exactly."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from usp_attention.layouts import LAYOUT_NAMES, build_layout
from usp_attention.reference import visible_key_counts

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "baselines" / "_layouts.py"
_spec = importlib.util.spec_from_file_location("baseline_layouts", _PATH)
baseline_layouts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_layouts)


@pytest.mark.parametrize("name", LAYOUT_NAMES)
@pytest.mark.parametrize("seq_len", [32768, 524288])
def test_layout_copy_matches(name: str, seq_len: int) -> None:
    ours = build_layout(name, seq_len, ring_size=8)
    theirs = baseline_layouts.build_layout(name, seq_len, ring_size=8)
    assert ours == theirs
    torch.testing.assert_close(
        visible_key_counts(*ours), baseline_layouts.visible_key_counts(*theirs)
    )


_MAGI_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "baselines" / "magi_attn.py"
)
_magi_spec = importlib.util.spec_from_file_location("magi_baseline", _MAGI_PATH)
magi_baseline = importlib.util.module_from_spec(_magi_spec)
_magi_spec.loader.exec_module(magi_baseline)


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_magi_slices_equal_dense_mask(name: str) -> None:
    from usp_attention.reference import dense_mask_rows
    from usp_attention.ring.packed_layout import build_packed_layout

    seq_len = 4096
    docs, splits, modes = build_layout(name, seq_len, ring_size=4)
    q_ranges, k_ranges, types = magi_baseline.mask_slices(docs, splits, modes)
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for (qa, qb), (ka, kb), kind in zip(q_ranges, k_ranges, types):
        block = torch.ones(qb - qa, kb - ka, dtype=torch.bool)
        if kind == "causal":
            assert (qa, qb) == (ka, kb)
            block = block.tril()
        mask[qa:qb, ka:kb] |= block
    expected = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, seq_len)
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("name", LAYOUT_NAMES)
def test_dense_mask_matches_reference(name: str) -> None:
    from usp_attention.reference import dense_mask_rows
    from usp_attention.ring.packed_layout import build_packed_layout

    seq_len = 4096
    docs, splits, modes = build_layout(name, seq_len, ring_size=4)
    mask = baseline_layouts.dense_mask(docs, splits, modes)
    expected = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, seq_len)
    assert torch.equal(mask, expected)
    assert torch.equal(mask.sum(1), visible_key_counts(docs, splits, modes))
