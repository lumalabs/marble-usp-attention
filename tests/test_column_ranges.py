# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Column ranges in vision-group order reproduce the packed mask exactly."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from usp_attention.fa4.column_ranges import column_query_ranges
from usp_attention.layouts import build_layout
from usp_attention.reference import dense_mask_rows
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.packed_layout import build_packed_layout


def _range_mask_matches(docs, splits, modes) -> bool:
    n = sum(docs)
    mask = dense_mask_rows(build_packed_layout(docs, splits, modes), 0, n)
    perm = vision_group_order(docs, splits, modes)
    if perm is not None:
        mask = mask[perm][:, perm]
    lo, hi = column_query_ranges(docs, splits, modes)
    rows = np.arange(n)[:, None]
    want = (rows >= lo[None, :]) & (rows < hi[None, :])
    return bool((torch.from_numpy(want) == mask).all())


@pytest.mark.parametrize(
    "name", ["docs", "mixed", "vlm", "video", "vlm_group", "video_group"]
)
def test_named_layouts(name: str) -> None:
    docs, splits, modes = build_layout(name, 8192, seed=0)[:3]
    assert _range_mask_matches(docs, splits, modes)


@pytest.mark.parametrize("seed", range(20))
def test_random_group_and_generation_documents(seed: int) -> None:
    rng = random.Random(seed)
    docs, splits, modes = [], [], []
    while sum(docs) < 3000:
        kind = rng.choice(["group", "generation", "text"])
        doc_s, doc_m = [rng.randint(1, 40)], ["causal"]
        for _ in range(rng.randint(0, 6)):
            if kind == "group":
                doc_s += [rng.randint(1, 60), rng.randint(1, 20)]
                doc_m += ["group", "causal"]
            elif kind == "generation":
                frame = rng.randint(1, 60)
                doc_s += [frame, frame]
                doc_m += ["noise", "full"]
            else:
                doc_s += [rng.randint(1, 60)]
                doc_m += ["causal"]
        docs.append(sum(doc_s))
        splits += doc_s
        modes += doc_m
    assert _range_mask_matches(docs, splits, modes)


def test_groups_with_generation_spans_are_rejected() -> None:
    with pytest.raises(ValueError):
        column_query_ranges([30], [10, 10, 10], ["causal", "group", "noise"])


@pytest.mark.parametrize("name", ["causal", "docs", "mixed", "vlm", "video"])
@pytest.mark.parametrize("window", [1, 100, 1024, 5000])
def test_sliding_window(name: str, window: int) -> None:
    docs, splits, modes = build_layout(name, 8192, seed=3)[:3]
    n = sum(docs)
    mask = dense_mask_rows(
        build_packed_layout(docs, splits, modes), 0, n, sliding_window=window
    )
    lo, hi = column_query_ranges(docs, splits, modes, sliding_window=window)
    rows = np.arange(n)[:, None]
    want = (rows >= lo[None, :]) & (rows < hi[None, :])
    assert bool((torch.from_numpy(want) == mask).all())


def test_sliding_window_rejects_groups() -> None:
    docs, splits, modes = build_layout("vlm_group", 8192, seed=0)[:3]
    with pytest.raises(ValueError):
        column_query_ranges(docs, splits, modes, sliding_window=128)
