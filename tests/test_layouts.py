# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Structure of the synthetic vision-language layouts."""

from __future__ import annotations

import pytest

from usp_attention.layouts import build_layout


def _documents(docs, splits, modes):
    """Yield each document's (split_lens, modes); splits never straddle documents."""
    i = 0
    for doc in docs:
        start, total = i, 0
        while total < doc:
            total += splits[i]
            i += 1
        assert total == doc
        yield splits[start:i], modes[start:i]
    assert i == len(splits)


@pytest.mark.parametrize("name", ["vlm", "video"])
@pytest.mark.parametrize("seq_len", [4096, 32768, 262144, 1048576])
def test_vlm_layouts(name: str, seq_len: int) -> None:
    docs, splits, modes = build_layout(name, seq_len)
    assert sum(docs) == sum(splits) == seq_len
    assert set(modes) <= {"causal", "full"}
    for doc_splits, doc_modes in _documents(docs, splits, modes):
        # Text opens and closes every document, and text and images alternate.
        assert doc_modes[0] == doc_modes[-1] == "causal"
        assert all(a != b for a, b in zip(doc_modes, doc_modes[1:]))
        for length, mode in zip(doc_splits, doc_modes):
            assert length > 0
            if mode == "full":
                assert length % 256 == 0
    image = sum(s for s, m in zip(splits, modes) if m == "full") / seq_len
    # Short videos hold proportionally more prompt and answer text.
    assert image > (0.9 if name == "video" and seq_len >= 32768 else 0.5)


def test_vlm_layout_is_seeded() -> None:
    assert build_layout("vlm", 262144, seed=1) == build_layout("vlm", 262144, seed=1)
    assert build_layout("vlm", 262144, seed=1) != build_layout("vlm", 262144, seed=2)


def test_vision_group_mask() -> None:
    """Every image token sees all image tokens of its document; text stays causal."""
    import torch

    from usp_attention.fa4.interval_mask import build_interval_plan
    from usp_attention.reference import dense_mask_rows, visible_key_counts
    from usp_attention.ring.dense_ring import dense_ring_supported
    from usp_attention.ring.packed_layout import build_packed_layout

    docs, splits, modes = build_layout("vlm_group", 16384)
    assert "group" in modes and "full" not in modes
    n = sum(splits)
    doc = torch.repeat_interleave(torch.arange(len(docs)), torch.tensor(docs))
    image = torch.repeat_interleave(
        torch.tensor([m == "group" for m in modes]), torch.tensor(splits)
    )
    q, k = torch.arange(n)[:, None], torch.arange(n)[None, :]
    expected = (doc[:, None] == doc[None, :]) & (
        (k <= q) | (image[:, None] & image[None, :])
    )
    layout = build_packed_layout(docs, splits, modes)
    assert torch.equal(dense_mask_rows(layout, 0, n), expected)
    assert torch.equal(visible_key_counts(docs, splits, modes), expected.sum(1))
    # A group token's keys are not one contiguous range: no dense paths.
    assert build_interval_plan(docs, splits, modes, device="cpu") is None
    assert not dense_ring_supported(docs, splits, modes)
