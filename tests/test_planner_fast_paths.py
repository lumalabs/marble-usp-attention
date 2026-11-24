# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Vectorized planner helpers equal their original per-span loops."""

from __future__ import annotations

import glob
import random

import numpy as np
import pytest
import torch
from torch import Tensor

from usp_attention.fa4.column_ranges import column_query_ranges
from usp_attention.fa4.flash4_attn import AttentionMode, _build_common_tensors
from usp_attention.layouts import load_layout_file
from usp_attention.reference import visible_key_counts
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.halo import needed_key_masks
from usp_attention.ring.packed_layout import segment_ids


def _counts_loop(document_lens, split_lens, attn_modes):
    """The original per-split implementation of visible_key_counts."""
    counts = []
    split_iter = iter(zip(split_lens, attn_modes, strict=True))
    for doc_len in document_lens:
        doc_splits, offset = [], 0
        while offset < doc_len:
            length, mode = next(split_iter)
            doc_splits.append((length, mode))
            offset += length
        group_after = sum(n for n, m in doc_splits if m == "group")
        offset = hidden_noise = 0
        for length, mode in doc_splits:
            visible_before = offset - hidden_noise
            if mode == "group":
                group_after -= length
            if mode == "causal":
                counts.append(visible_before + torch.arange(1, length + 1))
            else:
                later = group_after if mode == "group" else 0
                counts.append(torch.full((length,), visible_before + length + later))
            if mode == "noise":
                hidden_noise += length
            offset += length
    return torch.cat(counts)


def _common_loop(document_lens, split_lens, attn_modes):
    """The original list-building implementation of _build_common_tensors."""
    fn, noise = [], []
    ids = segment_ids(document_lens, split_lens, attn_modes)
    for i, (length, mode) in enumerate(zip(split_lens, attn_modes, strict=True)):
        fn.extend([ids[i]] * length)
        noise.extend([i if mode == "noise" else -1] * length)
    docs = torch.cat(
        [torch.full((n,), i, dtype=torch.int64) for i, n in enumerate(document_lens, 1)]
    )
    return (
        torch.tensor(fn, dtype=torch.float32),
        torch.tensor(noise, dtype=torch.float32),
        docs,
    )


def _needed_loop(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    ring_size: int,
    token_permutation: torch.Tensor | None,
    sliding_window: int | None,
) -> list[np.ndarray]:
    """The original per-split implementation of needed_key_masks."""
    seq_len = sum(document_lens)
    local = seq_len // ring_size
    window = sliding_window or seq_len
    doc0 = np.empty(seq_len, np.int64)
    lo = np.empty(seq_len, np.int64)
    hi = np.empty(seq_len, np.int64)
    noise = np.zeros(seq_len, bool)
    group_of_doc = np.zeros(seq_len, bool)  # query is a group token
    group_spans: dict[int, list[tuple[int, int]]] = {}
    noise_span_lo = np.full(seq_len, -1, np.int64)
    noise_span_hi = np.full(seq_len, -1, np.int64)
    pos = 0
    split_iter = iter(zip(split_lens, attn_modes, strict=True))
    for doc_len in document_lens:
        d0 = pos
        while pos < d0 + doc_len:
            n, mode = next(split_iter)
            q = np.arange(pos, pos + n)
            doc0[pos : pos + n] = d0
            start = np.maximum(d0, q - window + 1)
            if mode == "causal":
                lo[pos : pos + n], hi[pos : pos + n] = start, q + 1
            elif mode == "group":
                lo[pos : pos + n], hi[pos : pos + n] = start, q + 1
                group_of_doc[pos : pos + n] = True
                group_spans.setdefault(d0, []).append((pos, pos + n))
            else:  # full / noise: windowed prefix plus the whole span
                lo[pos : pos + n] = np.minimum(start, pos)
                hi[pos : pos + n] = pos + n
                if mode == "noise":
                    noise[pos : pos + n] = True
                    noise_span_lo[pos : pos + n] = pos
                    noise_span_hi[pos : pos + n] = pos + n
            pos += n
    perm = (
        np.arange(seq_len)
        if token_permutation is None
        else token_permutation.detach().cpu().numpy()
    )
    masks = []
    for r in range(ring_size):
        qs = perm[r * local : (r + 1) * local]
        diff = np.zeros(seq_len + 1, np.int64)
        np.add.at(diff, lo[qs], 1)
        np.add.at(diff, hi[qs], -1)
        need = (np.cumsum(diff[:-1]) > 0) & ~noise
        own = qs[noise_span_lo[qs] >= 0]
        if own.size:
            spans = np.unique(
                np.stack([noise_span_lo[own], noise_span_hi[own]], 1), axis=0
            )
            for a, b in spans:
                need[a:b] = True
        gq = qs[group_of_doc[qs]]
        for d0 in np.unique(doc0[gq]):
            for a, b in group_spans[int(d0)]:
                need[a:b] = True
        masks.append(need[perm])
    return masks


def _group_order_loop(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
) -> Tensor | None:
    """The original per-split group_order_loop."""
    if "group" not in attn_modes:
        return None
    pieces: list[Tensor] = []
    split_iter = iter(zip(split_lens, attn_modes, strict=True))
    offset = 0
    for doc_len in document_lens:
        text: list[Tensor] = []
        groups: list[Tensor] = []
        pos = offset
        while pos < offset + doc_len:
            length, mode = next(split_iter)
            span = torch.arange(pos, pos + length)
            (groups if mode == "group" else text).append(span)
            pos += length
        pieces += text + groups[::-1]
        offset += doc_len
    return torch.cat(pieces)


def _documents_loop(document_lens, split_lens, attn_modes):
    splits = iter(zip(split_lens, attn_modes))
    for d in document_lens:
        used, out = 0, []
        while used < d:
            n, m = next(splits)
            out.append((n, m))
            used += n
        if used != d:
            raise ValueError("a split straddles a document boundary")
        yield out


def _column_ranges_loop(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    sliding_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The original per-split column_ranges_loop."""
    n = sum(document_lens)
    lo = np.empty(n, np.int64)
    hi = np.empty(n, np.int64)
    s = 0
    for doc in _documents_loop(document_lens, split_lens, attn_modes):
        e = s + sum(length for length, _ in doc)
        spans, pos = [], s
        for length, mode in doc:
            spans.append((pos, pos + length, mode))
            pos += length
        if any(m == "group" for _, _, m in spans):
            if sliding_window is not None:
                raise ValueError("sliding windows are not combined with groups here")
            if any(m not in ("causal", "group") for _, _, m in spans):
                raise ValueError("vision groups mix only with causal text here")
            text = [(a, b) for a, b, m in spans if m != "group"]
            group = [(a, b) for a, b, m in spans if m == "group"]
            text_len = sum(b - a for a, b in text)
            starts, new = {}, s
            for a, b in text + group[::-1]:
                starts[(a, b)] = new
                new += b - a
            text_ends = np.cumsum([b - a for a, b in text])
            text_orig_ends = np.array([b for _, b in text])
            group_lens = np.array([b - a for a, b in group])
            group_starts = np.array([a for a, _ in group])
            for a, b, mode in spans:
                ns = starts[(a, b)]
                if mode == "group":
                    before = (
                        int(text_ends[text_orig_ends <= a][-1])
                        if (text_orig_ends <= a).any()
                        else 0
                    )
                    lo[ns : ns + (b - a)] = s + before
                    hi[ns : ns + (b - a)] = e
                else:
                    later = int(group_lens[group_starts >= b].sum())
                    lo[ns : ns + (b - a)] = np.arange(ns, ns + (b - a))
                    hi[ns : ns + (b - a)] = s + text_len + later
        else:
            for a, b, mode in spans:
                keys = np.arange(a, b)
                reach = (
                    e
                    if sliding_window is None
                    else np.minimum(e, keys + sliding_window)
                )
                if mode == "causal":
                    lo[a:b], hi[a:b] = keys, reach
                elif mode == "full":
                    lo[a:b], hi[a:b] = a, np.maximum(b, reach)
                elif mode == "noise":
                    lo[a:b], hi[a:b] = a, b
                else:
                    raise ValueError(f"unknown attention mode {mode!r}")
        s = e
    return lo.astype(np.int32), hi.astype(np.int32)


def _random_layout(seed: int):
    rng = random.Random(seed)
    docs, splits, modes = [], [], []
    for _ in range(rng.randint(1, 12)):
        doc = 0
        for _ in range(rng.randint(1, 6)):
            n = rng.choice([1, 7, 128, 300])
            splits.append(n)
            modes.append(rng.choice(["causal", "full", "noise", "group"]))
            doc += n
        docs.append(doc)
    return docs, splits, modes


def _layouts():
    for seed in range(20):
        yield _random_layout(seed)
    for path in sorted(glob.glob("benchmarks/layouts/packs/*_1024k.json.gz"))[:6]:
        yield load_layout_file(path.removesuffix(".gz") + "#0", 1 << 20)


@pytest.mark.parametrize("layout", list(_layouts()))
def test_vectorized_helpers_match_the_loops(layout):
    docs, splits, modes = layout
    assert torch.equal(visible_key_counts(docs, splits, modes), _counts_loop(*layout))
    for got, want in zip(
        _build_common_tensors(docs, splits, modes, device="cpu"),
        _common_loop(*layout),
        strict=True,
    ):
        assert got.dtype == want.dtype and torch.equal(got, want)


def test_cached_counts_are_independent_copies():
    docs, splits, modes = _random_layout(3)
    a = visible_key_counts(docs, splits, modes)
    a.zero_()
    assert torch.equal(
        visible_key_counts(docs, splits, modes), _counts_loop(docs, splits, modes)
    )


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("window", [None, 200])
def test_needed_key_masks_match_the_loop(seed, window):
    docs, splits, modes = _random_layout(seed)
    seq = sum(docs)
    pad = (-seq) % 4
    docs, splits, modes = (
        [*docs, pad] if pad else docs,
        [*splits, pad] if pad else splits,
        [*modes, "causal"] if pad else modes,
    )
    seq += pad
    for ring in (1, 2, 4):
        perm = torch.randperm(seq, generator=torch.Generator().manual_seed(seed))
        for order in (None, perm):
            kw = dict(ring_size=ring, token_permutation=order, sliding_window=window)
            got = needed_key_masks(docs, splits, modes, **kw)
            want = _needed_loop(docs, splits, modes, **kw)
            assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))


def _group_layout(seed: int):
    """Documents of causal text and vision groups (as Molmo2), or of the other modes."""
    rng = random.Random(seed)
    docs, splits, modes = [], [], []
    for _ in range(rng.randint(1, 10)):
        kinds = (
            ["causal", "group"] if rng.random() < 0.6 else ["causal", "full", "noise"]
        )
        doc = 0
        for _ in range(rng.randint(1, 8)):
            n = rng.choice([1, 5, 64, 129])
            splits.append(n)
            modes.append(rng.choice(kinds))
            doc += n
        docs.append(doc)
    return docs, splits, modes


def _group_layouts():
    for seed in range(30):
        yield _group_layout(seed)
    for path in sorted(glob.glob("benchmarks/layouts/packs/molmo2_*_1024k.json.gz"))[
        :2
    ]:
        yield load_layout_file(path.removesuffix(".gz") + "#0", 1 << 20)


@pytest.mark.parametrize("layout", list(_group_layouts()))
def test_group_order_and_column_ranges_match_the_loops(layout):
    docs, splits, modes = layout
    got, want = vision_group_order(*layout), _group_order_loop(*layout)
    assert (got is None) == (want is None)
    if want is not None:
        assert torch.equal(got, want)
    for window in (None, 3, 100):
        try:
            want_r = _column_ranges_loop(docs, splits, modes, sliding_window=window)
        except ValueError:
            with pytest.raises(ValueError):
                column_query_ranges(docs, splits, modes, sliding_window=window)
            continue
        got_r = column_query_ranges(docs, splits, modes, sliding_window=window)
        assert all(
            np.array_equal(g, w) and g.dtype == w.dtype
            for g, w in zip(got_r, want_r, strict=True)
        )
