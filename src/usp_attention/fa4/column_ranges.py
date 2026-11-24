# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Each key's visible queries as one contiguous range, in vision-group order.

In :func:`~usp_attention.ring.group_order.vision_group_order` (a document's text
splits first, in order, then its group splits in reverse order), the queries
that may see any key form one contiguous range ``[lo, hi)`` of the reordered
sequence:

- a causal text key ``k`` of document ``[s, e)``: ``[k, text end + later groups)``;
- a key of group ``j``: ``[s + text before group j, e)`` (all groups, the text after it);
- in a document without groups: a causal key ``k`` sees ``[k, e)``, a key of a
  full span ``[a, b)`` sees ``[a, e)`` and a key of a noise span sees ``[a, b)``.

A document without groups keeps the natural order, so there the ranges hold in
any token order as long as queries are compared by position: that is the form
ring steps use. A sliding window ``W`` (keys of the causal test limited to the
last ``W`` positions) shortens a causal key's range to ``[k, min(e, k + W))``
and a full span key's to ``[a, max(b, min(e, k + W)))``.

So the mask is ``lo[k] <= q < hi[k]``: two int32 loads per element instead of
the eight lookups of the position-aware predicate, which the block-sparse
kernel pays on every partially masked tile. FlashMask stores masks the same
way (its column-wise ``[LTS, UTE]`` = ``[hi, lo]``).
"""

from __future__ import annotations

import numpy as np

from usp_attention.ring.packed_layout import AttentionMode


def _documents(document_lens, split_lens, attn_modes):
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


def column_query_ranges(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    sliding_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-key visible query range ``[lo, hi)`` in vision-group order (int32).

    Raises:
        ValueError: If a document mixes groups with full or noise spans, or a
            sliding window meets a group, where one range per key does not hold.
    """
    n = sum(document_lens)
    lens = np.asarray(split_lens, dtype=np.int64)
    modes = np.asarray(attn_modes)
    if sum(split_lens) != n:
        raise ValueError("a split straddles a document boundary")
    a = np.cumsum(lens) - lens  # split starts (original order)
    b = a + lens
    doc_end = np.cumsum(np.asarray(document_lens, dtype=np.int64))
    doc_start = doc_end - np.asarray(document_lens, dtype=np.int64)
    doc = np.searchsorted(doc_end, a, side="right")
    if np.any(b > doc_end[doc]):
        raise ValueError("a split straddles a document boundary")
    s, e = doc_start[doc], doc_end[doc]
    group = modes == "group"
    grouped_doc = np.zeros(len(document_lens), bool)
    grouped_doc[doc[group]] = True
    in_grouped = grouped_doc[doc]
    if in_grouped.any():
        if sliding_window is not None:
            raise ValueError("sliding windows are not combined with groups here")
        if np.any(in_grouped & ~np.isin(modes, ("causal", "group"))):
            raise ValueError("vision groups mix only with causal text here")
    unknown = ~np.isin(modes, ("causal", "full", "noise", "group"))
    if np.any(unknown & ~in_grouped):
        raise ValueError(f"unknown attention mode {modes[unknown][0]!r}")

    # Per-document running sums over splits: text and group tokens before /
    # through each split, and each document's totals.
    def doc_cumsum(x: np.ndarray) -> np.ndarray:
        c = np.cumsum(x)
        first = np.searchsorted(doc, doc, side="left")
        return c - (c[first] - x[first])

    text_len = np.where(group, 0, lens)
    group_len = np.where(group, lens, 0)
    text_through = doc_cumsum(text_len)
    group_through = doc_cumsum(group_len)
    last = np.searchsorted(doc, doc, side="right") - 1
    text_total, group_total = text_through[last], group_through[last]
    # Group order: text splits first in order, group splits after in reverse.
    new_start = np.where(
        in_grouped,
        np.where(
            group,
            s + text_total + group_total - group_through,
            s + text_through - text_len,
        ),
        a,
    )
    causal = modes == "causal"
    full = modes == "full"
    # A key's lowest visible query is itself for causal text, else a constant.
    ramp = np.where(in_grouped, ~group, causal)
    lo_split = np.where(in_grouped & group, s + text_through - text_len, a)
    hi_split = np.select(
        [in_grouped & group, in_grouped, causal | full],
        [e, s + text_total + group_total - group_through, e],
        b,
    )
    within = np.arange(n, dtype=np.int64) - np.repeat(a, lens)
    key = np.repeat(new_start, lens) + within  # stored index of every token
    lo_tok = np.repeat(lo_split, lens)
    ramp_tok = np.repeat(ramp, lens)
    lo_tok[ramp_tok] = key[ramp_tok]
    hi_tok = np.repeat(hi_split, lens)
    if sliding_window is not None:
        # No groups here: causal keys reach the window, full keys at least their span.
        windowed = np.repeat(causal | full, lens)
        reach = np.minimum(hi_tok, np.arange(n, dtype=np.int64) + sliding_window)
        hi_tok = np.where(
            windowed,
            np.where(
                np.repeat(full, lens), np.maximum(np.repeat(b, lens), reach), reach
            ),
            hi_tok,
        )
    lo = np.empty(n, np.int64)
    hi = np.empty(n, np.int64)
    lo[key], hi[key] = lo_tok, hi_tok
    return lo.astype(np.int32), hi.astype(np.int32)
