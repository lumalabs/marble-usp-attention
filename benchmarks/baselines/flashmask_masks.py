# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Convert our packed layouts to FlashMask's column-wise mask, exactly.

For every key, the query rows that may see it form one contiguous range
[lo, hi) in the token order used here, so the mask is FlashMask's two-column
form [LTS, UTE] = [hi, lo]: masked rows [hi, N) and [0, lo).

- causal text key k of document [s, e): visible rows [k, min(e, k + W))
- key of a full span [a, b): [a, max(b, min(e, k + W)))
- key of a noise (generation) span [a, b): [a, b)
- vision groups ("group" splits): no contiguous range in natural order, so each
  document that has groups is reordered: its non-group splits first, in order,
  then its group splits in reverse order. Then a text key in run i sees the
  later text and the groups after it, and a group key in group j sees all
  groups and the text after group j; both are contiguous.

Returns (perm, lo, hi): perm[i] = original index of reordered token i
(None when no document needs reordering); lo/hi are in reordered coordinates.
"""

from __future__ import annotations

import numpy as np


def _docs(document_lens, split_lens, attn_modes):
    it = iter(zip(split_lens, attn_modes))
    for d in document_lens:
        used, out = 0, []
        while used < d:
            n, m = next(it)
            out.append((n, m))
            used += n
        assert used == d, "split straddles a document"
        yield out


def flashmask_ranges(document_lens, split_lens, attn_modes, window=None):
    n = sum(document_lens)
    lo = np.empty(n, np.int64)
    hi = np.empty(n, np.int64)
    perm = np.arange(n, dtype=np.int64)
    reordered = False
    s = 0
    for doc in _docs(document_lens, split_lens, attn_modes):
        e = s + sum(length for length, _ in doc)
        if any(m == "group" for _, m in doc):
            assert window is None, (
                "vision groups with a sliding window are not supported"
            )
            reordered = True
            # Original spans of each split.
            spans, pos = [], s
            for length, mode in doc:
                spans.append((pos, pos + length, mode))
                pos += length
            text = [(a, b) for a, b, m in spans if m != "group"]
            groups = [(a, b) for a, b, m in spans if m == "group"]
            assert all(m in ("causal", "group") for _, _, m in spans), (
                "groups mix only with causal text"
            )
            order = text + groups[::-1]
            new = s
            new_start = {}
            for a, b in order:
                perm[new : new + (b - a)] = np.arange(a, b)
                new_start[(a, b)] = new
                new += b - a
            text_len = sum(b - a for a, b in text)
            # Groups after each original position, as a reordered-prefix length.
            for a, b, mode in spans:
                ns = new_start[(a, b)]
                if mode == "group":
                    # Sees all groups and the text after this group.
                    text_before = sum(tb - ta for ta, tb in text if tb <= a)
                    lo[ns : ns + (b - a)] = s + text_before
                    hi[ns : ns + (b - a)] = e
                else:
                    later_groups = sum(gb - ga for ga, gb in groups if ga >= b)
                    lo[ns : ns + (b - a)] = np.arange(ns, ns + (b - a))
                    hi[ns : ns + (b - a)] = s + text_len + later_groups
        else:
            pos = s
            for length, mode in doc:
                a, b = pos, pos + length
                k = np.arange(a, b)
                reach = np.minimum(e, k + window) if window else np.full(length, e)
                if mode == "causal":
                    lo[a:b], hi[a:b] = k, reach
                elif mode == "full":
                    lo[a:b], hi[a:b] = a, np.maximum(b, reach)
                elif mode == "noise":
                    lo[a:b], hi[a:b] = a, b
                else:
                    raise ValueError(mode)
                pos = b
        s = e
    return (perm if reordered else None), lo, hi


def dense_visible(document_lens, split_lens, attn_modes, window=None):
    """Dense [q, k] visibility in natural order (small sizes; for checks)."""
    n = sum(document_lens)
    seg, doc, noise = [], [], []
    sid = 0
    for di, d in enumerate(_docs(document_lens, split_lens, attn_modes)):
        gid = None
        for length, mode in d:
            if mode == "group":
                gid = sid if gid is None else gid
                sg = gid
            else:
                sg = sid if mode in ("full", "noise") else -1
            seg += [sg] * length
            doc += [di] * length
            noise += [sid if mode == "noise" else -1] * length
            sid += 1
    seg, doc, noise = map(np.array, (seg, doc, noise))
    q = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    causal = q >= k
    if window:
        causal &= (q - k) < window
    same = (seg[:, None] == seg[None, :]) & (seg[:, None] >= 0)
    keep = ~((noise[None, :] >= 0) & (noise[:, None] != noise[None, :]))
    return (causal | same) & keep & (doc[:, None] == doc[None, :])


def check(document_lens, split_lens, attn_modes, window=None):
    perm, lo, hi = flashmask_ranges(document_lens, split_lens, attn_modes, window)
    vis = dense_visible(document_lens, split_lens, attn_modes, window)
    if perm is not None:
        vis = vis[np.ix_(perm, perm)]
    n = vis.shape[0]
    rows = np.arange(n)[:, None]
    want = (rows >= lo[None, :]) & (rows < hi[None, :])
    return bool((want == vis).all())
