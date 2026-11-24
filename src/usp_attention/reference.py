# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Dense reference attention for packed mixed-mode masks.

Evaluates the same predicate as the FA4 ``mask_mod`` from per-token metadata,
in fp32 or fp64, one query chunk at a time so it fits in memory at long
sequence lengths. Intended for accuracy measurements, not speed.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from usp_attention.fa4.flash4_attn import AttentionMode
from usp_attention.ring.packed_layout import PackedLayout, build_packed_layout


def dense_mask_rows(
    layout: PackedLayout,
    q_start: int,
    q_end: int,
    *,
    sliding_window: int | None = None,
) -> Tensor:
    """Boolean mask rows ``[q_start, q_end)`` x all keys for a packed layout.

    Mirrors the FA4 predicate
    ``(causal | same_full_or_noise_segment) & ~(kv_is_noise & different_noise_segment) & same_document``,
    with ``causal & (q - kv < sliding_window)`` in place of ``causal`` when a
    sliding window is set.
    """
    fn = layout.full_and_noise_seq_id
    noise = layout.noise_seq_id
    doc = layout.document_id
    device = fn.device
    q_pos = torch.arange(q_start, q_end, device=device)
    kv_pos = torch.arange(layout.seq_len, device=device)
    causal = q_pos[:, None] >= kv_pos[None, :]
    if sliding_window:
        causal &= (q_pos[:, None] - kv_pos[None, :]) < sliding_window
    fn_q = fn[q_start:q_end, None]
    full_noise = (fn_q == fn[None, :]) & (fn_q >= 0)
    remove_noise = ~(
        (noise[None, :] >= 0) & (noise[q_start:q_end, None] != noise[None, :])
    )
    same_doc = doc[q_start:q_end, None] == doc[None, :]
    return (causal | full_noise) & remove_noise & same_doc


def _chunk_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor, scale: float
) -> tuple[Tensor, Tensor]:
    # q: (B, Sq, H, D), k/v: (B, S, Hkv, D), mask: (Sq, S)
    heads, kv_heads = q.shape[2], k.shape[2]
    if heads != kv_heads:
        k = k.repeat_interleave(heads // kv_heads, dim=2)
        v = v.repeat_interleave(heads // kv_heads, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse[..., None]).nan_to_num(0.0)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v)
    return out, lse.transpose(1, 2)


def reference_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    softmax_scale: float | None = None,
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 1024,
    sliding_window: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Exact masked attention in ``dtype`` over full (unsharded) BSHD tensors.

    Differentiable with respect to ``q``, ``k`` and ``v``; each query chunk is
    recomputed in backward to bound memory.

    Returns:
        ``(out, lse)`` with ``out`` in ``(B, S, H, D)`` and ``lse`` in
        ``(B, S, H)``, both in ``dtype``. Rows with no visible keys get a zero
        output and ``-inf`` LSE.
    """
    layout = build_packed_layout(document_lens, split_lens, attn_modes, device=q.device)
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    q, k, v = (t.to(dtype) for t in (q, k, v))
    outs, lses = [], []
    for start in range(0, layout.seq_len, chunk_size):
        end = min(start + chunk_size, layout.seq_len)
        mask = dense_mask_rows(layout, start, end, sliding_window=sliding_window)
        out, lse = checkpoint(
            _chunk_attention, q[:, start:end], k, v, mask, scale, use_reentrant=False
        )
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs, dim=1), torch.cat(lses, dim=1)


def visible_key_counts(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    *,
    sliding_window: int | None = None,
) -> Tensor:
    """Exact number of visible keys for every query row, in closed form.

    A causal token sees the earlier tokens of its document (itself included);
    a full or noise token sees its whole split plus the earlier tokens of its
    document. Tokens of an earlier noise split are invisible to everyone
    outside that split. A group token also sees every later group split of
    its document. Splits never straddle documents.

    With a sliding window, the causal keys are limited to the last
    ``sliding_window`` positions of the document, while a token still sees
    every token of its own full, noise or group segment.

    Returns:
        ``int64`` tensor of shape ``(seq_len,)`` on CPU.
    """
    key = (
        tuple(document_lens),
        tuple(split_lens),
        tuple(attn_modes),
        sliding_window or 0,
    )
    cached = _COUNTS_CACHE.get(key)
    if cached is None:
        if sliding_window:
            cached = _windowed_visible_key_counts(
                document_lens, split_lens, attn_modes, sliding_window
            )
        else:
            cached = _visible_key_counts(document_lens, split_lens, attn_modes)
        if len(_COUNTS_CACHE) >= 4:  # a handful of layouts per batch
            _COUNTS_CACHE.pop(next(iter(_COUNTS_CACHE)))
        _COUNTS_CACHE[key] = cached
    return cached.clone()


# Layout -> visible-key counts: one plan asks several times per batch.
_COUNTS_CACHE: dict[tuple, Tensor] = {}


def _visible_key_counts(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
) -> Tensor:
    """:func:`visible_key_counts` without a window, vectorized over splits."""
    base = np.empty(len(split_lens), dtype=np.int64)
    ramp = np.empty(len(split_lens), dtype=bool)
    split = 0
    for doc_len in document_lens:
        first = split
        offset = 0
        while offset < doc_len:
            offset += split_lens[split]
            split += 1
        if offset != doc_len:
            raise ValueError("A split straddles a document boundary.")
        group_after = sum(
            split_lens[i] for i in range(first, split) if attn_modes[i] == "group"
        )
        offset = 0
        hidden_noise = 0
        for i in range(first, split):
            length, mode = split_lens[i], attn_modes[i]
            visible_before = offset - hidden_noise
            if mode == "group":
                group_after -= length
            ramp[i] = mode == "causal"
            # A group token also sees the group splits after its own.
            later = group_after if mode == "group" else 0
            base[i] = visible_before if ramp[i] else visible_before + length + later
            if mode == "noise":
                hidden_noise += length
            offset += length
    lens = np.asarray(split_lens, dtype=np.int64)
    starts = np.cumsum(lens) - lens
    within = np.arange(int(lens.sum()), dtype=np.int64) - np.repeat(starts, lens)
    counts = np.repeat(base, lens) + np.where(np.repeat(ramp, lens), within + 1, 0)
    return torch.from_numpy(counts)


def _windowed_visible_key_counts(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    window: int,
) -> Tensor:
    """:func:`visible_key_counts` with a sliding window, vectorized.

    A token at ``p`` sees the keys of ``[lo, p]`` (``lo`` = the later of its
    document start and ``p - window + 1``) that are not noise, plus every token
    of its own segment; those segment tokens already inside ``[lo, p]`` and not
    noise are counted once.
    """
    layout = build_packed_layout(document_lens, split_lens, attn_modes)
    n = layout.seq_len
    pos = torch.arange(n, dtype=torch.int64)
    starts = torch.tensor([0, *document_lens], dtype=torch.int64).cumsum(0)[:-1]
    doc_start = starts[layout.document_id.to(torch.int64) - 1]
    lo = torch.maximum(doc_start, pos - window + 1)
    visible = (layout.noise_seq_id < 0).to(torch.int64)
    prefix = torch.cat([torch.zeros(1, dtype=torch.int64), visible.cumsum(0)])
    counts = prefix[pos + 1] - prefix[lo]
    seg = layout.full_and_noise_seq_id.to(torch.int64)
    in_seg = seg >= 0
    if bool(in_seg.any()):
        seg_size = torch.bincount(seg[in_seg])
        # Segment tokens that are not noise, sorted by (segment, position).
        keyed = in_seg & (visible > 0)
        keys = torch.sort(seg[keyed] * (n + 1) + pos[keyed]).values
        q_seg = seg[in_seg]
        base = q_seg * (n + 1)
        inside = torch.searchsorted(keys, base + pos[in_seg], right=True) - (
            torch.searchsorted(keys, base + lo[in_seg], right=False)
        )
        counts[in_seg] += seg_size[q_seg] - inside
    return counts
