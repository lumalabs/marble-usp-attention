# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Sliding-window attention on top of the packed mixed-mode mask.

A sliding window of ``W`` keys restricts the causal part of the mask: a query
at position ``p`` sees an earlier key at ``k`` only when ``p - k < W`` (its own
token included, so ``W`` keys at most). Bidirectional spans are exempt, as in
Gemma 3's sliding layers: a token of a full, noise or group span still sees its
whole span. The predicate is::

    ((causal & (q_pos - kv_pos < W)) | same_full_or_noise_segment)
        & remove_noise & same_document

A window only removes pairs, so the block classification of the unwindowed
mask stays valid after a cheap post-pass (:func:`window_block_types`): blocks
entirely outside the window become empty unless a bidirectional segment can
connect them, and full blocks the window cuts become partial (the windowed
``mask_mod`` then decides each pair).
"""

from __future__ import annotations

import torch
from torch import Tensor

BLOCK_EMPTY = 0
BLOCK_PARTIAL = 1
BLOCK_FULL = 2

# Position used for tokens outside any bidirectional segment: no query
# position is at or below it.
_NO_SEGMENT = -1


def validate_window(window: int | None) -> int | None:
    """Return ``window`` after checking it is ``None`` or a positive int.

    Raises:
        ValueError: If the window is not positive.
    """
    if window is not None and window <= 0:
        raise ValueError(f"sliding_window must be positive, got {window}.")
    return window


def segment_end_positions(
    full_and_noise_seq_id: Tensor, position_ids: Tensor | None = None
) -> Tensor:
    """Last position of each token's bidirectional segment, ``-1`` for causal tokens.

    Args:
        full_and_noise_seq_id: Segment id per token in original order (``-1``
            for causal tokens), as in :class:`~usp_attention.ring.packed_layout.PackedLayout`.
        position_ids: Position of each token; defaults to ``arange``.

    Returns:
        ``int64`` tensor with, for every token of a full, noise or group
        segment, the largest position of any token in that segment.
    """
    seg = full_and_noise_seq_id.to(torch.int64)
    if position_ids is None:
        position_ids = torch.arange(seg.numel(), device=seg.device)
    position_ids = position_ids.to(device=seg.device, dtype=torch.int64)
    in_seg = seg >= 0
    out = torch.full_like(seg, _NO_SEGMENT)
    if not bool(in_seg.any()):
        return out
    ids = seg[in_seg]
    ends = torch.full(
        (int(ids.max()) + 1,), _NO_SEGMENT, dtype=torch.int64, device=seg.device
    )
    ends.scatter_reduce_(0, ids, position_ids[in_seg], reduce="amax")
    out[in_seg] = ends[ids]
    return out


def _block_reduce(x: Tensor, block_size: int, op: str, fill: int) -> Tensor:
    pad = (-x.numel()) % block_size
    if pad:
        x = torch.cat([x, x.new_full((pad,), fill)])
    blocks = x.view(-1, block_size)
    return blocks.amin(dim=1) if op == "min" else blocks.amax(dim=1)


def window_block_types(
    block_type: Tensor,
    *,
    q_pos: Tensor,
    kv_pos: Tensor,
    kv_segment_end: Tensor,
    window: int,
    block_size: tuple[int, int],
) -> Tensor:
    """Apply a sliding window to a block classification of the unwindowed mask.

    Args:
        block_type: ``(n_q_blocks, n_kv_blocks)`` classification of the mask
            without a window.
        q_pos: Position of every query token (in the order of ``block_type``'s rows).
        kv_pos: Position of every key token.
        kv_segment_end: :func:`segment_end_positions` of every key token.
        window: Sliding window in keys.
        block_size: ``(q_block, kv_block)`` sizes of ``block_type``.

    Returns:
        A new ``block_type``: never less sparse, and exact on empty blocks
        whenever the window alone empties them.
    """
    big = torch.iinfo(torch.int64).max
    q_pos = q_pos.to(torch.int64)
    kv_pos = kv_pos.to(torch.int64)
    q_min = _block_reduce(q_pos, block_size[0], "min", big)[:, None]
    q_max = _block_reduce(q_pos, block_size[0], "max", -1)[:, None]
    kv_min = _block_reduce(kv_pos, block_size[1], "min", big)[None, :]
    kv_max = _block_reduce(kv_pos, block_size[1], "max", -1)[None, :]
    kv_reach = _block_reduce(kv_segment_end.to(torch.int64), block_size[1], "max", -1)
    # Every pair is at least `window` apart: only a shared segment can connect
    # them, which needs a key segment that reaches the query block.
    outside = (q_min - kv_max) >= window
    cut = (q_max - kv_min) >= window
    connected = q_min <= kv_reach[None, :]
    out = block_type.clone()
    out[cut & (out == BLOCK_FULL)] = BLOCK_PARTIAL
    out[outside & ~connected] = BLOCK_EMPTY
    return out


def bcsr_from_block_type(
    block_type: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Forward and backward BCSR tensors of a ``(n_q, n_kv)`` block classification.

    Returns:
        ``(kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
        q_num_blocks, q_indices, full_q_num_blocks, full_q_indices)``, each
        with leading ``(1, 1)`` batch and head dims; the ``kv_*`` / ``q_*``
        pairs hold partial blocks only, as in ``BlockMask``.
    """
    # Imported here: flash4_block_sparse imports this module's constants.
    from usp_attention.fa4.flash4_block_sparse import (
        _dense_to_ordered,
        _transpose_ordered,
    )

    partial = (block_type == BLOCK_PARTIAL).to(torch.int8)[None, None]
    full = (block_type == BLOCK_FULL).to(torch.int8)[None, None]
    kv_cnt, kv_idx = _dense_to_ordered(partial)
    full_kv_cnt, full_kv_idx = _dense_to_ordered(full)
    n_kv = block_type.shape[1]
    q_cnt, q_idx = _transpose_ordered(kv_cnt, kv_idx, num_cols=n_kv)
    full_q_cnt, full_q_idx = _transpose_ordered(full_kv_cnt, full_kv_idx, num_cols=n_kv)
    return (
        kv_cnt,
        kv_idx,
        full_kv_cnt,
        full_kv_idx,
        q_cnt,
        q_idx,
        full_q_cnt,
        full_q_idx,
    )


def _rebuilt(result, block_type: Tensor):
    """``result`` (a block-sparse result dataclass) with ``block_type`` and its BCSR replaced."""
    import dataclasses

    (kv_cnt, kv_idx, full_kv_cnt, full_kv_idx, q_cnt, q_idx, full_q_cnt, full_q_idx) = (
        bcsr_from_block_type(block_type)
    )
    return dataclasses.replace(
        result,
        block_type=block_type,
        kv_num_blocks=kv_cnt,
        kv_indices=kv_idx,
        full_kv_num_blocks=full_kv_cnt,
        full_kv_indices=full_kv_idx,
        q_num_blocks=q_cnt,
        q_indices=q_idx,
        full_q_num_blocks=full_q_cnt,
        full_q_indices=full_q_idx,
    )


def windowed_self_attention_blocks(result, full_and_noise_seq_id: Tensor, window: int):
    """Single-rank block classification with a sliding window (positions = indices)."""
    pos = torch.arange(
        full_and_noise_seq_id.numel(), device=full_and_noise_seq_id.device
    )
    block_type = window_block_types(
        result.block_type,
        q_pos=pos,
        kv_pos=pos,
        kv_segment_end=segment_end_positions(full_and_noise_seq_id, pos),
        window=window,
        block_size=result.block_size,
    )
    return _rebuilt(result, block_type)


def windowed_cross_blocks(
    result, aux_tensors: list[Tensor], kv_segment_end: Tensor, window: int
):
    """Ring-step (cross Q/KV) block classification with a sliding window.

    Args:
        result: Classification of the unwindowed step mask.
        aux_tensors: The step's 8 cross Q/KV aux tensors (positions at 6 and 7).
        kv_segment_end: :func:`segment_end_positions` of the step's key tokens.
        window: Sliding window in keys.
    """
    block_type = window_block_types(
        result.block_type,
        q_pos=aux_tensors[6],
        kv_pos=aux_tensors[7],
        kv_segment_end=kv_segment_end,
        window=window,
        block_size=result.block_size,
    )
    return _rebuilt(result, block_type)
