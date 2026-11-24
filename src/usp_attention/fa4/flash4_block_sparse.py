# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Direct block classification for single-rank packed self-attention.

The element mask is:

``(causal | full_noise) & remove_noise & same_document``.

This module classifies block pairs without materializing the quadratic element
mask. Classification is conservative: uncertain blocks remain partial so the
FA4 kernel evaluates the element-level mask function.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

import torch

BLOCK_EMPTY = 0
BLOCK_PARTIAL = 1
BLOCK_FULL = 2


@dataclass(kw_only=True, slots=True)
class SelfAttentionBlockSparseResult:
    """Backend-independent block classification and BCSR indices."""

    n_blocks: int
    block_size: tuple[int, int]
    block_type: torch.Tensor
    kv_num_blocks: torch.Tensor
    kv_indices: torch.Tensor
    full_kv_num_blocks: torch.Tensor
    full_kv_indices: torch.Tensor
    q_num_blocks: torch.Tensor
    q_indices: torch.Tensor
    full_q_num_blocks: torch.Tensor
    full_q_indices: torch.Tensor


def _pad_to_blocks(
    tensor: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    remainder = tensor.numel() % block_size
    if remainder == 0:
        return tensor
    # Repeating the final valid value keeps tail-block min/max and category
    # membership identical to the unpadded valid region.
    return torch.cat([tensor, tensor[-1:].expand(block_size - remainder)])


def _category_memberships(
    columns: tuple[torch.Tensor, ...],
    *,
    base_mask: torch.Tensor,
    subset_mask: torch.Tensor | None,
    block_size: int,
    n_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build per-block membership for token categories and an optional subset."""
    token_indices = torch.where(base_mask)[0]
    if token_indices.numel() == 0:
        empty = torch.zeros(n_blocks, 0, dtype=torch.bool, device=base_mask.device)
        return empty, empty if subset_mask is not None else None

    keys = torch.stack(
        [column[token_indices].to(torch.int64) for column in columns], dim=1
    )
    unique_keys, category_ids = torch.unique(keys, dim=0, return_inverse=True)
    n_categories = unique_keys.shape[0]
    block_ids = token_indices // block_size

    membership = torch.zeros(
        n_blocks, n_categories, dtype=torch.bool, device=base_mask.device
    )
    membership[block_ids, category_ids] = True
    if subset_mask is None:
        return membership, None

    in_subset = subset_mask[token_indices]
    subset_membership = torch.zeros_like(membership)
    subset_membership[block_ids[in_subset], category_ids[in_subset]] = True
    return membership, subset_membership


def _shares_category(
    q_membership: torch.Tensor, kv_membership: torch.Tensor
) -> torch.Tensor:
    """Whether each Q/KV block pair shares at least one category."""
    if q_membership.shape[1] == 0:
        return torch.zeros(
            q_membership.shape[0],
            kv_membership.shape[0],
            dtype=torch.bool,
            device=q_membership.device,
        )
    return (q_membership.float() @ kv_membership.float().T) > 0


def _dense_to_ordered(dense_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: the count of set columns and all columns, set ones first.

    Same result as a stable descending argsort of the 0 / 1 mask (set columns
    ascending, then unset columns ascending), computed with prefix sums and
    one scatter in linear time.
    """
    mask = dense_mask.to(dtype=torch.bool)
    num_blocks_in_row = mask.sum(dim=-1, dtype=torch.int32)
    pos_set = mask.cumsum(dim=-1, dtype=torch.int32) - 1
    pos_unset = (~mask).cumsum(dim=-1, dtype=torch.int32) - 1
    pos = torch.where(mask, pos_set, num_blocks_in_row.unsqueeze(-1) + pos_unset)
    cols = torch.arange(mask.shape[-1], device=mask.device, dtype=torch.int32)
    col_indices = torch.empty_like(pos)
    col_indices.scatter_(-1, pos.long(), cols.expand_as(pos))
    return num_blocks_in_row.contiguous(), col_indices.contiguous()


def _ordered_to_dense(
    num_blocks_in_row: torch.Tensor,
    col_indices: torch.Tensor,
    *,
    num_cols: int,
) -> torch.Tensor:
    out_shape = list(num_blocks_in_row.shape) + [num_cols]
    dense = torch.zeros(out_shape, dtype=torch.int32, device=num_blocks_in_row.device)
    col_range = torch.arange(col_indices.shape[-1], device=col_indices.device)
    valid = col_range < num_blocks_in_row.unsqueeze(-1)
    safe_indices = col_indices.clamp(max=num_cols - 1).long()
    dense.scatter_(-1, safe_indices, valid.to(torch.int32))
    return dense.contiguous()


def _transpose_ordered(
    num_blocks_in_row: torch.Tensor,
    col_indices: torch.Tensor,
    *,
    num_cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense = _ordered_to_dense(num_blocks_in_row, col_indices, num_cols=num_cols)
    return _dense_to_ordered(dense.transpose(-2, -1))


def _classify_self_attention_blocks_torch(
    full_and_noise_seq_id: torch.Tensor,
    noise_seq_id: torch.Tensor,
    document_id: torch.Tensor,
    *,
    block_size: int = 128,
) -> SelfAttentionBlockSparseResult:
    """Classify a packed single-rank self-attention mask directly.

    The three inputs are the same per-position tensors consumed by the
    single-rank CuTe ``mask_mod``. Causality follows sequence order, so unlike
    ring attention no separate Q/KV metadata or absolute-position tensors are
    needed.

    Args:
        full_and_noise_seq_id: Segment ID for full/noise positions, or ``-1``
            for causal-only positions.
        noise_seq_id: Noise segment ID for noise positions, or ``-1``.
        document_id: Non-negative document/sample ID per position.
        block_size: Square attention block size.

    Returns:
        Conservative block types and partial/full BCSR tensors for forward and
        backward.

    Raises:
        ValueError: If tensors are not compatible one-dimensional sequences or
            ``block_size`` is invalid.
    """
    tensors = (full_and_noise_seq_id, noise_seq_id, document_id)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError(
            "FA4 self-attention aux tensors must be one-dimensional, got "
            f"{[tuple(tensor.shape) for tensor in tensors]}."
        )
    lengths = {tensor.numel() for tensor in tensors}
    if len(lengths) != 1:
        raise ValueError(
            "FA4 self-attention aux tensors must have equal lengths, got "
            f"{[tensor.numel() for tensor in tensors]}."
        )
    seq_len = full_and_noise_seq_id.numel()
    if seq_len == 0:
        raise ValueError("FA4 self-attention aux tensors must be non-empty.")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(
            "FA4 self-attention aux tensors must share a device, got "
            f"{[tensor.device for tensor in tensors]}."
        )

    fn = _pad_to_blocks(full_and_noise_seq_id, block_size)
    noise = _pad_to_blocks(noise_seq_id, block_size)
    doc = _pad_to_blocks(document_id, block_size)
    n_blocks = fn.numel() // block_size

    fn_blocks = fn.reshape(n_blocks, block_size)
    noise_blocks = noise.reshape(n_blocks, block_size)
    doc_blocks = doc.reshape(n_blocks, block_size)

    fn_min = fn_blocks.min(dim=1).values
    fn_max = fn_blocks.max(dim=1).values
    noise_min = noise_blocks.min(dim=1).values
    noise_max = noise_blocks.max(dim=1).values
    doc_min = doc_blocks.min(dim=1).values
    doc_max = doc_blocks.max(dim=1).values

    q_fn_min, kv_fn_min = fn_min[:, None], fn_min[None, :]
    q_fn_max, kv_fn_max = fn_max[:, None], fn_max[None, :]
    q_noise_min = noise_min[:, None]
    q_noise_max = noise_max[:, None]
    kv_noise_max = noise_max[None, :]
    q_doc_min, kv_doc_min = doc_min[:, None], doc_min[None, :]
    q_doc_max, kv_doc_max = doc_max[:, None], doc_max[None, :]

    q_single_doc = q_doc_min == q_doc_max
    kv_single_doc = kv_doc_min == kv_doc_max
    same_doc = q_single_doc & kv_single_doc & (q_doc_min == kv_doc_min)

    q_fn_uniform = q_fn_min == q_fn_max
    kv_fn_uniform = kv_fn_min == kv_fn_max
    all_full_attn = (
        same_doc
        & q_fn_uniform
        & kv_fn_uniform
        & (q_fn_min >= 0)
        & (kv_fn_min >= 0)
        & (q_fn_min == kv_fn_min)
    )

    all_tokens = torch.ones_like(noise, dtype=torch.bool)
    no_noise = noise < 0
    positive_noise = ~no_noise
    positive_fn = fn >= 0

    # Exact existence checks for the two non-diagonal cases:
    #
    # below diagonal: causal is true, so a pair attends iff it shares a
    # document and remove_noise passes;
    # above diagonal: causal is false, so a pair additionally needs the same
    # positive full/noise segment ID.
    doc_membership, doc_without_noise = _category_memberships(
        (doc,),
        base_mask=all_tokens,
        subset_mask=no_noise,
        block_size=block_size,
        n_blocks=n_blocks,
    )
    doc_noise_membership, _ = _category_memberships(
        (doc, noise),
        base_mask=positive_noise,
        subset_mask=None,
        block_size=block_size,
        n_blocks=n_blocks,
    )
    if doc_without_noise is None:
        raise RuntimeError("Internal FA4 document category construction failed.")
    causal_active = _shares_category(
        doc_membership, doc_without_noise
    ) | _shares_category(doc_noise_membership, doc_noise_membership)

    doc_fn_membership, doc_fn_without_noise = _category_memberships(
        (doc, fn),
        base_mask=positive_fn,
        subset_mask=positive_fn & no_noise,
        block_size=block_size,
        n_blocks=n_blocks,
    )
    doc_fn_noise_membership, _ = _category_memberships(
        (doc, fn, noise),
        base_mask=positive_fn & positive_noise,
        subset_mask=None,
        block_size=block_size,
        n_blocks=n_blocks,
    )
    if doc_fn_without_noise is None:
        raise RuntimeError("Internal FA4 full/noise category construction failed.")
    full_noise_active = _shares_category(
        doc_fn_membership, doc_fn_without_noise
    ) | _shares_category(doc_fn_noise_membership, doc_fn_noise_membership)

    kv_no_noise = kv_noise_max < 0
    # For remove_noise to hold for every pair, Q must be uniformly one positive
    # noise ID N and every positive KV ID must also be N (KV -1 is allowed).
    # Summarizing positive KV IDs by min/max avoids materializing the previous
    # [n_q_blocks, n_kv_blocks, n_noise_ids] broadcast.
    positive_noise_min = (
        torch.where(
            noise_blocks >= 0,
            noise_blocks,
            torch.full_like(noise_blocks, torch.inf),
        )
        .min(dim=1)
        .values
    )
    positive_noise_max = (
        torch.where(
            noise_blocks >= 0,
            noise_blocks,
            torch.full_like(noise_blocks, -torch.inf),
        )
        .max(dim=1)
        .values
    )
    kv_single_positive_noise = (positive_noise_min == positive_noise_max) & (
        positive_noise_min >= 0
    )
    q_single_positive_noise = (q_noise_min == q_noise_max) & (q_noise_min >= 0)
    remove_noise_all = kv_no_noise | (
        q_single_positive_noise
        & kv_single_positive_noise[None, :]
        & (q_noise_min == positive_noise_min[None, :])
    )
    full_attn_safe = all_full_attn & remove_noise_all

    block_index = torch.arange(n_blocks, device=fn.device)
    below_diagonal = block_index[:, None] > block_index[None, :]
    above_diagonal = block_index[:, None] < block_index[None, :]
    diagonal = block_index[:, None] == block_index[None, :]
    below_diagonal_full = below_diagonal & same_doc & remove_noise_all
    valid_counts = torch.full(
        (n_blocks,), block_size, dtype=torch.int64, device=fn.device
    )
    valid_counts[-1] = seq_len - (n_blocks - 1) * block_size
    singleton_diagonal_full = diagonal & (valid_counts[:, None] == 1)
    block_empty = (below_diagonal & ~causal_active) | (
        above_diagonal & ~full_noise_active
    )

    block_type = torch.full(
        (n_blocks, n_blocks),
        BLOCK_PARTIAL,
        dtype=torch.int32,
        device=fn.device,
    )
    block_type[block_empty] = BLOCK_EMPTY
    block_type[full_attn_safe] = BLOCK_FULL
    block_type[below_diagonal_full] = BLOCK_FULL
    block_type[singleton_diagonal_full] = BLOCK_FULL

    partial_mask = (block_type == BLOCK_PARTIAL).to(torch.int8)[None, None]
    full_mask = (block_type == BLOCK_FULL).to(torch.int8)[None, None]
    kv_num_blocks, kv_indices = _dense_to_ordered(partial_mask)
    full_kv_num_blocks, full_kv_indices = _dense_to_ordered(full_mask)
    q_num_blocks, q_indices = _transpose_ordered(
        kv_num_blocks, kv_indices, num_cols=n_blocks
    )
    full_q_num_blocks, full_q_indices = _transpose_ordered(
        full_kv_num_blocks, full_kv_indices, num_cols=n_blocks
    )

    return SelfAttentionBlockSparseResult(
        n_blocks=n_blocks,
        block_size=(block_size, block_size),
        block_type=block_type,
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
    )


def classify_self_attention_blocks(
    full_and_noise_seq_id: torch.Tensor,
    noise_seq_id: torch.Tensor,
    document_id: torch.Tensor,
    *,
    block_size: int = 128,
) -> SelfAttentionBlockSparseResult:
    """Classify packed self-attention blocks with the fastest available path.

    The CuTe DSL kernel is the default for CUDA float32 segment/noise IDs,
    int64 document IDs, and supported power-of-two block sizes on Hopper GPUs.
    The backend-independent PyTorch implementation is used otherwise, when
    CUTLASS DSL is unavailable, or when an ID does not fit the kernel's
    per-block bitmaps (segment or noise IDs of 16384 or more, document IDs
    outside ``[0, 16384)``).

    Args:
        full_and_noise_seq_id: Segment ID for full/noise positions, or ``-1``
            for causal-only positions.
        noise_seq_id: Noise segment ID for noise positions, or ``-1``.
        document_id: Non-negative document/sample ID per position.
        block_size: Square attention block size.

    Returns:
        Exact block types and partial/full BCSR tensors for forward and
        backward.

    Raises:
        ValueError: If tensors are not compatible one-dimensional sequences or
            ``block_size`` is invalid.
    """
    tensors = (full_and_noise_seq_id, noise_seq_id, document_id)
    use_cute_dsl = (
        4 <= block_size <= 1024
        and block_size & (block_size - 1) == 0
        and full_and_noise_seq_id.dtype == torch.float32
        and noise_seq_id.dtype == torch.float32
        and document_id.dtype == torch.int64
        and all(tensor.is_cuda for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and torch.cuda.get_device_capability(full_and_noise_seq_id.device)[0] >= 9
    )
    if use_cute_dsl:
        try:
            # [MLR-02.08] cutlass is an optional, platform-specific dependency.
            from usp_attention.fa4.flash4_block_sparse_cutedsl import (
                BitmapIdRangeError,
                classify_self_attention_blocks_cute_dsl,
            )
        except ImportError:
            pass
        else:
            try:
                return classify_self_attention_blocks_cute_dsl(
                    full_and_noise_seq_id,
                    noise_seq_id,
                    document_id,
                    block_size=block_size,
                )
            except BitmapIdRangeError:
                pass  # IDs past the kernel's bitmaps; the torch path has no limit.

    return _classify_self_attention_blocks_torch(
        full_and_noise_seq_id,
        noise_seq_id,
        document_id,
        block_size=block_size,
    )


def _is_sm100_class(device: torch.device | str | None) -> bool:
    if device is None or torch.device(device).type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major in (10, 11)


def sparse_q_block_factor(device: torch.device | str | None) -> int:
    """How many 128-token Q blocks one FA4 block-sparse Q block spans on ``device``.

    FA4 on SM100-class GPUs (B200 / B300) processes two 128-row Q tiles per CTA,
    so its block-sparse metadata must use 256-row Q blocks. Other GPUs use
    128-row Q blocks. ``USP_FA4_SPARSE_Q_FACTOR`` overrides the choice (used to
    validate the coarse layout on Hopper).
    """
    override = os.environ.get("USP_FA4_SPARSE_Q_FACTOR")
    if override:
        return int(override)
    return 2 if _is_sm100_class(device) else 1


def _fa4_sparse_bwd_supports_2cta() -> bool:
    """Whether the installed FA4 runs block-sparse SM100 backward with 2-CTA MMA.

    FA4 releases that ship ``block_sparse_bwd_supports_2cta`` (4.0.0b34 and
    later) do so when a sparse KV block spans an even number of 128-wide tiles;
    earlier releases fall back to single-CTA backward for any block sparsity.
    """
    try:
        # [MLR-02.08] flash_attn.cute is an optional, platform-specific dependency.
        from flash_attn.cute import block_sparsity
    except ImportError:
        return False
    return hasattr(block_sparsity, "block_sparse_bwd_supports_2cta")


def sparse_kv_block_factor(device: torch.device | str | None) -> int:
    """How many 128-token KV blocks one FA4 block-sparse KV block spans.

    256-wide sparse KV blocks let FA4 >= 4.0.0b34 run the SM100 block-sparse
    backward with 2-CTA MMA. On B300 that path is currently slower than the
    single-CTA backward (mixed 512k, 4 heads: 374 vs 344 ms forward + backward),
    so the default stays at 128-wide blocks; ``USP_FA4_SPARSE_KV_FACTOR=2``
    opts in (requires an FA4 with ``block_sparse_bwd_supports_2cta``).
    """
    override = os.environ.get("USP_FA4_SPARSE_KV_FACTOR")
    if override:
        factor = int(override)
        if factor > 1 and not (
            _is_sm100_class(device) and _fa4_sparse_bwd_supports_2cta()
        ):
            raise ValueError(
                "USP_FA4_SPARSE_KV_FACTOR > 1 needs an SM100-class GPU and FA4 >= 4.0.0b34."
            )
        return factor
    return 1


def coarsen_q_blocks(
    full_cnt: torch.Tensor,
    full_idx: torch.Tensor,
    mask_cnt: torch.Tensor,
    mask_idx: torch.Tensor,
    *,
    num_kv_blocks: int,
    factor: int,
    kv_factor: int = 1,
) -> tuple[torch.Tensor, ...]:
    """Merge ``factor`` Q blocks and ``kv_factor`` KV blocks of block-sparse lists.

    A merged block is full when every member block is full, empty when every
    member is empty, and masked otherwise (the ``mask_mod`` then resolves it
    exactly). Missing trailing blocks count as empty.

    Returns:
        Forward ``(full_cnt, full_idx, mask_cnt, mask_idx)`` (KV-indexed) followed
        by backward ``(full_cnt, full_idx, mask_cnt, mask_idx)`` (Q-indexed), all
        int32 and contiguous.
    """
    full = _ordered_to_dense(full_cnt, full_idx, num_cols=num_kv_blocks).bool()
    masked = _ordered_to_dense(mask_cnt, mask_idx, num_cols=num_kv_blocks).bool()
    pad_q = (-full.shape[-2]) % factor
    pad_kv = (-full.shape[-1]) % kv_factor
    if pad_q or pad_kv:
        full = torch.nn.functional.pad(full, (0, pad_kv, 0, pad_q))
        masked = torch.nn.functional.pad(masked, (0, pad_kv, 0, pad_q))
    rows, cols = full.shape[-2] // factor, full.shape[-1] // kv_factor
    shape = (*full.shape[:-2], rows, factor, cols, kv_factor)
    full = full.reshape(shape)
    visible = full | masked.reshape(shape)
    full_c = full.all(dim=-1).all(dim=-2)
    mask_c = visible.any(dim=-1).any(dim=-2) & ~full_c
    fwd = (*_dense_to_ordered(full_c), *_dense_to_ordered(mask_c))
    bwd = (
        *_dense_to_ordered(full_c.transpose(-2, -1)),
        *_dense_to_ordered(mask_c.transpose(-2, -1)),
    )
    return (*fwd, *bwd)


_SM100_FWD_SCHEDULER_PATCHED = False


def install_sm100_fwd_scheduler_patch() -> None:
    """Run FA4's SM100 block-sparse forward with one CTA per tile.

    For non-causal calls FA4 uses a static persistent scheduler that deals
    query tiles to CTAs round-robin. With a packed block-sparse mask the tiles
    carry very different amounts of work, so the CTA that draws the heaviest
    tiles sets the kernel time. Launching one CTA per tile lets the hardware
    balance them: 11-12% faster forward on packed 512k masks on B300, identical
    results. Applies only to calls with a ``mask_mod`` and no causal flag;
    ``USP_FA4_FWD_PERSISTENT=1`` keeps FA4's default.
    """
    global _SM100_FWD_SCHEDULER_PATCHED
    if _SM100_FWD_SCHEDULER_PATCHED or os.environ.get("USP_FA4_FWD_PERSISTENT") == "1":
        return
    try:
        # [MLR-02.08] flash_attn.cute is an optional, platform-specific dependency.
        from flash_attn.cute import flash_fwd_sm100
    except ImportError:
        return
    cls = flash_fwd_sm100.FlashAttentionForwardSm100
    init = cls.__init__
    params = inspect.signature(init).parameters
    # FA4 renamed the flag between releases.
    flag = next(
        (p for p in ("is_static_persistent", "is_persistent") if p in params), None
    )
    if flag is None:
        return

    def patched_init(self, *args, **kwargs):
        if kwargs.get("mask_mod") is not None and not kwargs.get("is_causal", False):
            kwargs[flag] = False
        init(self, *args, **kwargs)

    cls.__init__ = patched_init
    _SM100_FWD_SCHEDULER_PATCHED = True
