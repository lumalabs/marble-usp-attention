# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-FileCopyrightText: 2025 Bytedance Ltd. and/or its affiliates (packed-mask encoding adapted from BAGEL create_sparse_mask)
# SPDX-License-Identifier: MIT AND Apache-2.0
"""
Direct Flash Attention 4 (FA4) with flex attention mask_mod.

Translates the packed-sequence mask of BAGEL's ``create_sparse_mask()``
(``data/data_utils.py`` in https://github.com/ByteDance-Seed/Bagel, Apache-2.0:
per-token full/noise segment, noise and document ids combined as
``(causal | same segment) & remove_noise & same document``) into a CuTe DSL
mask_mod for use with flash_attn.cute.interface._flash_attn_fwd directly,
bypassing torch.compile and the torch flex_attention translation layer.

The key insight is that torch's flex_attention pipeline:

  1. Traces your Python mask_mod through dynamo
  2. Lowers it to inductor IR
  3. Generates CuTe DSL code
  4. Compiles the CuTe DSL kernel via flash_attn

This module skips steps 1-3 by writing the CuTe DSL mask_mod directly,
calls flash_attn's _flash_attn_fwd with it, and classifies block-level
sparsity directly from packed-sequence metadata. ``create_block_mask`` is
only used when a caller explicitly requests the reference ``BlockMask``.

Typical usage (mask built once, reused across layers)::

    from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask, flash4_attn

    # Build once (e.g. at the start of a forward pass)
    mask = build_fa4_sparse_mask(document_lens, split_lens, attn_modes, device="cuda")

    # Reuse in every attention layer
    for layer in transformer.layers:
        q, k, v = layer.qkv_proj(x)
        out = flash4_attn(q, k, v, mask)
"""

from __future__ import annotations

import functools
import inspect
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

HAS_FLASH4 = False
try:
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
        flash_attn_func,
        flash_attn_varlen_func,
    )

    HAS_FLASH4 = True
except Exception as e:
    if not isinstance(e, ImportError):
        import logging

        logging.getLogger(__name__).error(
            f"Error importing flash_attn.cute but issue is not an ImportError! {e}!"
        )
    HAS_FLASH4 = False
    _flash_attn_fwd = None
    _flash_attn_bwd = None
    flash_attn_func = None
    flash_attn_varlen_func = None

import numpy as np
import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    or_masks,
)

from usp_attention.fa4.flash4_block_sparse import (
    BLOCK_EMPTY,
    classify_self_attention_blocks,
    coarsen_q_blocks,
    install_sm100_fwd_scheduler_patch,
    sparse_kv_block_factor,
    sparse_q_block_factor,
)
from usp_attention.ring.packed_layout import segment_ids
from usp_attention.ring.ring_planning import CREATE_BLOCK_MASK_DYNAMO_CONFIG
from usp_attention.sliding_window import (
    validate_window,
    windowed_self_attention_blocks,
)

if HAS_FLASH4:
    install_sm100_fwd_scheduler_patch()

# FA4 releases whose backward takes the gradient of the returned LSE.
_FA4_BWD_TAKES_DLSE = (
    HAS_FLASH4 and "dlse" in inspect.signature(_flash_attn_bwd).parameters
)

AttentionMode = Literal["causal", "full", "noise", "group"]


class CreateBlockMaskFn(Protocol):
    def __call__(
        self,
        mask_mod: Callable,
        B: int | None,
        H: int | None,
        Q_LEN: int,
        KV_LEN: int,
        device: str | torch.device | int | None = None,
        BLOCK_SIZE: int | tuple[int, int] = 128,
        _compile: bool = False,
    ) -> BlockMask: ...


CREATE_BLOCK_MASK: CreateBlockMaskFn | None = None

# ── CuTe DSL helpers for reading aux tensors ─────────────────────────────────
#
# SSA (Static Single Assignment) values in CuTe DSL cannot be used directly as
# tensor indices. We must convert SSA → scalar via a register-memory fragment,
# perform the indexed load, then convert back to SSA for further computation.


# ── CuTe DSL mask_mod (singleton — the function is data-independent) ────────
if HAS_FLASH4:

    @cute.jit
    def _ssa_to_idx(ssa_value: cute.TensorSSA, dtype: str) -> cute.Numeric:
        """Convert an SSA value to an indexable scalar."""
        frag = cute.make_rmem_tensor(1, dtype)
        frag.store(ssa_value)
        return frag[0]

    @cute.jit
    def _load_f32(tensor, idx_ssa: cute.TensorSSA) -> cute.TensorSSA:
        """Load a float32 element from a CuTe tensor at an SSA index."""
        idx = _ssa_to_idx(idx_ssa, cutlass.Int32)
        buf = cute.make_rmem_tensor(1, cutlass.Float32)
        buf[0] = tensor[idx]
        return buf.load()

    @cute.jit
    def _load_i64(tensor, idx_ssa: cute.TensorSSA) -> cute.TensorSSA:
        """Load an int64 element from a CuTe tensor at an SSA index."""
        idx = _ssa_to_idx(idx_ssa, cutlass.Int32)
        buf = cute.make_rmem_tensor(1, cutlass.Int64)
        buf[0] = tensor[idx]
        return buf.load()


def _make_sparse_mask_mod(window: int | None = None):
    """
    Create the CuTe DSL mask_mod that implements::

        and_masks(
            or_masks(causal_mask, full_and_noise_mask),
            remove_noise_mask,
            sample_mask,
        )

    The mask_mod is **data-independent** — it always reads from aux_tensors:

        aux_tensors[0] = full_and_noise_seq_id  (float32)
        aux_tensors[1] = noise_seq_id           (float32)
        aux_tensors[2] = document_id            (int64)

    Per-element logic:

        causal        = q_idx >= kv_idx
        full_noise    = (fn_id[q] == fn_id[kv]) & (fn_id[q] >= 0)
        remove_noise  = ~((noise_id[kv] >= 0) & (noise_id[q] != noise_id[kv]))
        sample        = doc_id[q] == doc_id[kv]
        output        = (causal | full_noise) & remove_noise & sample

    Note on ``== 0`` for logical NOT:

        CuTe DSL SSA booleans are 0/1 integers. ``~`` does bitwise NOT
        (gives -1/-2), not logical NOT. We use ``(a & b) == 0`` instead.
        Python precedence: ``&`` binds tighter than ``==``, so
        ``x & y == 0`` parses as ``(x & y) == 0``.

    With a sliding ``window``, the causal term becomes
    ``causal & (q_idx - kv_idx < window)`` (see
    :mod:`usp_attention.sliding_window`). The window is a compile-time
    constant captured in the closure, which FA4's compile cache keys on.
    """

    @cute.jit
    def mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        fn_id = aux_tensors[0]  # full_and_noise_seq_id
        noise_id = aux_tensors[1]  # noise_seq_id
        doc_id = aux_tensors[2]  # document_id

        # causal: q_idx >= kv_idx
        causal = operator.ge(q_idx, kv_idx)
        if cutlass.const_expr(window is not None):
            dist = operator.sub(q_idx, kv_idx)
            causal = causal & operator.lt(dist, cute.full_like(dist, window))

        # full_and_noise: same segment & segment is full/noise
        fn_q = _load_f32(fn_id, q_idx)
        fn_kv = _load_f32(fn_id, kv_idx)
        fn_same = operator.eq(fn_q, fn_kv)
        fn_q_reload = _load_f32(fn_id, q_idx)
        fn_valid = operator.ge(fn_q_reload, cute.full_like(fn_q_reload, 0))
        full_noise = fn_same & fn_valid

        # remove_noise: ~(noise_kv_valid & noise_different)
        noise_kv = _load_f32(noise_id, kv_idx)
        noise_kv_valid = operator.ge(noise_kv, cute.full_like(noise_kv, 0))
        noise_q = _load_f32(noise_id, q_idx)
        noise_kv2 = _load_f32(noise_id, kv_idx)
        noise_diff = operator.ne(noise_q, noise_kv2)

        # sample: same document
        doc_q = _load_i64(doc_id, q_idx)
        doc_kv = _load_i64(doc_id, kv_idx)
        same_doc = operator.eq(doc_q, doc_kv)

        # combine: (causal | full_noise) & ~(noise_kv_valid & noise_diff) & same_doc
        output = ((causal | full_noise) & (noise_kv_valid & noise_diff == 0)) & same_doc
        return output

    return mask_mod


# Module-level singleton: the mask_mod function never changes, only the
# aux_tensors it reads from do.
SPARSE_MASK_MOD: Callable = _make_sparse_mask_mod() if HAS_FLASH4 else lambda: None


@functools.cache
def sparse_mask_mod(window: int | None = None) -> Callable:
    """The packed-layout ``mask_mod``, with an optional sliding window.

    ``None`` (or ``0``) returns :data:`SPARSE_MASK_MOD`; each window gets one
    cached function, so FA4 compiles each window once.
    """
    if not window:
        return SPARSE_MASK_MOD
    return _make_sparse_mask_mod(window)


# ── FA4SparseMask: precomputed mask data ─────────────────────────────────────


@dataclass(kw_only=True, slots=True)
class FA4SparseMask:
    """
    All precomputed mask data needed by ``flash4_attn``.

    Build once per batch/sequence configuration, then pass to every
    attention layer.

    Attributes:

        block_sparse: Forward-direction block-level sparsity tensors (which
            128×128 blocks to skip / fully unmask / per-element mask).
        block_sparse_bwd: Backward-direction block-level sparsity tensors
            (Q-indexed, transposed from forward). ``None`` means backward
            skips block sparsity (still correct, just slower).
        aux_tensors: The three per-position lookup tensors read by the
            CuTe DSL mask_mod at runtime:
            ``[full_and_noise_seq_id, noise_seq_id, document_id]``.
        mask_mod: The CuTe DSL mask_mod function (same singleton for all
            instances — included here for convenience / explicitness).
        sparsity: Block-level sparsity percentage, or None if not computed.
        block_mask: The ``BlockMask`` from ``create_block_mask``, retained
            so callers (e.g. ring attention planning) can reuse it instead
            of rebuilding. ``None`` when not requested.
        sliding_window: Sliding window in keys (see
            :mod:`usp_attention.sliding_window`), ``None`` for none.
    """

    block_sparse: "BlockSparseTensorsTorch" | None
    block_sparse_bwd: "BlockSparseTensorsTorch" | None
    aux_tensors: list[torch.Tensor]
    mask_mod: Callable
    sparsity: float | None
    block_mask: BlockMask | None = None
    sliding_window: int | None = None


# ── Builders ─────────────────────────────────────────────────────────────────


def _build_common_tensors(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build common tensors shared by aux_tensors and block_sparsity builders.

    Args:

        document_lens: length of each segment (for document/sample boundaries).
        split_lens: length of each split (for full/noise attention assignment).
        attn_modes: attention mode per split ("causal", "full", "noise").
        device: target CUDA device.

    Returns:

        Tuple of (full_and_noise_seq, noise_seq, document_ids):

        - full_and_noise_seq: float32 tensor with segment indices for full/noise, -1 elsewhere
        - noise_seq: float32 tensor with segment indices for noise only, -1 elsewhere
        - document_ids: int64 tensor with document index per position
    """
    ids = segment_ids(document_lens, split_lens, attn_modes)
    lens = np.asarray(split_lens, dtype=np.int64)
    noise_ids = np.where(
        np.asarray(attn_modes) == "noise", np.arange(len(split_lens)), -1
    )
    full_and_noise_seq = torch.from_numpy(
        np.repeat(np.asarray(ids, dtype=np.float32), lens)
    ).to(device)
    noise_seq = torch.from_numpy(np.repeat(noise_ids.astype(np.float32), lens)).to(
        device
    )
    document_ids = torch.from_numpy(
        np.repeat(
            np.arange(1, len(document_lens) + 1, dtype=np.int64),
            np.asarray(document_lens, dtype=np.int64),
        )
    ).to(device)

    return full_and_noise_seq, noise_seq, document_ids


def _build_aux_tensors(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    device: str | torch.device = "cuda",
) -> list[torch.Tensor]:
    """
    Build the three per-position lookup tensors that the CuTe DSL mask_mod
    indexes into at runtime.

    Args:

        document_lens: length of each segment (for document/sample boundaries).
        split_lens: length of each split (for full/noise attention assignment).
            Zipped with attn_modes, so len may differ from document_lens.
        attn_modes: attention mode per split ("causal", "full", "noise").
        device: target CUDA device.

    Returns:

        A list of three tensors ``[full_and_noise_seq_id, noise_seq_id, document_id]``:

        - ``full_and_noise_seq_id``: float32 [total_seq_len]
            Segment index for positions in "full" or "noise" splits, -1 elsewhere.
        - ``noise_seq_id``: float32 [total_seq_len]
            Segment index for positions in "noise" splits, -1 elsewhere.
        - ``document_id``: int64 [total_seq_len]
            Document index per position (1-indexed from document_lens).
    """
    if not HAS_FLASH4:
        raise RuntimeError("flash_attn.cute is not installed!")

    full_and_noise_seq_id, noise_seq_id, document_id = _build_common_tensors(
        document_lens, split_lens, attn_modes, device
    )
    return [full_and_noise_seq_id, noise_seq_id, document_id]


def _build_block_sparsity_from_tensors(
    fn_seq: torch.Tensor,
    n_seq: torch.Tensor,
    doc_ids: torch.Tensor,
    seq_len: int,
    device: str | torch.device = "cuda",
    return_sparsity: bool = False,
    sliding_window: int | None = None,
) -> tuple[
    "BlockSparseTensorsTorch", "BlockSparseTensorsTorch" | None, float | None, BlockMask
]:
    """
    Compute block-level sparsity from pre-built per-token tensors.

    Returns:

        block_sparse_fwd: BlockSparseTensorsTorch (KV-indexed, for forward)
        block_sparse_bwd: BlockSparseTensorsTorch (Q-indexed, for backward)
            or None if the BlockMask doesn't provide Q-direction tensors.
        sparsity_pct: block-level sparsity percentage, or None
        block_mask: the underlying ``BlockMask`` object
    """
    if not HAS_FLASH4:
        raise RuntimeError("flash_attn.cute is not installed!")

    def causal_mask(b, h, q_idx, kv_idx):
        if sliding_window:
            return (q_idx >= kv_idx) & (q_idx - kv_idx < sliding_window)
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (fn_seq[q_idx] == fn_seq[kv_idx]) & (fn_seq[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~((n_seq[kv_idx] >= 0) & (n_seq[q_idx] != n_seq[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        return doc_ids[q_idx] == doc_ids[kv_idx]

    sparse_mask = and_masks(
        or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask
    )
    global CREATE_BLOCK_MASK
    with torch._dynamo.config.patch(**CREATE_BLOCK_MASK_DYNAMO_CONFIG):
        if CREATE_BLOCK_MASK is None:
            CREATE_BLOCK_MASK = torch.compile(
                create_block_mask, dynamic=False, fullgraph=True
            )
        block_mask = CREATE_BLOCK_MASK(
            sparse_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

    block_sparse_fwd = BlockSparseTensorsTorch(
        mask_block_cnt=block_mask.kv_num_blocks,
        mask_block_idx=block_mask.kv_indices,
        full_block_cnt=block_mask.full_kv_num_blocks,
        full_block_idx=block_mask.full_kv_indices,
        block_size=block_mask.BLOCK_SIZE,
    )

    block_sparse_bwd = None
    if block_mask.q_num_blocks is not None and block_mask.q_indices is not None:
        block_sparse_bwd = BlockSparseTensorsTorch(
            mask_block_cnt=block_mask.q_num_blocks,
            mask_block_idx=block_mask.q_indices,
            full_block_cnt=block_mask.full_q_num_blocks,
            full_block_idx=block_mask.full_q_indices,
            block_size=block_mask.BLOCK_SIZE,
        )

    sparsity = block_mask.sparsity() if return_sparsity else None
    return block_sparse_fwd, block_sparse_bwd, sparsity, block_mask


def _build_direct_block_sparsity_from_tensors(
    fn_seq: torch.Tensor,
    n_seq: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    block_size: int = 128,
    return_sparsity: bool = False,
    sliding_window: int | None = None,
) -> tuple["BlockSparseTensorsTorch", "BlockSparseTensorsTorch", float | None]:
    """Build FA4 BCSR tensors with the generic single-rank classifier."""
    if not HAS_FLASH4:
        raise RuntimeError("flash_attn.cute is not installed!")
    result = classify_self_attention_blocks(
        fn_seq,
        n_seq,
        doc_ids,
        block_size=block_size,
    )
    if sliding_window:
        result = windowed_self_attention_blocks(result, fn_seq, sliding_window)
    factor = sparse_q_block_factor(fn_seq.device)
    kv_factor = sparse_kv_block_factor(fn_seq.device)
    if factor > 1 or kv_factor > 1:
        # SM100-class FA4 needs (factor * 128)-row Q blocks, and runs its
        # backward with 2-CTA MMA only for (kv_factor * 128)-wide KV blocks.
        fc, fi, mc, mi, bfc, bfi, bmc, bmi = coarsen_q_blocks(
            result.full_kv_num_blocks,
            result.full_kv_indices,
            result.kv_num_blocks,
            result.kv_indices,
            num_kv_blocks=result.kv_indices.shape[-1],
            factor=factor,
            kv_factor=kv_factor,
        )
        coarse_size = (result.block_size[0] * factor, result.block_size[1] * kv_factor)
        block_sparse_fwd = BlockSparseTensorsTorch(
            mask_block_cnt=mc,
            mask_block_idx=mi,
            full_block_cnt=fc,
            full_block_idx=fi,
            block_size=coarse_size,
        )
        block_sparse_bwd = BlockSparseTensorsTorch(
            mask_block_cnt=bmc,
            mask_block_idx=bmi,
            full_block_cnt=bfc,
            full_block_idx=bfi,
            block_size=coarse_size,
        )
    else:
        block_sparse_fwd = BlockSparseTensorsTorch(
            mask_block_cnt=result.kv_num_blocks,
            mask_block_idx=result.kv_indices,
            full_block_cnt=result.full_kv_num_blocks,
            full_block_idx=result.full_kv_indices,
            block_size=result.block_size,
        )
        block_sparse_bwd = BlockSparseTensorsTorch(
            mask_block_cnt=result.q_num_blocks,
            mask_block_idx=result.q_indices,
            full_block_cnt=result.full_q_num_blocks,
            full_block_idx=result.full_q_indices,
            block_size=result.block_size,
        )
    sparsity = (
        100.0 * (result.block_type == BLOCK_EMPTY).to(torch.float32).mean().item()
        if return_sparsity
        else None
    )
    return block_sparse_fwd, block_sparse_bwd, sparsity


def build_fa4_sparse_mask(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    device: str | torch.device = "cuda",
    return_sparsity: bool = False,
    retain_block_mask: bool = False,
    build_block_sparse: bool = True,
    sliding_window: int | None = None,
) -> FA4SparseMask:
    """
    Build all mask data needed by ``flash4_attn``.  Call once per
    batch / sequence configuration, then pass the result to every
    attention layer.

    Args:

        document_lens: per-segment lengths defining document boundaries.
        split_lens: per-split lengths for attention mode assignment
            (zipped with attn_modes).
        attn_modes: attention mode per split — "causal", "full", or "noise".
        device: target CUDA device.
        return_sparsity: whether to compute and store the sparsity percentage.
        retain_block_mask: if True, additionally build and store the reference
            ``BlockMask`` so callers can reuse it (e.g. for ring planning).
        build_block_sparse: if False, skip direct block classification and
            return aux-only metadata. This is useful when a caller only needs
            the FA4 aux tensors and will build sparsity elsewhere.
        sliding_window: restrict causal attention to the last
            ``sliding_window`` keys (bidirectional spans exempt); see
            :mod:`usp_attention.sliding_window`.


    Returns:

        FA4SparseMask containing aux tensors plus either full block sparsity
            metadata or an aux-only view.
    """
    if not HAS_FLASH4:
        raise RuntimeError(
            "flash_attn.cute is not installed! Cannot construct flash4 sparse mask."
        )
    if return_sparsity and not build_block_sparse:
        raise ValueError(
            "Cannot compute FA4 sparsity when build_block_sparse is False."
        )
    if retain_block_mask and not build_block_sparse:
        raise ValueError("Cannot retain a BlockMask when build_block_sparse is False.")
    sliding_window = validate_window(sliding_window)
    seq_len = sum(document_lens)
    aux_tensors = _build_aux_tensors(document_lens, split_lens, attn_modes, device)
    block_sparse_fwd = None
    block_sparse_bwd = None
    sparsity = None
    blk_mask = None
    if build_block_sparse:
        fn_seq, n_seq, doc_ids = aux_tensors
        block_sparse_fwd, block_sparse_bwd, sparsity = (
            _build_direct_block_sparsity_from_tensors(
                fn_seq,
                n_seq,
                doc_ids,
                return_sparsity=return_sparsity,
                sliding_window=sliding_window,
            )
        )
        if retain_block_mask:
            _, _, _, blk_mask = _build_block_sparsity_from_tensors(
                fn_seq,
                n_seq,
                doc_ids,
                seq_len,
                device,
                return_sparsity=False,
                sliding_window=sliding_window,
            )
    return FA4SparseMask(
        block_sparse=block_sparse_fwd,
        block_sparse_bwd=block_sparse_bwd,
        aux_tensors=aux_tensors,
        mask_mod=sparse_mask_mod(sliding_window),
        sparsity=sparsity,
        block_mask=blk_mask if retain_block_mask else None,
        sliding_window=sliding_window,
    )


# ── Dispatcher-visible custom op ─────────────────────────────────────────────
#
# An opaque torch.autograd.Function would be invisible to the dispatcher, so a
# selective-activation-checkpointing (SAC) policy could not MUST_SAVE its
# output. The custom op below registers usp_attention::flash4_attn as a
# real dispatcher op so a SAC context_fn can see it (op is
# torch.ops.usp_attention.flash4_attn.default) and force-save its result.
#
# torch.library.custom_op cannot accept a Callable or a dataclass, so the schema
# is flattened to tensors + simple scalars. The mask_mod is NOT passed; the op
# references the module-level SPARSE_MASK_MOD singleton internally. Block-sparse
# tensors are passed individually (None when absent), and block_size is split
# into two ints. The backward-direction block-sparse tensors are forward inputs
# only (ignored by the forward kernel) so they can be saved for backward.


def _rebuild_block_sparse(
    full_block_cnt: torch.Tensor | None,
    full_block_idx: torch.Tensor | None,
    mask_block_cnt: torch.Tensor | None,
    mask_block_idx: torch.Tensor | None,
    block_size_m: int,
    block_size_n: int,
) -> "BlockSparseTensorsTorch" | None:
    """Rebuild a BlockSparseTensorsTorch from flattened tensors, or None."""
    if all(
        t is None
        for t in (full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx)
    ):
        return None
    return BlockSparseTensorsTorch(
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        block_size=(block_size_m, block_size_n),
    )


if HAS_FLASH4:

    @torch.library.custom_op("usp_attention::flash4_attn", mutates_args=())
    def _flash4_attn_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        aux_tensors: list[torch.Tensor],
        full_block_cnt: torch.Tensor | None,
        full_block_idx: torch.Tensor | None,
        mask_block_cnt: torch.Tensor | None,
        mask_block_idx: torch.Tensor | None,
        block_size_m: int,
        block_size_n: int,
        bwd_full_block_cnt: torch.Tensor | None,
        bwd_full_block_idx: torch.Tensor | None,
        bwd_mask_block_cnt: torch.Tensor | None,
        bwd_mask_block_idx: torch.Tensor | None,
        bwd_block_size_m: int,
        bwd_block_size_n: int,
        softmax_scale: float | None,
        sliding_window: int,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The bwd_* block-sparse tensors and ``deterministic`` are forward
        # inputs only — they are ignored here and saved for backward via
        # setup_context.
        block_sparse_tensors = _rebuild_block_sparse(
            full_block_cnt,
            full_block_idx,
            mask_block_cnt,
            mask_block_idx,
            block_size_m,
            block_size_n,
        )
        out, lse, *_ = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=None,
            window_size_right=None,
            learnable_sink=None,
            softcap=0.0,
            num_splits=1,
            pack_gqa=None,
            mask_mod=sparse_mask_mod(sliding_window),
            block_sparse_tensors=block_sparse_tensors,
            aux_tensors=aux_tensors,
            return_lse=True,
        )
        # Return contiguous tensors so the output layout is deterministic and
        # matches register_fake below (which allocates contiguous). _flash_attn_fwd
        # already produces contiguous output; the .contiguous() calls are no-ops in
        # that case but make the fwd/fake stride contract explicit — a non-contiguous
        # q (e.g. a transposed view) must not leak input strides into the output, or
        # torch.compile's stride guards mismatch the meta output and fail.
        return out.contiguous(), lse.contiguous()

    @_flash4_attn_op.register_fake
    def _flash4_attn_op_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        aux_tensors: list[torch.Tensor],
        full_block_cnt: torch.Tensor | None,
        full_block_idx: torch.Tensor | None,
        mask_block_cnt: torch.Tensor | None,
        mask_block_idx: torch.Tensor | None,
        block_size_m: int,
        block_size_n: int,
        bwd_full_block_cnt: torch.Tensor | None,
        bwd_full_block_idx: torch.Tensor | None,
        bwd_mask_block_cnt: torch.Tensor | None,
        bwd_mask_block_idx: torch.Tensor | None,
        bwd_block_size_m: int,
        bwd_block_size_n: int,
        softmax_scale: float | None,
        sliding_window: int,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q is (B, S, H_q, D); out has q's shape. Allocate CONTIGUOUS (not
        # empty_like, which would copy q's strides) to match the contiguous output
        # the real op returns — otherwise a non-contiguous q makes the meta strides
        # disagree with the real output and torch.compile's stride guards fail.
        # lse shape (B, H_q, S) is a best guess — CONFIRM/ADJUST against
        # _flash_attn_fwd on GPU.
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse = q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32)
        return out, lse

    def _flash4_attn_op_setup_context(ctx, inputs, output):
        (
            q,
            k,
            v,
            aux_tensors,
            full_block_cnt,
            full_block_idx,
            mask_block_cnt,
            mask_block_idx,
            block_size_m,
            block_size_n,
            bwd_full_block_cnt,
            bwd_full_block_idx,
            bwd_mask_block_cnt,
            bwd_mask_block_idx,
            bwd_block_size_m,
            bwd_block_size_n,
            softmax_scale,
            sliding_window,
            deterministic,
        ) = inputs
        out, lse = output
        # Save q, k, v, out, lse plus the four backward-direction block-sparse
        # tensors so backward can rebuild the bwd BlockSparseTensorsTorch.
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            bwd_full_block_cnt,
            bwd_full_block_idx,
            bwd_mask_block_cnt,
            bwd_mask_block_idx,
        )
        ctx.softmax_scale = softmax_scale
        ctx.aux_tensors = aux_tensors
        ctx.bwd_block_size = (bwd_block_size_m, bwd_block_size_n)
        ctx.sliding_window = sliding_window
        ctx.deterministic = deterministic
        if _FA4_BWD_TAKES_DLSE:
            # As upstream FlashAttnFunc: an output the loss does not use gets
            # None instead of zeros, so the LSE gradient is only sent to FA4
            # when the LSE reaches the loss.
            ctx.set_materialize_grads(False)
        else:
            # This FA4 has no LSE gradient: make the LSE a constant instead of
            # silently dropping its gradient.
            ctx.mark_non_differentiable(lse)

    @torch.library.custom_op("usp_attention::flash4_attn_bwd", mutates_args=())
    def _flash4_attn_bwd_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        dlse: torch.Tensor | None,
        aux_tensors: list[torch.Tensor],
        bwd_full_block_cnt: torch.Tensor | None,
        bwd_full_block_idx: torch.Tensor | None,
        bwd_mask_block_cnt: torch.Tensor | None,
        bwd_mask_block_idx: torch.Tensor | None,
        bwd_block_size_m: int,
        bwd_block_size_n: int,
        softmax_scale: float | None,
        sliding_window: int,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Wrap the FA4 backward kernel as its own dispatcher-visible op so the
        # forward op's registered-autograd backward is traceable under
        # torch.compile (AOTAutograd). Calling _flash_attn_bwd directly there
        # would execute a raw CUTE kernel during the FakeTensor trace of the
        # compiled backward and raise "Cannot access data pointer of FakeTensor".
        bwd_block_sparse = _rebuild_block_sparse(
            bwd_full_block_cnt,
            bwd_full_block_idx,
            bwd_mask_block_cnt,
            bwd_mask_block_idx,
            bwd_block_size_m,
            bwd_block_size_n,
        )
        # dlse is None unless the LSE reached the loss (and this FA4 takes it).
        lse_grad = {} if dlse is None else {"dlse": dlse.contiguous()}
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            softmax_scale,
            False,
            0.0,
            window_size_left=None,
            window_size_right=None,
            deterministic=deterministic,
            mask_mod=sparse_mask_mod(sliding_window),
            aux_tensors=aux_tensors,
            block_sparse_tensors=bwd_block_sparse,
            **lse_grad,
        )
        # Contiguous so the grad strides are deterministic and match register_fake
        # (see the forward op's rationale — non-contiguous q/k/v must not leak
        # strides into dq/dk/dv or torch.compile's stride guards mismatch the meta).
        return dq.contiguous(), dk.contiguous(), dv.contiguous()

    @_flash4_attn_bwd_op.register_fake
    def _flash4_attn_bwd_op_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        dlse: torch.Tensor | None,
        aux_tensors: list[torch.Tensor],
        bwd_full_block_cnt: torch.Tensor | None,
        bwd_full_block_idx: torch.Tensor | None,
        bwd_mask_block_cnt: torch.Tensor | None,
        bwd_mask_block_idx: torch.Tensor | None,
        bwd_block_size_m: int,
        bwd_block_size_n: int,
        softmax_scale: float | None,
        sliding_window: int,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # dq/dk/dv have q/k/v shapes (k/v may have fewer heads under GQA). Allocate
        # CONTIGUOUS (not empty_like) to match the contiguous grads the real op
        # returns, so meta strides can't disagree with the real output.
        return (
            torch.empty(q.shape, dtype=q.dtype, device=q.device),
            torch.empty(k.shape, dtype=k.dtype, device=k.device),
            torch.empty(v.shape, dtype=v.dtype, device=v.device),
        )

    def _flash4_attn_op_backward(ctx, dout, dlse):
        (
            q,
            k,
            v,
            out,
            lse,
            bwd_full_block_cnt,
            bwd_full_block_idx,
            bwd_mask_block_cnt,
            bwd_mask_block_idx,
        ) = ctx.saved_tensors
        # Without materialized grads, dout is None when only the LSE reached
        # the loss; dlse is None when it did not, or when the LSE is a
        # constant (FA4 without an LSE gradient).
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = torch.ops.usp_attention.flash4_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            dlse if _FA4_BWD_TAKES_DLSE else None,
            ctx.aux_tensors,
            bwd_full_block_cnt,
            bwd_full_block_idx,
            bwd_mask_block_cnt,
            bwd_mask_block_idx,
            ctx.bwd_block_size[0],
            ctx.bwd_block_size[1],
            ctx.softmax_scale,
            ctx.sliding_window,
            ctx.deterministic,
        )
        # Gradients for (q, k, v), then the aux_tensors slot, then None for the
        # remaining 15 inputs. aux_tensors is a Tensor[] input, so its gradient
        # must be a list of the same length (all None) — register_autograd
        # validates that the returned structure matches the input tree, and a
        # flat None there fails the TreeSpec check.
        return (dq, dk, dv, [None] * len(ctx.aux_tensors), *((None,) * 15))

    torch.library.register_autograd(
        "usp_attention::flash4_attn",
        _flash4_attn_op_backward,
        setup_context=_flash4_attn_op_setup_context,
    )


def _flash4_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    aux_tensors: list[torch.Tensor] | None = None,
    full_block_cnt: torch.Tensor | None = None,
    full_block_idx: torch.Tensor | None = None,
    mask_block_cnt: torch.Tensor | None = None,
    mask_block_idx: torch.Tensor | None = None,
    block_size: tuple[int, int] | None = None,
    bwd_block_sparse: "BlockSparseTensorsTorch" | None = None,
    sliding_window: int | None = None,
    deterministic: bool = False,
):
    if not HAS_FLASH4:
        raise RuntimeError(
            "flash_attn.cute is not installed! Cannot perform flash4 attention."
        )
    # Route through the dispatcher-visible custom op so a SAC policy can see and
    # MUST_SAVE the attention output. Flatten the block-sparse tensors + sizes to
    # the op's tensor/int schema. The FA4 mask path always uses fixed kernel
    # settings (causal=False, no window/softcap/sink, num_splits=1, pack_gqa=None,
    # mask_mod=SPARSE_MASK_MOD), which the op applies internally — so this wrapper
    # intentionally does not accept those knobs; the mask is carried entirely by
    # aux_tensors + the block-sparse metadata.
    block_size_m, block_size_n = block_size if block_size is not None else (0, 0)
    if bwd_block_sparse is not None:
        bwd_full_block_cnt = bwd_block_sparse.full_block_cnt
        bwd_full_block_idx = bwd_block_sparse.full_block_idx
        bwd_mask_block_cnt = bwd_block_sparse.mask_block_cnt
        bwd_mask_block_idx = bwd_block_sparse.mask_block_idx
        bwd_block_size_m, bwd_block_size_n = bwd_block_sparse.block_size
    else:
        bwd_full_block_cnt = None
        bwd_full_block_idx = None
        bwd_mask_block_cnt = None
        bwd_mask_block_idx = None
        bwd_block_size_m, bwd_block_size_n = 0, 0
    return torch.ops.usp_attention.flash4_attn(
        q,
        k,
        v,
        aux_tensors if aux_tensors is not None else [],
        full_block_cnt,
        full_block_idx,
        mask_block_cnt,
        mask_block_idx,
        block_size_m,
        block_size_n,
        bwd_full_block_cnt,
        bwd_full_block_idx,
        bwd_mask_block_cnt,
        bwd_mask_block_idx,
        bwd_block_size_m,
        bwd_block_size_n,
        softmax_scale,
        sliding_window or 0,
        deterministic,
    )


def flash4_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: FA4SparseMask,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    deterministic: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Run Flash Attention 4 with a precomputed sparse mask.

    This bypasses torch.compile / flex_attention and calls ``_flash_attn_fwd``
    from ``flash_attn.cute.interface`` directly with a hand-written CuTe DSL
    ``mask_mod``.

    Args:

        q: (B, S, H_q, D) bfloat16 or float16 query tensor.
        k: (B, S, H_kv, D) key tensor.
        v: (B, S, H_kv, D) value tensor.
        mask: precomputed ``FA4SparseMask`` from ``build_fa4_sparse_mask``.
        softmax_scale: optional; defaults to ``1 / sqrt(head_dim)``.
        return_lse: optional; defaults to False.
        deterministic: Use FA4's deterministic backward. FA4 raises in the
            backward when it cannot run it deterministically for this mask
            (for example on FA4 releases whose deterministic block-sparse
            backward needs dQ write-order metadata, which
            ``build_fa4_sparse_mask`` does not build).

    Returns:

        out: (B, S, H_q, D) attention output, and the (B, H_q, S) fp32 LSE
        when ``return_lse`` is set. The LSE is differentiable when the
        installed FA4 backward takes its gradient (``dlse``); otherwise it is
        returned as a constant (``requires_grad=False``), so a loss through it
        contributes no gradient.

    Note:

        FA4 expects Q/K/V in **(B, Seq, Heads, Dim)** layout, NOT the
        ``(B, Heads, Seq, Dim)`` layout used by torch's ``flex_attention``.
        Transpose before calling if your tensors are in BHSD format.
    """
    if not HAS_FLASH4:
        raise RuntimeError(
            "flash_attn.cute is not installed! Cannot perform flash4 attention."
        )
    if mask.block_sparse is None:
        raise ValueError(
            "flash4_attn requires FA4 block sparsity tensors. Build the mask with "
            "`build_block_sparse=True` before calling flash4_attn directly."
        )
    out, _lse = _flash4_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        full_block_cnt=mask.block_sparse.full_block_cnt,
        full_block_idx=mask.block_sparse.full_block_idx,
        mask_block_cnt=mask.block_sparse.mask_block_cnt,
        mask_block_idx=mask.block_sparse.mask_block_idx,
        block_size=mask.block_sparse.block_size,
        aux_tensors=mask.aux_tensors,
        bwd_block_sparse=mask.block_sparse_bwd,
        sliding_window=mask.sliding_window,
        deterministic=deterministic,
    )
    if return_lse:
        return out, _lse
    else:
        return out


def _window_size(
    sliding_window: int | None, *, fa3: bool = False
) -> tuple[int | None, int | None]:
    """``window_size`` for a causal FA call: the last ``sliding_window`` keys."""
    if not sliding_window:
        return (-1, -1) if fa3 else (None, None)
    return (sliding_window - 1, 0)


def _fa3_flash_attn_func(device: torch.device, varlen: bool = False) -> Callable | None:
    """FlashAttention-3's (varlen) ``flash_attn_func`` on an SM90 device, else ``None``."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9:
        return None
    try:
        # [MLR-02.08] flash-attn-3 is an optional, Hopper-only dependency.
        import flash_attn_interface
    except ImportError:
        return None
    return (
        flash_attn_interface.flash_attn_varlen_func
        if varlen
        else flash_attn_interface.flash_attn_func
    )


def _flash_attn_varlen_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool,
    softmax_scale: float | None,
    deterministic: bool,
    return_lse: bool,
    use_fa3_on_hopper: bool,
    sliding_window: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Varlen attention over the documents of (B, S, H, D) inputs."""
    batch, seq = q.shape[:2]
    if batch > 1:
        offsets = (
            torch.arange(batch, device=cu_seqlens.device, dtype=cu_seqlens.dtype) * seq
        )
        cu_seqlens = torch.cat(
            [
                (cu_seqlens[:-1] + offsets[:, None]).flatten(),
                cu_seqlens[-1:] + offsets[-1],
            ]
        )
    flat = [t.reshape(batch * seq, *t.shape[2:]) for t in (q, k, v)]
    fa3 = (
        _fa3_flash_attn_func(q.device, varlen=True)
        if use_fa3_on_hopper and causal
        else None
    )
    if fa3 is not None:
        result = fa3(
            *flat,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=_window_size(sliding_window, fa3=True),
            deterministic=deterministic,
            return_attn_probs=return_lse,
        )
    else:
        result = flash_attn_varlen_func(
            *flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=_window_size(sliding_window),
            deterministic=deterministic,
            return_lse=return_lse,
        )
    out, lse = result if isinstance(result, tuple) else (result, None)
    out = out.reshape(batch, seq, *out.shape[1:])
    if not return_lse:
        return out
    # Varlen LSE is (H, B * S); return (B, H, S) like the batched kernels.
    return out, lse.reshape(lse.shape[0], batch, seq).transpose(0, 1).contiguous()


def flash4_attn_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
    use_fa3_on_hopper: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run FA4 over one plain causal or full sequence with its native mask.

    Equivalent to :func:`flash4_attn` with a single causal or full split, but
    without block-sparse metadata or a ``mask_mod``, which FA4 evaluates on
    every block and which costs a few percent on long sequences.

    Args:
        q: (B, S, H_q, D) query tensor.
        k: (B, S, H_kv, D) key tensor.
        v: (B, S, H_kv, D) value tensor.
        causal: Causal (``True``) or full (``False``) attention.
        softmax_scale: Defaults to ``1 / sqrt(head_dim)``.
        deterministic: Use the deterministic FA4 backward.
        return_lse: Also return the log-sum-exp.
        use_fa3_on_hopper: On SM90, run a causal document with
            FlashAttention-3 instead when it is installed
            (``flash_attn_interface``); its causal kernel is 1-3% faster than
            FA4's on H100, while FA4 is faster on full masks.
        cu_seqlens: Document boundaries (int32 ``[0, ..., S]``) to run varlen
            attention over several documents of each sequence instead.
        max_seqlen: Longest document; required with ``cu_seqlens``.
        sliding_window: With ``causal``, attend only to the last
            ``sliding_window`` keys (FA4's local attention).

    Returns:
        The (B, S, H_q, D) output, and the LSE when ``return_lse`` is set.

    Raises:
        ValueError: If ``sliding_window`` is set without ``causal``, or
            ``cu_seqlens`` without ``max_seqlen``.
    """
    if sliding_window and not causal:
        raise ValueError("sliding_window requires causal attention.")
    if cu_seqlens is not None:
        if max_seqlen is None:
            raise ValueError("max_seqlen is required with cu_seqlens.")
        return _flash_attn_varlen_causal(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            causal=causal,
            softmax_scale=softmax_scale,
            deterministic=deterministic,
            return_lse=return_lse,
            use_fa3_on_hopper=use_fa3_on_hopper,
            sliding_window=sliding_window,
        )
    # FA3 only wins on causal masks; FA4's full-mask kernel is faster on H100.
    if (
        use_fa3_on_hopper
        and causal
        and (fa3 := _fa3_flash_attn_func(q.device)) is not None
    ):
        result = fa3(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=_window_size(sliding_window, fa3=True),
            deterministic=deterministic,
            return_attn_probs=return_lse,
        )
        return (
            result
            if return_lse
            else (result[0] if isinstance(result, tuple) else result)
        )
    if not HAS_FLASH4:
        raise RuntimeError(
            "flash_attn.cute is not installed! Cannot perform flash4 attention."
        )
    out, lse = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=_window_size(sliding_window),
        deterministic=deterministic,
        return_lse=return_lse,
    )
    return (out, lse) if return_lse else out
