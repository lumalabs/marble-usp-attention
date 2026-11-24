# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import contextlib
import functools
import itertools
import operator
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask

HAS_FLASH4_CUTE = False
try:
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    from usp_attention.online_flash import online_flash_attn_func

    HAS_FLASH4_CUTE = True
except ImportError:
    HAS_FLASH4_CUTE = False

# ---------------------------------------------------------------------------
# Ring infrastructure that lives in other modules of this package,
# re-exported here for backward compatibility.
# ---------------------------------------------------------------------------
from usp_attention.fa4.flash4_block_sparse import (  # noqa: E402
    coarsen_q_blocks,
    sparse_kv_block_factor,
    sparse_q_block_factor,
)
from usp_attention.ring.fp32_grad import (  # noqa: E402
    Fp32GradState,
    convert_dkv,
    fp32_grad_step,
)
from usp_attention.ring.ring_combine import (  # noqa: E402
    _all_gather_cat,
    _ensure_lse_bwd_layout,
    _LocalRingResultMerger,
    _LocalRingResultMergerFlash,
    _normalize_step_lse,
    _split_local_shard,
)
from usp_attention.ring.ring_pipeline import (  # noqa: E402
    PendingGradExchange,
    PendingKVExchange,
    _launch_async_grad_rotation,
    _launch_async_kv_rotation,
    _wait_pending_grad_exchange,
    _wait_pending_kv_exchange,
    run_async_ring_kv_pipeline,
)
from usp_attention.ring.ring_planning import (  # noqa: E402
    CREATE_BLOCK_MASK_DYNAMO_CONFIG,
    CreateBlockMaskFn,
    RankRingPlan,
    RingStepPlan,
    _python_cross_qkv_mask_from_aux,
    apply_token_permutation,
    build_cross_qkv_step_aux_tensors,
    build_ring_plan,
)
from usp_attention.ring.ring_planning import (
    create_layout_block_mask as create_layout_block_mask,
)
from usp_attention.ring.ring_planning import (
    get_rank_plan as get_rank_plan,
)
from usp_attention.ring.ring_planning import (
    inverse_permutation as inverse_permutation,
)
from usp_attention.sliding_window import windowed_cross_blocks  # noqa: E402
from usp_attention.types import RingCombineMode

# Separate compiled create_block_mask for cross-qkv mask building
# (independent instance from the one in ring_planning).
_CREATE_BLOCK_MASK: CreateBlockMaskFn | None = None

# ---------------------------------------------------------------------------
# Flash4-specific code: CuTe helpers, cross-qkv masking, ring step kernels.
# ---------------------------------------------------------------------------

if HAS_FLASH4_CUTE:

    @cute.jit
    def _ssa_to_idx(ssa_value: cute.TensorSSA, dtype: str) -> cute.Numeric:
        frag = cute.make_rmem_tensor(1, dtype)
        frag.store(ssa_value)
        return frag[0]

    @cute.jit
    def _load_f32(tensor, idx_ssa: cute.TensorSSA) -> cute.TensorSSA:
        idx = _ssa_to_idx(idx_ssa, cutlass.Int32)
        buf = cute.make_rmem_tensor(1, cutlass.Float32)
        buf[0] = tensor[idx]
        return buf.load()

    @cute.jit
    def _load_i64(tensor, idx_ssa: cute.TensorSSA) -> cute.TensorSSA:
        idx = _ssa_to_idx(idx_ssa, cutlass.Int32)
        buf = cute.make_rmem_tensor(1, cutlass.Int64)
        buf[0] = tensor[idx]
        return buf.load()

    @cute.jit
    def _load_i32(tensor, idx_ssa: cute.TensorSSA) -> cute.TensorSSA:
        idx = _ssa_to_idx(idx_ssa, cutlass.Int32)
        buf = cute.make_rmem_tensor(1, cutlass.Int32)
        buf[0] = tensor[idx]
        return buf.load()


def _cross_qkv_rows(q, aux):
    return aux[6][q], aux[0][q], aux[2][q], aux[4][q]


def _cross_qkv_cols(kv, aux):
    return aux[7][kv], aux[1][kv], aux[3][kv], aux[5][kv]


def cross_qkv_separable(window: int | None = None):
    """The position-aware predicate as ``(row_fn, col_fn, combine)``.

    Per query ``(pos, fn, noise, doc)`` and per key the same four values; the
    form :mod:`~usp_attention.fa4.separable_mask_sm90` evaluates with one load
    per tile row and column. Plain Python over the operands, so it also runs
    on torch tensors (tests compare it with the dense mask).
    """

    def combine(row, col):
        q_pos, q_fn, q_noise, q_doc = row
        kv_pos, kv_fn, kv_noise, kv_doc = col
        causal = q_pos >= kv_pos
        if window is not None:
            causal = causal & (q_pos - kv_pos < window)
        full_noise = (q_fn == kv_fn) & (q_fn >= 0.0)
        # A noise key is visible only inside its own noise span.
        noise_ok = (kv_noise < 0.0) | (q_noise == kv_noise)
        return (causal | full_noise) & noise_ok & (q_doc == kv_doc)

    return _cross_qkv_rows, _cross_qkv_cols, combine


def _column_range_rows(q, aux):
    return (q,)


def _column_range_cols(kv, aux):
    return aux[0][kv], aux[1][kv]


def _column_range_combine(row, col):
    return (row[0] >= col[0]) & (row[0] < col[1])


COLUMN_RANGE_SEPARABLE = (_column_range_rows, _column_range_cols, _column_range_combine)


def _position_range_rows(q, aux):
    return (aux[0][q],)


def _position_range_cols(kv, aux):
    return aux[1][kv], aux[2][kv]


POSITION_RANGE_SEPARABLE = (
    _position_range_rows,
    _position_range_cols,
    _column_range_combine,
)


def _make_cross_qkv_sparse_mask_mod(window: int | None = None):
    """Create a CuTe mask_mod over separate q/kv metadata tensors.

    With a sliding ``window`` the causal term also requires
    ``q_pos - kv_pos < window`` (see :mod:`usp_attention.sliding_window`).
    """
    if not HAS_FLASH4_CUTE:
        return None
    from usp_attention.fa4 import separable_mask_sm90

    @cute.jit
    def mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del b_idx, h_idx, seqlen_info
        q_fn = aux_tensors[0]
        kv_fn = aux_tensors[1]
        q_noise = aux_tensors[2]
        kv_noise = aux_tensors[3]
        q_doc = aux_tensors[4]
        kv_doc = aux_tensors[5]
        q_pos = aux_tensors[6]
        kv_pos = aux_tensors[7]

        q_abs = _load_i64(q_pos, q_idx)
        kv_abs = _load_i64(kv_pos, kv_idx)
        causal = operator.ge(q_abs, kv_abs)
        if cutlass.const_expr(window is not None):
            dist = operator.sub(q_abs, kv_abs)
            causal = causal & operator.lt(dist, cute.full_like(dist, window))

        fn_q = _load_f32(q_fn, q_idx)
        fn_kv = _load_f32(kv_fn, kv_idx)
        fn_same = operator.eq(fn_q, fn_kv)
        fn_valid = operator.ge(fn_q, cute.full_like(fn_q, 0))
        full_noise = fn_same & fn_valid

        noise_kv = _load_f32(kv_noise, kv_idx)
        noise_kv_valid = operator.ge(noise_kv, cute.full_like(noise_kv, 0))
        noise_q = _load_f32(q_noise, q_idx)
        noise_diff = operator.ne(noise_q, noise_kv)

        doc_q = _load_i64(q_doc, q_idx)
        doc_kv = _load_i64(kv_doc, kv_idx)
        same_doc = operator.eq(doc_q, doc_kv)

        output = ((causal | full_noise) & (noise_kv_valid & noise_diff == 0)) & same_doc
        return output

    if separable_mask_sm90.install():
        separable_mask_sm90.mark(
            mask_mod, *cross_qkv_separable(window), rows_hoisted_only=True
        )
    return mask_mod


CROSS_QKV_SPARSE_MASK_MOD = _make_cross_qkv_sparse_mask_mod()


def _make_column_range_mask_mod():
    """``lo[kv] <= q < hi[kv]``: each key's visible queries as one range.

    For a single attention call over a layout stored in vision-group order
    (:func:`~usp_attention.fa4.column_ranges.column_query_ranges`); two int32
    loads per element instead of the eight of the position-aware predicate,
    and with :mod:`~usp_attention.fa4.separable_mask_sm90` on Hopper two loads
    per key column of a tile.
    """
    if not HAS_FLASH4_CUTE:
        return None
    from usp_attention.fa4 import separable_mask_sm90

    @cute.jit
    def mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del b_idx, h_idx, seqlen_info
        lo = _load_i32(aux_tensors[0], kv_idx)
        hi = _load_i32(aux_tensors[1], kv_idx)
        return operator.ge(q_idx, lo) & operator.lt(q_idx, hi)

    if separable_mask_sm90.install():
        separable_mask_sm90.mark(mask_mod, *COLUMN_RANGE_SEPARABLE)
    return mask_mod


COLUMN_RANGE_MASK_MOD = _make_column_range_mask_mod()


def _make_position_range_mask_mod():
    """``lo[kv] <= pos[q] < hi[kv]`` for one ring step: aux ``[q_pos, kv_lo, kv_hi]``.

    ``pos`` is each token's index in an order in which every key's visible
    queries are one range (:func:`~usp_attention.fa4.column_ranges.column_query_ranges`),
    so the test holds in any token order: three int32 loads and two compares
    per element instead of the position-aware predicate's eight loads (see
    :func:`use_position_ranges`).
    """
    if not HAS_FLASH4_CUTE:
        return None
    from usp_attention.fa4 import separable_mask_sm90

    @cute.jit
    def mask_mod(b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        del b_idx, h_idx, seqlen_info
        pos = _load_i32(aux_tensors[0], q_idx)
        lo = _load_i32(aux_tensors[1], kv_idx)
        hi = _load_i32(aux_tensors[2], kv_idx)
        return operator.ge(pos, lo) & operator.lt(pos, hi)

    if separable_mask_sm90.install():
        separable_mask_sm90.mark(mask_mod, *POSITION_RANGE_SEPARABLE)
    return mask_mod


POSITION_RANGE_MASK_MOD = _make_position_range_mask_mod()


def use_position_ranges(
    plan: "PrecomputedCrossQKVAuxPlan", positions: Tensor, lo: Tensor, hi: Tensor
) -> None:
    """Run every step of ``plan`` with :data:`POSITION_RANGE_MASK_MOD`'s aux tensors.

    ``positions``, ``lo`` and ``hi`` are int32 per token in the plan's global
    token order. Block sparsity is already classified from the position-aware
    aux tensors; only the tensors the kernel's ``mask_mod`` reads change.
    """
    q = positions[plan.q_start : plan.q_start + plan.q_len].contiguous()
    for step in plan.steps:
        kv = slice(step.kv_start, step.kv_start + step.kv_len)
        step.aux_tensors = [q, lo[kv].contiguous(), hi[kv].contiguous()]


@functools.cache
def cross_qkv_sparse_mask_mod(window: int | None = None):
    """The ring-step ``mask_mod``, one cached function per sliding window."""
    if not window:
        return CROSS_QKV_SPARSE_MASK_MOD
    return _make_cross_qkv_sparse_mask_mod(window)


@dataclass(kw_only=True, slots=True)
class PrecomputedStepAux:
    """Per-ring-step precomputed aux tensors for cross q/kv masking.

    Forward block sparse fields (KV-indexed) tell the forward kernel which
    KV blocks to visit per Q block.  Backward block sparse fields (Q-indexed)
    tell the backward kernel which Q blocks contributed per KV block.
    """

    step: int
    kv_owner_rank: int
    kv_start: int
    kv_len: int
    aux_tensors: list[Tensor]
    block_size: tuple[int, int] | None = None
    # Forward (KV-indexed)
    full_block_cnt: Tensor | None = None
    full_block_idx: Tensor | None = None
    mask_block_cnt: Tensor | None = None
    mask_block_idx: Tensor | None = None
    # Backward (Q-indexed, transposed from forward)
    bwd_full_block_cnt: Tensor | None = None
    bwd_full_block_idx: Tensor | None = None
    bwd_mask_block_cnt: Tensor | None = None
    bwd_mask_block_idx: Tensor | None = None


@dataclass(kw_only=True, slots=True)
class PrecomputedCrossQKVAuxPlan:
    """Precomputed per-step aux tensors for one rank."""

    rank: int
    world_size: int
    q_start: int
    q_len: int
    steps: list[PrecomputedStepAux]

    def step_data(self, step: int) -> PrecomputedStepAux:
        """Return full per-step precomputed data.

        Raises:
            ValueError: If step is out of range or precomputed step ordering is
                inconsistent.
        """
        if step < 0 or step >= len(self.steps):
            raise ValueError(
                f"Step out of range for precomputed aux plan: {step=}, "
                f"{len(self.steps)=}."
            )
        step_data = self.steps[step]
        if step_data.step != step:
            raise ValueError(
                f"Precomputed step ordering mismatch: requested {step=}, "
                f"found {step_data.step=}."
            )
        return step_data

    def aux_for_step(self, step: int) -> list[Tensor]:
        """Return aux tensors for a specific ring step."""
        return self.step_data(step).aux_tensors


@dataclass(kw_only=True, slots=True)
class NativeRingMask:
    """A ring over one plain document, run with FA4's native masks.

    ``"full"``: every step is full attention. ``"causal_striped"``: the
    sequence is striped over the ring (token ``i`` lives on ring rank
    ``i % ring_size``, in order), so a step whose KV comes from a rank at or
    below this one is causal on local indices, and one whose KV comes from a
    rank above is strictly causal (key index < query index). Every step runs
    one FA4 call on the full local Q and KV, without block sparsity, and every
    rank does the same work at every step.

    Attributes:
        mode: ``"full"`` or ``"causal_striped"``.
        ring_rank: This rank's position in the ring.
        ring_size: Number of ring ranks.
    """

    mode: Literal["full", "causal_striped"]
    ring_rank: int
    ring_size: int

    def kv_owner(self, step: int) -> int:
        """Ring rank whose KV this rank holds at ``step``."""
        return (self.ring_rank - step) % self.ring_size

    def step_mask(self, step: int) -> tuple[bool, int | None]:
        """``(causal, window_size_right)`` for FA4 at ``step``."""
        if self.mode == "full":
            return False, None
        if self.kv_owner(step) <= self.ring_rank:
            return True, None
        # Strictly causal: FA4's local window with right = -1 (no left bound).
        return False, -1


@dataclass(kw_only=True, slots=True)
class Flash4RingStepMeta:
    """All metadata needed by the internal ring FA4 step kernels.

    Packages the CuTe mask_mod, precomputed per-step aux/sparsity data,
    and attention hyperparameters so that ring_flash4_attn can call
    _flash_attn_fwd / _flash_attn_bwd directly without an external
    step-kernel callback.

    Attributes:
        mask_mod: CuTe DSL mask_mod function (e.g. CROSS_QKV_SPARSE_MASK_MOD);
            ``None`` with ``native``.
        precomputed_aux_plan: Per-step aux tensors and block sparse data;
            ``None`` with ``native``.
        softmax_scale: Attention softmax scale, None for 1/sqrt(d).
        native: Run every step with FA4's native masks instead of block
            sparsity (one plain document; see :class:`NativeRingMask`).
    """

    mask_mod: Callable | None = None
    precomputed_aux_plan: PrecomputedCrossQKVAuxPlan | None = None
    softmax_scale: float | None = None
    native: NativeRingMask | None = None

    def __post_init__(self) -> None:
        if self.native is None and (
            self.mask_mod is None or self.precomputed_aux_plan is None
        ):
            raise ValueError(
                "Flash4RingStepMeta needs mask_mod and precomputed_aux_plan "
                "unless native is set."
            )


@dataclass(kw_only=True, slots=True)
class _StepBlockSparse:
    """Forward and backward block sparse tensors extracted from a BlockMask."""

    block_size: tuple[int, int]
    # Forward (KV-indexed)
    full_block_cnt: Tensor | None
    full_block_idx: Tensor | None
    mask_block_cnt: Tensor | None
    mask_block_idx: Tensor | None
    # Backward (Q-indexed)
    bwd_full_block_cnt: Tensor | None
    bwd_full_block_idx: Tensor | None
    bwd_mask_block_cnt: Tensor | None
    bwd_mask_block_idx: Tensor | None
    # KV blocks of the step when the index lists are compact (shorter rows).
    num_kv_blocks: int | None = None


def _coarsen_step_block_sparse(
    sparse: _StepBlockSparse, factor: int, kv_factor: int = 1
) -> _StepBlockSparse:
    """Merge Q (and KV) blocks of a step's block sparsity (SM100-class FA4)."""
    assert sparse.mask_block_idx is not None and sparse.full_block_cnt is not None
    fc, fi, mc, mi, bfc, bfi, bmc, bmi = coarsen_q_blocks(
        sparse.full_block_cnt,
        sparse.full_block_idx,
        sparse.mask_block_cnt,
        sparse.mask_block_idx,
        num_kv_blocks=sparse.num_kv_blocks or sparse.mask_block_idx.shape[-1],
        factor=factor,
        kv_factor=kv_factor,
    )
    return _StepBlockSparse(
        block_size=(sparse.block_size[0] * factor, sparse.block_size[1] * kv_factor),
        full_block_cnt=fc,
        full_block_idx=fi,
        mask_block_cnt=mc,
        mask_block_idx=mi,
        bwd_full_block_cnt=bfc,
        bwd_full_block_idx=bfi,
        bwd_mask_block_cnt=bmc,
        bwd_mask_block_idx=bmi,
    )


def build_step_block_sparse_tensors_from_aux(
    aux_tensors: list[Tensor],
    sliding_window: int | None = None,
) -> _StepBlockSparse:
    """Build per-step FA4 block-sparse tensors from cross-qkv aux slices."""
    global _CREATE_BLOCK_MASK
    q_len = int(aux_tensors[0].numel())
    kv_len = int(aux_tensors[1].numel())
    device = aux_tensors[0].device
    with torch._dynamo.config.patch(**CREATE_BLOCK_MASK_DYNAMO_CONFIG):
        if _CREATE_BLOCK_MASK is None:
            _CREATE_BLOCK_MASK = torch.compile(
                create_block_mask, dynamic=False, fullgraph=True
            )
        block_mask = _CREATE_BLOCK_MASK(
            _python_cross_qkv_mask_from_aux(aux_tensors, sliding_window=sliding_window),
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
    bwd_full_block_cnt = None
    bwd_full_block_idx = None
    bwd_mask_block_cnt = None
    bwd_mask_block_idx = None
    if block_mask.q_num_blocks is not None and block_mask.q_indices is not None:
        bwd_mask_block_cnt = block_mask.q_num_blocks
        bwd_mask_block_idx = block_mask.q_indices
        bwd_full_block_cnt = block_mask.full_q_num_blocks
        bwd_full_block_idx = block_mask.full_q_indices
    return _StepBlockSparse(
        block_size=tuple(block_mask.BLOCK_SIZE),
        full_block_cnt=block_mask.full_kv_num_blocks,
        full_block_idx=block_mask.full_kv_indices,
        mask_block_cnt=block_mask.kv_num_blocks,
        mask_block_idx=block_mask.kv_indices,
        bwd_full_block_cnt=bwd_full_block_cnt,
        bwd_full_block_idx=bwd_full_block_idx,
        bwd_mask_block_cnt=bwd_mask_block_cnt,
        bwd_mask_block_idx=bwd_mask_block_idx,
    )


def _use_cutedsl_block_sparse() -> bool:
    """Opt-in gate for the CuTe DSL block-sparse implementation.

    The CuTe DSL path (~161us vs ~1.4-1.8ms eager on H100) is bit-exact with
    the torch op on production inputs (integral fn/noise split ids; see
    ``block_sparse_equivalence_test.py``) but is opt-in while it soaks.
    Requires sm_90+ even when the flag is set: the kernels are developed and
    validated on H100s (all of production), and CuTe DSL cannot compile for
    older archs like the sm_75 T4 CI runners.
    """
    if os.environ.get("FLASH4_BLOCK_SPARSE_CUTEDSL", "0").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def build_step_block_sparse_direct(
    aux_tensors: list[Tensor],
    block_size: int = 128,
    *,
    sliding_window: int | None = None,
    kv_segment_end: Tensor | None = None,
) -> _StepBlockSparse:
    """Build per-step FA4 block-sparse tensors using direct classification.

    Avoids materializing a dense Q*KV mask by classifying block sparsity
    directly from aux tensor statistics at block granularity. Set
    ``FLASH4_BLOCK_SPARSE_CUTEDSL=1`` to use the CuTe DSL kernels instead of
    the torch implementation.
    """
    if _use_cutedsl_block_sparse():
        from usp_attention.ring.block_sparse_cutedsl import (
            build_block_sparse_from_cross_qkv_aux_cutedsl as _build_block_sparse,
        )
    else:
        from usp_attention.ring.block_sparse import (
            build_block_sparse_from_cross_qkv_aux as _build_block_sparse,
        )

    if _use_cutedsl_block_sparse():
        result = _build_block_sparse(aux_tensors, block_size=block_size)
    else:
        # Candidate pairs only, compact lists; the window needs the type grid.
        result = _build_block_sparse(
            aux_tensors,
            block_size=block_size,
            compact=True,
            with_types=bool(sliding_window),
        )
    if sliding_window:
        if kv_segment_end is None:
            raise ValueError("kv_segment_end is required with a sliding window.")
        result = windowed_cross_blocks(
            result, aux_tensors, kv_segment_end, sliding_window
        )
    return _StepBlockSparse(
        block_size=result.block_size,
        full_block_cnt=result.full_kv_num_blocks,
        full_block_idx=result.full_kv_indices,
        mask_block_cnt=result.kv_num_blocks,
        mask_block_idx=result.kv_indices,
        bwd_full_block_cnt=result.full_q_num_blocks,
        bwd_full_block_idx=result.full_q_indices,
        bwd_mask_block_cnt=result.q_num_blocks,
        bwd_mask_block_idx=result.q_indices,
        num_kv_blocks=result.n_kv_blocks,
    )


def build_precomputed_cross_qkv_aux_plan(
    *,
    global_aux_tensors: list[Tensor],
    rank: int,
    world_size: int,
    local_len: int,
    q_start: int | None = None,
    precompute_block_sparse: bool = True,
    use_direct_block_sparse: bool = True,
    global_position_ids: Tensor | None = None,
    sliding_window: int | None = None,
    global_segment_end: Tensor | None = None,
) -> PrecomputedCrossQKVAuxPlan:
    """Precompute all per-step aux tensors for a rank's ring execution.

    With ``sliding_window``, ``global_segment_end`` (from
    :func:`usp_attention.sliding_window.segment_end_positions`, in the same
    token order as ``global_aux_tensors``) lets the classifier skip blocks the
    window empties.

    Raises:
        ValueError: If world_size, rank, local_len, or q_start are invalid.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size=}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank out of range: {rank=}, {world_size=}.")
    if local_len <= 0:
        raise ValueError(f"local_len must be positive, got {local_len=}.")
    q_start = rank * local_len if q_start is None else q_start
    if q_start < 0:
        raise ValueError(f"q_start must be >= 0, got {q_start=}.")

    def build_step(step: int) -> PrecomputedStepAux:
        kv_owner_rank = (rank - step) % world_size
        kv_start = kv_owner_rank * local_len
        aux_tensors = build_cross_qkv_step_aux_tensors(
            global_aux_tensors=global_aux_tensors,
            q_start=q_start,
            q_len=local_len,
            kv_start=kv_start,
            kv_len=local_len,
            global_position_ids=global_position_ids,
        )
        sparse: _StepBlockSparse | None = None
        if precompute_block_sparse:
            if use_direct_block_sparse:
                sparse = build_step_block_sparse_direct(
                    aux_tensors,
                    sliding_window=sliding_window,
                    kv_segment_end=(
                        global_segment_end[kv_start : kv_start + local_len]
                        if sliding_window
                        else None
                    ),
                )
            else:
                sparse = build_step_block_sparse_tensors_from_aux(
                    aux_tensors, sliding_window=sliding_window
                )
            factor = sparse_q_block_factor(aux_tensors[0].device)
            kv_factor = sparse_kv_block_factor(aux_tensors[0].device)
            if factor > 1 or kv_factor > 1:
                sparse = _coarsen_step_block_sparse(sparse, factor, kv_factor)
        return PrecomputedStepAux(
            step=step,
            kv_owner_rank=kv_owner_rank,
            kv_start=kv_start,
            kv_len=local_len,
            aux_tensors=aux_tensors,
            block_size=sparse.block_size if sparse else None,
            full_block_cnt=sparse.full_block_cnt if sparse else None,
            full_block_idx=sparse.full_block_idx if sparse else None,
            mask_block_cnt=sparse.mask_block_cnt if sparse else None,
            mask_block_idx=sparse.mask_block_idx if sparse else None,
            bwd_full_block_cnt=sparse.bwd_full_block_cnt if sparse else None,
            bwd_full_block_idx=sparse.bwd_full_block_idx if sparse else None,
            bwd_mask_block_cnt=sparse.bwd_mask_block_cnt if sparse else None,
            bwd_mask_block_idx=sparse.bwd_mask_block_idx if sparse else None,
        )

    if global_aux_tensors[0].device.type == "cpu" and world_size > 1:
        # CPU planning: steps are independent and torch releases the GIL, so
        # small per-step ops run side by side instead of one after another.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(world_size, 8)) as pool:
            steps = list(pool.map(build_step, range(world_size)))
    else:
        steps = [build_step(step) for step in range(world_size)]

    return PrecomputedCrossQKVAuxPlan(
        rank=rank,
        world_size=world_size,
        q_start=q_start,
        q_len=local_len,
        steps=steps,
    )


def _step_fwd_block_sparse(
    step_data: PrecomputedStepAux,
) -> "BlockSparseTensorsTorch | None":
    """Build forward BlockSparseTensorsTorch from precomputed step data."""
    if not HAS_FLASH4_CUTE:
        return None
    if step_data.mask_block_cnt is None or step_data.mask_block_idx is None:
        return None
    return BlockSparseTensorsTorch(
        full_block_cnt=step_data.full_block_cnt,
        full_block_idx=step_data.full_block_idx,
        mask_block_cnt=step_data.mask_block_cnt,
        mask_block_idx=step_data.mask_block_idx,
        block_size=step_data.block_size,
    )


def _step_bwd_block_sparse(
    step_data: PrecomputedStepAux,
) -> "BlockSparseTensorsTorch | None":
    """Build backward BlockSparseTensorsTorch from precomputed step data."""
    if not HAS_FLASH4_CUTE:
        return None
    if step_data.bwd_mask_block_cnt is None or step_data.bwd_mask_block_idx is None:
        return None
    return BlockSparseTensorsTorch(
        full_block_cnt=step_data.bwd_full_block_cnt,
        full_block_idx=step_data.bwd_full_block_idx,
        mask_block_cnt=step_data.bwd_mask_block_cnt,
        mask_block_idx=step_data.bwd_mask_block_idx,
        block_size=step_data.block_size,
    )


class _OnlineRingFwdState:
    """Mutable container threading online-softmax state across ring steps."""

    __slots__ = ("out", "max_", "sum_", "lse")

    def __init__(self) -> None:
        self.out: Tensor | None = None
        self.max_: Tensor | None = None
        self.sum_: Tensor | None = None
        self.lse: Tensor | None = None


def _ring_step_trace_name(
    prefix: str,
    step: int,
    *,
    step_meta: Flash4RingStepMeta,
    step_plan: RingStepPlan | None = None,
) -> str:
    """Build a compact profiler range name for a ring attention step."""
    if step_meta.native is not None:
        causal, right = step_meta.native.step_mask(step)
        mask = "causal" if causal else ("strict_causal" if right == -1 else "full")
        return "/".join(
            [
                prefix,
                f"step={step}",
                f"kv_owner={step_meta.native.kv_owner(step)}",
                f"native={mask}",
            ]
        )
    assert step_meta.precomputed_aux_plan is not None
    step_data = step_meta.precomputed_aux_plan.step_data(step)
    parts = [
        prefix,
        f"step={step}",
        f"kv_owner={step_data.kv_owner_rank}",
    ]
    if step_plan is not None:
        parts.append(f"est_blocks={step_plan.estimated_block_intersections}")
        parts.append(f"active_q={int(step_plan.active_local_q_blocks.numel())}")
    return "/".join(parts)


def _ring_transport_trace_name(
    *,
    use_symm_mem: bool,
    use_symm_mem_rotator: bool,
    use_symm_mem_direct: bool,
    use_nccl4py_ring: bool,
) -> str:
    """Return the transport label used in profiler ranges."""
    if use_nccl4py_ring:
        return "nccl4py_rma"
    if use_symm_mem_direct:
        return "symm_direct"
    if use_symm_mem_rotator:
        return "symm_rotator"
    if use_symm_mem:
        return "symm"
    return "nccl_p2p"


class _CaptureSplitPartials:
    """Capture FA4's split-KV fp32 partials instead of combining them to bf16."""

    def __init__(self) -> None:
        self.out: Tensor | None = None
        self.lse: Tensor | None = None
        self._orig = None

    def __enter__(self) -> "_CaptureSplitPartials":
        import flash_attn.cute.interface as fa

        self._orig = fa._flash_attn_fwd_combine

        def capture(out_partial, lse_partial, *args, **kwargs) -> None:
            self.out, self.lse = out_partial, lse_partial

        fa._flash_attn_fwd_combine = capture
        return self

    def __exit__(self, *exc) -> None:
        import flash_attn.cute.interface as fa

        fa._flash_attn_fwd_combine = self._orig


def _fwd_fp32_partials(
    q: Tensor, k: Tensor, v: Tensor, *, num_splits: int = 2, **kwargs
) -> tuple[Tensor, Tensor]:
    """One FA4 forward whose output stays fp32: ``(out (B,S,H,D) fp32, lse (B,H,S))``.

    Runs the split-KV forward, takes the per-split fp32 partials (each
    normalized, with its LSE) and merges them in fp32.
    """
    with _CaptureSplitPartials() as cap:
        _flash_attn_fwd(q, k, v, num_splits=num_splits, return_lse=True, **kwargs)
    if cap.out is None or cap.lse is None:
        raise RuntimeError("FA4 did not take the split-KV path; no fp32 partials.")
    lse_parts = cap.lse  # (splits, B, S, H)
    lse = torch.logsumexp(lse_parts, dim=0)
    weights = torch.exp(lse_parts - lse.unsqueeze(0)).nan_to_num(0.0)
    out = (cap.out * weights.unsqueeze(-1)).sum(dim=0)
    return out, lse.transpose(1, 2).contiguous()


@torch._dynamo.disable()
def _ring_step_fwd(
    q_local: Tensor,
    k_step: Tensor,
    v_step: Tensor,
    step: int,
    step_meta: Flash4RingStepMeta,
    *,
    online_state: _OnlineRingFwdState | None = None,
    total_steps: int = 1,
    fp32_partials: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run one forward ring step.

    With *fp32_partials* (and no *online_state*) the step output is fp32,
    taken from FA4's split-KV partials (see :func:`_fwd_fp32_partials`).

    When *online_state* is provided the step uses ``online_flash_attn_func``
    so that the online-softmax reduction happens inside the kernel.  The
    running ``out / max_ / sum_`` tensors are read from and written back to
    *online_state*; the returned LSE is only populated on the last step.

    When *online_state* is ``None`` the original ``_flash_attn_fwd`` path is
    used (backward-compatible).
    """
    if step_meta.native is not None:
        causal, right = step_meta.native.step_mask(step)
        if online_state is not None:
            online_state.out, lse, online_state.max_, online_state.sum_ = (
                online_flash_attn_func(
                    q_local,
                    k_step,
                    v_step,
                    out=online_state.out,
                    max_=online_state.max_,
                    sum_=online_state.sum_,
                    softmax_scale=step_meta.softmax_scale,
                    causal=causal,
                    window_size=(None, right),
                    is_online_first=step == 0,
                    is_online_last=step == total_steps - 1,
                )
            )
            assert online_state.out is not None
            return online_state.out, lse
        if fp32_partials:
            return _fwd_fp32_partials(
                q_local,
                k_step,
                v_step,
                softmax_scale=step_meta.softmax_scale,
                causal=causal,
                window_size_right=right,
            )
        out, lse, *_ = _flash_attn_fwd(
            q_local,
            k_step,
            v_step,
            softmax_scale=step_meta.softmax_scale,
            causal=causal,
            window_size_right=right,
            return_lse=True,
        )
        return out, lse
    assert step_meta.precomputed_aux_plan is not None
    step_data = step_meta.precomputed_aux_plan.step_data(step)
    if online_state is not None:
        is_first = step == 0
        is_last = step == total_steps - 1
        bsp = _step_fwd_block_sparse(step_data)
        online_state.out, lse, online_state.max_, online_state.sum_ = (
            online_flash_attn_func(
                q_local,
                k_step,
                v_step,
                out=online_state.out,
                max_=online_state.max_,
                sum_=online_state.sum_,
                softmax_scale=step_meta.softmax_scale,
                block_sparse_tensors=bsp,
                is_online_first=is_first,
                is_online_last=is_last,
                mask_mod=step_meta.mask_mod,
                aux_tensors=step_data.aux_tensors,
            )
        )
        assert online_state.out is not None
        return online_state.out, lse  # lse is None on non-last steps
    if fp32_partials:
        return _fwd_fp32_partials(
            q_local,
            k_step,
            v_step,
            softmax_scale=step_meta.softmax_scale,
            mask_mod=step_meta.mask_mod,
            block_sparse_tensors=_step_fwd_block_sparse(step_data),
            aux_tensors=step_data.aux_tensors,
        )
    out, lse, *_ = _flash_attn_fwd(
        q_local,
        k_step,
        v_step,
        softmax_scale=step_meta.softmax_scale,
        mask_mod=step_meta.mask_mod,
        block_sparse_tensors=_step_fwd_block_sparse(step_data),
        aux_tensors=step_data.aux_tensors,
        return_lse=True,
    )
    return out, lse


@torch._dynamo.disable()
def _ring_step_bwd(
    q_local: Tensor,
    k_step: Tensor,
    v_step: Tensor,
    out_local: Tensor,
    dout: Tensor,
    lse_bhs: Tensor,
    step: int,
    step_meta: Flash4RingStepMeta,
    *,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one backward ring step using _flash_attn_bwd directly.

    lse_bhs must be in (B, H, S) layout as expected by _flash_attn_bwd.
    """
    if step_meta.native is not None:
        causal, right = step_meta.native.step_mask(step)
        return _flash_attn_bwd(
            q_local,
            k_step,
            v_step,
            out_local,
            dout,
            lse_bhs,
            step_meta.softmax_scale,
            causal=causal,
            window_size_right=right,
            softcap=0.0,
            deterministic=deterministic,
        )
    assert step_meta.precomputed_aux_plan is not None
    step_data = step_meta.precomputed_aux_plan.step_data(step)
    dq, dk, dv = _flash_attn_bwd(
        q_local,
        k_step,
        v_step,
        out_local,
        dout,
        lse_bhs,
        step_meta.softmax_scale,
        causal=False,
        softcap=0.0,
        deterministic=deterministic,
        mask_mod=step_meta.mask_mod,
        aux_tensors=step_data.aux_tensors,
        block_sparse_tensors=_step_bwd_block_sparse(step_data),
    )
    return dq, dk, dv


def _add_step_dq(
    dq_accum: Tensor | None,
    dq_step: Tensor,
    grad_state: Fp32GradState | None,
    *,
    is_last: bool,
) -> Tensor | None:
    """Add one backward ring step's dq to the running sum and return the sum.

    With fp32 gradients FA4 keeps the running sum in its fp32 dq accumulator
    (:func:`~usp_attention.ring.fp32_grad.fp32_grad_step`): only the last
    step's dq is valid, and it is the total. Where FA4 writes dq directly
    instead (no accumulator to seed, as in its head_dim-256 backward on
    Blackwell), every step's dq is a partial, summed here in fp32 as without
    fp32 gradients.
    """
    if grad_state is not None and grad_state.dq_in_accum:
        return dq_step if is_last else dq_accum
    if dq_accum is None:
        return dq_step.float()
    dq_accum += dq_step.float()
    return dq_accum


def _run_ring_backward(
    *,
    q_local: Tensor,
    k_local: Tensor,
    v_local: Tensor,
    out_local: Tensor,
    dout: Tensor,
    lse_local: Tensor,
    step_meta: Flash4RingStepMeta,
    process_group: dist.ProcessGroup,
    deterministic: bool = False,
    use_cuda_comm_stream: bool = True,
    use_symm_mem: bool = False,
    use_nccl4py_ring: bool = False,
    kv_comm_stream: torch.cuda.Stream | None = None,
    grad_comm_stream: torch.cuda.Stream | None = None,
    fp32_grads: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the backward ring loop.

    KV rotates in the same direction as forward (send to next, recv from prev).
    dk/dv accumulators travel in the same direction as KV: each rank adds its
    local dk/dv contribution for the KV shard it currently holds, then the
    accumulator rotates forward with the KV. After W in-loop steps and one
    final rotation, each accumulator returns to the KV-owning rank with all
    W contributions summed.

    When *use_symm_mem* is True, both KV rotation and dk/dv gradient rotation
    use symmetric memory (zero-copy peer reads via NVLink) instead of NCCL P2P
    ops. KV rotation uses the direct-read symmetric-memory transport, and
    gradient rotation uses :class:`~ring_pipeline_symm.SymmMemGradRing`.

    When *use_nccl4py_ring* is True, both transports use NCCL4Py one-sided
    windows instead of torch symmetric memory. This supports cross-node rings
    on NCCL versions with the one-sided APIs.
    """
    world_size = dist.get_world_size(process_group)
    lse_bhs = _ensure_lse_bwd_layout(lse_local, out_local)

    if world_size == 1:
        dq, dk, dv = _ring_step_bwd(
            q_local,
            k_local,
            v_local,
            out_local,
            dout,
            lse_bhs,
            step=0,
            step_meta=step_meta,
            deterministic=deterministic,
        )
        return dq, dk, dv

    if use_symm_mem and use_nccl4py_ring:
        raise ValueError("use_symm_mem and use_nccl4py_ring are mutually exclusive.")

    # --- KV source setup (symm_mem/nccl4py vs NCCL P2P) ---
    kv_direct = None
    recv_key_bufs = None
    recv_val_bufs = None

    if use_nccl4py_ring:
        from usp_attention.ring.ring_pipeline_nccl4py import (
            _get_or_create_direct,
        )

        kv_direct = _get_or_create_direct(
            k_local.shape,
            k_local.dtype,
            q_local.device,
            process_group,
            kv_comm_stream,
            tag="bwd_direct",
        )
        kv_direct.initial_publish(k_local, v_local)
    elif use_symm_mem:
        from usp_attention.ring.ring_pipeline_symm import (
            _get_or_create_direct,
        )

        kv_direct = _get_or_create_direct(
            k_local.shape,
            k_local.dtype,
            q_local.device,
            process_group,
            kv_comm_stream,
            tag="bwd_direct",
        )
        kv_direct.initial_publish(k_local, v_local)
    else:
        if kv_comm_stream is None:
            kv_comm_stream = (
                torch.cuda.Stream(device=q_local.device)
                if use_cuda_comm_stream and q_local.is_cuda
                else None
            )
        recv_key_bufs = [torch.empty_like(k_local), torch.empty_like(k_local)]
        recv_val_bufs = [torch.empty_like(v_local), torch.empty_like(v_local)]

    # --- Grad rotation setup (symm_mem/nccl4py or NCCL P2P) ---
    # Created on first use: with fp32 gradients and GQA the rotating dk/dv
    # accumulators are FA4's raw fp32 buffers, whose shape is only known once
    # step 0 has run. Every rank reaches step 0, so collective setup is safe.
    grad_ring = None
    recv_dk_bufs: list[Tensor] | None = None
    recv_dv_bufs: list[Tensor] | None = None
    pending_grad: PendingGradExchange | None = None

    def _setup_grad_transport(dk_like: Tensor, dv_like: Tensor) -> None:
        nonlocal grad_ring, recv_dk_bufs, recv_dv_bufs, grad_comm_stream
        if (use_nccl4py_ring or use_symm_mem) and dk_like.shape != dv_like.shape:
            raise ValueError(
                "One-sided gradient rings need dk and dv accumulators of the same "
                f"shape, got {tuple(dk_like.shape)} and {tuple(dv_like.shape)}."
            )
        if use_nccl4py_ring:
            from usp_attention.ring.ring_pipeline_nccl4py import (
                _get_or_create_grad_ring,
            )

            grad_ring = _get_or_create_grad_ring(
                dk_like.shape,
                torch.float32,
                q_local.device,
                process_group,
                grad_comm_stream,
            )
        elif use_symm_mem:
            from usp_attention.ring.ring_pipeline_symm import (
                SymmMemGradRing,
            )

            grad_ring = SymmMemGradRing(
                grad_shape=dk_like.shape,
                dtype=torch.float32,
                device=q_local.device,
                process_group=process_group,
                comm_stream=grad_comm_stream,
            )
        else:
            if grad_comm_stream is None:
                grad_comm_stream = (
                    torch.cuda.Stream(device=q_local.device)
                    if use_cuda_comm_stream and q_local.is_cuda
                    else None
                )
            recv_dk_bufs = [
                torch.empty_like(dk_like, dtype=torch.float32),
                torch.empty_like(dk_like, dtype=torch.float32),
            ]
            recv_dv_bufs = [
                torch.empty_like(dv_like, dtype=torch.float32),
                torch.empty_like(dv_like, dtype=torch.float32),
            ]

    grad_state = Fp32GradState() if fp32_grads else None
    current_key = k_local
    current_value = v_local
    dq_accum: Tensor | None = None
    dk_accum: Tensor | None = None
    dv_accum: Tensor | None = None

    def _start_kv_prefetch(step: int) -> None:
        """Enqueue async KV read for the given step on the comm stream."""
        if kv_direct is not None:
            ready_event = kv_direct.prepare_prefetch(step)
            kv_direct.start_prefetch(step, ready_event)
        else:
            assert recv_key_bufs is not None and recv_val_bufs is not None
            buf_idx = (step - 1) % 2
            _pending_kv_holder[0] = _launch_async_kv_rotation(
                curr_key=current_key,
                curr_value=current_value,
                recv_key=recv_key_bufs[buf_idx],
                recv_value=recv_val_bufs[buf_idx],
                process_group=process_group,
                comm_stream=kv_comm_stream,
            )

    def _wait_kv_prefetch(step: int) -> tuple[Tensor, Tensor]:
        """Wait for the KV prefetch and return (key, value)."""
        if kv_direct is not None:
            kv_direct.wait_prefetch()
            kv_direct.ack_prefetch(step)
            return kv_direct.staging(step)
        assert recv_key_bufs is not None and recv_val_bufs is not None
        return _wait_pending_kv_exchange(
            _pending_kv_holder[0]  # type: ignore[arg-type]
        )

    _pending_kv_holder: list[PendingKVExchange | None] = [None]

    for step in range(world_size):
        # 1. Pre-fetch next step's KV. The symm path performs the READY wait
        # before compute, then leaves only the peer copy on the comm stream.
        if step < world_size - 1:
            _start_kv_prefetch(step + 1)

        # 2. Start async dk/dv read from predecessor.
        # The symm path keeps signal waits off the grad comm stream:
        # prepare_read_prev waits READY on the current stream, start_read_prev
        # enqueues only the DMA copy, and ack_read_prev fires DONE after the
        # copy has landed. The actual DMA wait is deferred to step 4b, after
        # backward compute, so the transfer can overlap with the flash kernel.
        if grad_ring is not None and step > 0:
            ready_event = grad_ring.prepare_read_prev(step - 1)
            grad_ring.start_read_prev(step - 1, ready_event)
        elif pending_grad is not None and step > 0:
            pass  # NCCL grad rotation is already in-flight from prev step 6.

        # 3. Compute local backward for this step's KV.
        # This only needs Q, K, V, out, dout, lse — NOT the incoming dk/dv
        # accumulator. Running it before waiting for dk/dv overlaps the
        # dk/dv transfer with compute.
        with torch.autograd.profiler.record_function(
            _ring_step_trace_name(
                "ring_step_bwd",
                step,
                step_meta=step_meta,
            )
        ):
            accumulate = (
                fp32_grad_step(
                    grad_state, is_first=step == 0, is_last=step == world_size - 1
                )
                if grad_state is not None
                else contextlib.nullcontext()
            )
            with accumulate:
                dq_step, dk_step, dv_step = _ring_step_bwd(
                    q_local,
                    current_key,
                    current_value,
                    out_local,
                    dout,
                    lse_bhs,
                    step=step,
                    step_meta=step_meta,
                    deterministic=deterministic,
                )

        # 4a. Accumulate dq locally (all steps contribute to local Q grad). With
        # fp32 gradients FA4 keeps the running sum and only the last step's dq
        # is materialized, unless FA4 writes dq directly (see _add_step_dq).
        dq_accum = _add_step_dq(
            dq_accum, dq_step, grad_state, is_last=step == world_size - 1
        )

        # This step's dk/dv contribution for the KV shard it holds: FA4's raw
        # fp32 accumulators with fp32 gradients and GQA, else the step output.
        if grad_state is not None and grad_state.dk_raw is not None:
            assert grad_state.dv_raw is not None
            dk_contrib, dv_contrib = grad_state.dk_raw, grad_state.dv_raw
        else:
            dk_contrib, dv_contrib = dk_step.float(), dv_step.float()
        if step == 0:
            _setup_grad_transport(dk_contrib, dv_contrib)

        # 4b. NOW wait for incoming dk/dv accumulator (transfer ran during
        # step 3 compute).
        if grad_ring is not None and step > 0:
            dk_accum, dv_accum = grad_ring.wait_read()
            grad_ring.ack_read_prev(step - 1)
        elif pending_grad is not None:
            recv_dk, recv_dv = _wait_pending_grad_exchange(pending_grad)
            pending_grad = None
            dk_accum = recv_dk
            dv_accum = recv_dv

        # 5. Add this rank's dk/dv contribution for the current KV shard.
        dk_accum = dk_contrib if dk_accum is None else dk_accum + dk_contrib
        dv_accum = dv_contrib if dv_accum is None else dv_accum + dv_contrib

        # 6. Rotate the dk/dv accumulator forward.  The symm-mem path uses
        # initial_publish at step 0 (no prior publish to ack) and republish
        # afterwards (waits DONE(step-1) from our successor before clobbering
        # buf, then fires READY(step)).
        if step < world_size - 1:
            if grad_ring is not None:
                if step == 0:
                    grad_ring.initial_publish(dk_accum, dv_accum)
                else:
                    grad_ring.republish(dk_accum, dv_accum, step)
            else:
                grad_recv_idx = step % 2
                pending_grad = _launch_async_grad_rotation(
                    dk=dk_accum,
                    dv=dv_accum,
                    recv_dk=recv_dk_bufs[grad_recv_idx],
                    recv_dv=recv_dv_bufs[grad_recv_idx],
                    process_group=process_group,
                    comm_stream=grad_comm_stream,
                )

        # 7. Obtain next step's KV.
        if step < world_size - 1:
            current_key, current_value = _wait_kv_prefetch(step + 1)

    if kv_direct is not None:
        kv_direct.finalize()

    # After the loop, dk_accum has all W contributions but sits one hop short
    # of the KV-owning rank. One final rotation delivers it home.  The
    # finalize() call drains the trailing DONE(W-1) so signal counters end
    # the backward call balanced and no pre-seeding is needed next time.
    if grad_ring is not None:
        final_step = world_size - 1
        grad_ring.republish(dk_accum, dv_accum, final_step)
        ready_event = grad_ring.prepare_read_prev(final_step)
        grad_ring.start_read_prev(final_step, ready_event)
        dk_final, dv_final = grad_ring.wait_read()
        grad_ring.ack_read_prev(final_step)
        grad_ring.finalize(final_step)
    else:
        grad_recv_idx = (world_size - 1) % 2
        final_grad = _launch_async_grad_rotation(
            dk=dk_accum,
            dv=dv_accum,
            recv_dk=recv_dk_bufs[grad_recv_idx],
            recv_dv=recv_dv_bufs[grad_recv_idx],
            process_group=process_group,
            comm_stream=grad_comm_stream,
        )
        dk_final, dv_final = _wait_pending_grad_exchange(final_grad)

    assert dq_accum is not None
    if grad_state is not None and grad_state.dk_raw is not None:
        # The summed raw fp32 accumulators are home; convert them once.
        dk_out = torch.empty_like(k_local)
        dv_out = torch.empty_like(v_local)
        convert_dkv(grad_state, dk_final, dv_final, dk_out, dv_out)
        return dq_accum.to(q_local.dtype), dk_out, dv_out
    # One-sided grad rings hand back their receive staging buffers, which the
    # NCCL4Py ring caches and the next backward call overwrites. With fp32 K/V
    # .to() would return them as is, so copy.
    staged = grad_ring is not None
    return (
        dq_accum.to(q_local.dtype),
        dk_final.to(k_local.dtype, copy=staged),
        dv_final.to(v_local.dtype, copy=staged),
    )


@dataclass(kw_only=True, slots=True)
class _RingForwardCtx:
    """Transient non-tensor state for one ring_flash4_attn forward call.

    Holds everything the dispatcher-visible usp_attention::ring_flash4_attn op needs
    that cannot be passed as a tensor argument: the CuTe step metadata, the
    process group, the load-balance plan, token permutations, the seq dim,
    and all the forward orchestration flags. A handle (int) into the registry
    is the only non-tensor argument the custom op takes.
    """

    step_meta: Flash4RingStepMeta
    process_group: dist.ProcessGroup
    rank_plan: RankRingPlan | None
    token_permutation: Tensor | None
    restore_permutation: Tensor | None
    seq_dim: int
    input_is_sharded: bool
    gather_output: bool
    use_cuda_comm_stream: bool
    combine_mode: RingCombineMode
    deterministic: bool
    use_symm_mem: bool
    use_symm_mem_rotator: bool
    use_symm_mem_direct: bool
    use_nccl4py_ring: bool
    kv_comm_stream: torch.cuda.Stream | None = None
    grad_comm_stream: torch.cuda.Stream | None = None
    fp32_grads: bool = False

    def __post_init__(self) -> None:
        # token_permutation and restore_permutation are an inverse pair: the
        # forward permutes inputs by token_permutation and restores the output by
        # restore_permutation, and the backward undoes each by applying the other.
        # That is only valid when they are both set or both None; enforce it here
        # so a caller supplying one without the other fails loudly rather than
        # silently mis-permuting (or hitting apply_token_permutation(x, None)).
        if (self.token_permutation is None) != (self.restore_permutation is None):
            raise ValueError(
                "token_permutation and restore_permutation must both be set or "
                "both be None (they are an inverse pair)."
            )


@dataclass(kw_only=True, slots=True)
class _RingForwardResult:
    """Outputs of _run_ring_forward.

    out/lse are the (optionally gathered + restored) tensors returned to the
    caller. The *_local tensors are the per-rank shards the backward pass needs.
    """

    out: Tensor
    lse: Tensor
    q_local: Tensor
    k_local: Tensor
    v_local: Tensor
    out_local: Tensor
    lse_local: Tensor
    world_size: int


# Transient per-call registry mapping an int handle to the forward context.
# A custom op cannot accept a process group, a Callable mask_mod, or a nested
# dataclass as an argument, so ring_flash4_attn stashes that state here and
# passes only the int handle to torch.ops.usp_attention.ring_flash4_attn. The entry is
# registered immediately before the op call and deregistered in a finally block
# right after, so it is alive throughout forward (including setup_context, which
# copies the Python objects it needs onto ctx for backward).
_RING_CTX_REGISTRY: dict[int, _RingForwardCtx] = {}
# itertools.count().__next__() is atomic at the C level, so handle generation is
# race-free even if forwards were ever issued from multiple Python threads (the
# plain `x += 1` it replaced was a non-atomic read-modify-write). Each handle is
# unique, so the single-key dict ops below need no additional lock.
_RING_CTX_COUNTER = itertools.count()


def _register_ring_ctx(fwd_ctx: _RingForwardCtx) -> int:
    """Register a forward context and return its int handle."""
    cid = next(_RING_CTX_COUNTER)
    _RING_CTX_REGISTRY[cid] = fwd_ctx
    return cid


def _lookup_ring_ctx(cid: int) -> _RingForwardCtx:
    """Look up a registered forward context by handle.

    Raises:
        KeyError: If the handle is not (or no longer) registered.
    """
    return _RING_CTX_REGISTRY[cid]


def _deregister_ring_ctx(cid: int) -> None:
    """Drop a registered forward context (idempotent)."""
    _RING_CTX_REGISTRY.pop(cid, None)


def _to_local_qkv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    fwd_ctx: _RingForwardCtx,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply the forward's token permutation and local sharding to q/k/v.

    Shared by the forward body and the custom op backward so the per-rank
    shards the ring loop operates on are reconstructed identically.
    """
    seq_dim = fwd_ctx.seq_dim
    process_group = fwd_ctx.process_group
    if fwd_ctx.token_permutation is not None:
        q = apply_token_permutation(q, fwd_ctx.token_permutation, seq_dim=seq_dim)
        k = apply_token_permutation(k, fwd_ctx.token_permutation, seq_dim=seq_dim)
        v = apply_token_permutation(v, fwd_ctx.token_permutation, seq_dim=seq_dim)
    if fwd_ctx.input_is_sharded:
        return q.contiguous(), k.contiguous(), v.contiguous()
    q_local = _split_local_shard(q, dim=seq_dim, process_group=process_group)
    k_local = _split_local_shard(k, dim=seq_dim, process_group=process_group)
    v_local = _split_local_shard(v, dim=seq_dim, process_group=process_group)
    return q_local, k_local, v_local


def _run_ring_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    fwd_ctx: _RingForwardCtx,
) -> _RingForwardResult:
    """Run the ring FA4 forward and return local + (optionally) gathered tensors.

    This is the forward body of the dispatcher-visible
    usp_attention::ring_flash4_attn op. It performs the optional
    token permutation, local sharding, ring KV pipeline (selecting the symm-mem /
    nccl4py / NCCL transport), online/flash/python combine, and the optional
    output gather. It does NOT touch any autograd ctx.

    out/lse are gathered + restored when gather_output is True, else equal to the
    local tensors. The q/k/v/out/lse local shards are always the per-rank tensors
    the backward pass needs.
    """
    step_meta = fwd_ctx.step_meta
    process_group = fwd_ctx.process_group
    seq_dim = fwd_ctx.seq_dim
    world_size = dist.get_world_size(process_group)

    q_local, k_local, v_local = _to_local_qkv(q, k, v, fwd_ctx)

    use_online_combine = fwd_ctx.combine_mode == "online"
    use_flash_combine = not use_online_combine and fwd_ctx.combine_mode not in (
        "python",
        "fp32",
    )
    merger: _LocalRingResultMerger | None = None
    online_state: _OnlineRingFwdState | None = None

    if use_online_combine:
        online_state = _OnlineRingFwdState()
    elif use_flash_combine:
        merger = _LocalRingResultMergerFlash()
    else:
        merger = _LocalRingResultMerger()

    def _step_callback(
        step: int,
        step_k: Tensor,
        step_v: Tensor,
        step_plan: RingStepPlan | None,
    ) -> None:
        with torch.autograd.profiler.record_function(
            _ring_step_trace_name(
                "ring_step_fwd",
                step,
                step_meta=step_meta,
                step_plan=step_plan,
            )
        ):
            step_out, step_lse = _ring_step_fwd(
                q_local,
                step_k,
                step_v,
                step,
                step_meta,
                online_state=online_state,
                total_steps=world_size,
                fp32_partials=fwd_ctx.combine_mode == "fp32",
            )
        if use_online_combine:
            if step_lse is not None:
                assert online_state is not None
                online_state.lse = step_lse
        else:
            step_lse = _normalize_step_lse(step_out, step_lse)
            assert merger is not None
            merger.step(step_out, step_lse)

    if fwd_ctx.use_symm_mem_rotator and not fwd_ctx.use_symm_mem:
        raise ValueError(
            "use_symm_mem_rotator=True requires use_symm_mem=True; "
            "the rotator is a symmetric-memory forward pipeline variant."
        )
    if fwd_ctx.use_symm_mem_direct and not fwd_ctx.use_symm_mem:
        raise ValueError(
            "use_symm_mem_direct=True requires use_symm_mem=True; "
            "the direct-read pipeline is a symmetric-memory forward variant."
        )
    if fwd_ctx.use_symm_mem_direct and fwd_ctx.use_symm_mem_rotator:
        raise ValueError(
            "use_symm_mem_direct and use_symm_mem_rotator are mutually "
            "exclusive forward pipeline variants; enable at most one."
        )
    if fwd_ctx.use_symm_mem and fwd_ctx.use_nccl4py_ring:
        raise ValueError(
            "use_symm_mem and use_nccl4py_ring are mutually exclusive ring "
            "transport backends."
        )

    _pipeline_fn = run_async_ring_kv_pipeline
    if fwd_ctx.use_nccl4py_ring:
        from usp_attention.ring.ring_pipeline_nccl4py import (
            run_async_ring_kv_pipeline_nccl4py_direct,
        )

        _pipeline_fn = run_async_ring_kv_pipeline_nccl4py_direct
    elif fwd_ctx.use_symm_mem:
        from usp_attention.ring.ring_pipeline_symm import (
            run_async_ring_kv_pipeline_symm,
            run_async_ring_kv_pipeline_symm_direct,
            run_async_ring_kv_pipeline_symm_rotator,
        )

        if fwd_ctx.use_symm_mem_direct:
            _pipeline_fn = run_async_ring_kv_pipeline_symm_direct
        elif fwd_ctx.use_symm_mem_rotator:
            _pipeline_fn = run_async_ring_kv_pipeline_symm_rotator
        else:
            _pipeline_fn = run_async_ring_kv_pipeline_symm

    transport_name = _ring_transport_trace_name(
        use_symm_mem=fwd_ctx.use_symm_mem,
        use_symm_mem_rotator=fwd_ctx.use_symm_mem_rotator,
        use_symm_mem_direct=fwd_ctx.use_symm_mem_direct,
        use_nccl4py_ring=fwd_ctx.use_nccl4py_ring,
    )
    with torch.autograd.profiler.record_function(
        f"ring_flash4_fwd/transport={transport_name}/combine={fwd_ctx.combine_mode}"
    ):
        _pipeline_fn(
            initial_key=k_local,
            initial_value=v_local,
            process_group=process_group,
            step_callback=_step_callback,
            rank_plan=fwd_ctx.rank_plan,
            use_cuda_comm_stream=fwd_ctx.use_cuda_comm_stream,
            comm_stream=fwd_ctx.kv_comm_stream,
        )

    if use_online_combine:
        assert online_state is not None and online_state.out is not None
        assert online_state.lse is not None, (
            "LSE was not captured from the last online step"
        )
        out_local = online_state.out.to(q_local.dtype)
        lse_local = online_state.lse
    else:
        assert merger is not None
        out_local, lse_local = merger.result()
        out_local = out_local.to(q_local.dtype)

    if fwd_ctx.gather_output:
        lse_seq_dim = seq_dim
        out = _all_gather_cat(out_local, dim=seq_dim, process_group=process_group)
        lse = _all_gather_cat(lse_local, dim=lse_seq_dim, process_group=process_group)
        if fwd_ctx.restore_permutation is not None:
            out = apply_token_permutation(
                out, fwd_ctx.restore_permutation, seq_dim=seq_dim
            )
            lse = apply_token_permutation(
                lse, fwd_ctx.restore_permutation, seq_dim=lse_seq_dim
            )
    else:
        out, lse = out_local, lse_local

    return _RingForwardResult(
        out=out,
        lse=lse,
        q_local=q_local,
        k_local=k_local,
        v_local=v_local,
        out_local=out_local,
        lse_local=lse_local,
        world_size=world_size,
    )


def _run_ring_backward_from_saved(
    *,
    dout: Tensor,
    dlse: Tensor,
    q_local: Tensor,
    k_local: Tensor,
    v_local: Tensor,
    out_local: Tensor,
    lse_local: Tensor,
    fwd_ctx: _RingForwardCtx,
) -> tuple[Tensor, Tensor, Tensor]:
    """Shared ring FA4 backward body for both the autograd Function and the op.

    Reverses the forward's gather/permute/shard sequence on dout, runs the ring
    backward loop, then gathers + restores the q/k/v gradients to match the
    forward's input layout. Returns (dq, dk, dv).

    ``dlse`` is unused: the op marks its LSE output non-differentiable.
    """
    del dlse
    process_group = fwd_ctx.process_group
    seq_dim = fwd_ctx.seq_dim

    if fwd_ctx.gather_output:
        # Forward output was: permute -> shard -> compute -> gather -> restore.
        # Incoming dout is in original (restored) order. Undo restore, then shard.
        if fwd_ctx.token_permutation is not None:
            dout = apply_token_permutation(
                dout,
                fwd_ctx.token_permutation,
                seq_dim=seq_dim,
            )
        dout = _split_local_shard(dout, dim=seq_dim, process_group=process_group)

    dout = dout.contiguous()

    transport_name = _ring_transport_trace_name(
        use_symm_mem=fwd_ctx.use_symm_mem,
        use_symm_mem_rotator=fwd_ctx.use_symm_mem_rotator,
        use_symm_mem_direct=fwd_ctx.use_symm_mem_direct,
        use_nccl4py_ring=fwd_ctx.use_nccl4py_ring,
    )
    with torch.autograd.profiler.record_function(
        f"ring_flash4_bwd/transport={transport_name}"
    ):
        dq_local, dk_local, dv_local = _run_ring_backward(
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            out_local=out_local,
            dout=dout,
            lse_local=lse_local,
            step_meta=fwd_ctx.step_meta,
            process_group=process_group,
            deterministic=fwd_ctx.deterministic,
            fp32_grads=fwd_ctx.fp32_grads,
            use_cuda_comm_stream=fwd_ctx.use_cuda_comm_stream,
            use_symm_mem=fwd_ctx.use_symm_mem,
            use_nccl4py_ring=fwd_ctx.use_nccl4py_ring,
            kv_comm_stream=fwd_ctx.kv_comm_stream,
            grad_comm_stream=fwd_ctx.grad_comm_stream,
        )

    if not fwd_ctx.input_is_sharded:
        dq = _all_gather_cat(dq_local, dim=seq_dim, process_group=process_group)
        dk = _all_gather_cat(dk_local, dim=seq_dim, process_group=process_group)
        dv = _all_gather_cat(dv_local, dim=seq_dim, process_group=process_group)
        if fwd_ctx.restore_permutation is not None:
            dq = apply_token_permutation(
                dq, fwd_ctx.restore_permutation, seq_dim=seq_dim
            )
            dk = apply_token_permutation(
                dk, fwd_ctx.restore_permutation, seq_dim=seq_dim
            )
            dv = apply_token_permutation(
                dv, fwd_ctx.restore_permutation, seq_dim=seq_dim
            )
    else:
        dq, dk, dv = dq_local, dk_local, dv_local
        if fwd_ctx.restore_permutation is not None:
            dq = apply_token_permutation(
                dq, fwd_ctx.restore_permutation, seq_dim=seq_dim
            )
            dk = apply_token_permutation(
                dk, fwd_ctx.restore_permutation, seq_dim=seq_dim
            )
            dv = apply_token_permutation(
                dv, fwd_ctx.restore_permutation, seq_dim=seq_dim
            )

    return dq, dk, dv


def _ring_lse_shape(
    q_shape: tuple[int, ...] | torch.Size, combine_mode: RingCombineMode
) -> tuple[int, int, int]:
    """Shape of the LSE the ring op returns for a ``(B, S, H, D)`` query.

    The online kernel returns FA4's ``(B, H, S)``; the python, fp32 and flash
    merges (and ``"auto"``, which runs the flash merge) return ``(B, S, H)``
    (:func:`~usp_attention.ring.ring_combine._normalize_step_lse`).
    """
    batch, seq, heads = q_shape[0], q_shape[1], q_shape[2]
    if combine_mode == "online":
        return (batch, heads, seq)
    return (batch, seq, heads)


# ---------------------------------------------------------------------------
# Dispatcher-visible custom op (ring path).
#
# An opaque torch.autograd.Function would be invisible to the dispatcher, so a
# selective-activation-checkpointing (SAC) policy could not MUST_SAVE its
# output. The op below registers usp_attention::ring_flash4_attn as a real
# dispatcher op so a SAC context_fn can see it (torch.ops.usp_attention.ring_flash4_attn
# .default) and force-save its result, mirroring usp_attention::flash4_attn in
# usp_attention/fa4/flash4_attn.py.
#
# A custom op cannot accept a process group, the CuTe mask_mod Callable, or the
# nested Flash4RingStepMeta as arguments, so only q, k, v, and an int registry
# handle (ctx_id) are passed. The handle resolves to a _RingForwardCtx that
# carries step_meta (incl. mask_mod), the process group, and the forward flags.
# ---------------------------------------------------------------------------

if HAS_FLASH4_CUTE:

    @torch.library.custom_op("usp_attention::ring_flash4_attn", mutates_args=())
    def _ring_flash4_attn_op(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        ctx_id: int,
    ) -> tuple[Tensor, Tensor]:
        fwd_ctx = _lookup_ring_ctx(ctx_id)
        result = _run_ring_forward(q, k, v, fwd_ctx)
        # Contiguous so the output layout is deterministic and matches
        # register_fake (which allocates contiguous) — a non-contiguous q must not
        # leak input strides into the output or torch.compile's stride guards fail.
        return result.out.contiguous(), result.lse.contiguous()

    @_ring_flash4_attn_op.register_fake
    def _ring_flash4_attn_op_fake(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        ctx_id: int,
    ) -> tuple[Tensor, Tensor]:
        # q is (B, S, H_q, D); out has q's shape. Allocate CONTIGUOUS (not
        # empty_like, which would copy q's strides) to match the contiguous output
        # the real op returns. The LSE layout depends on the combine mode, read
        # from the registered forward context like the real op does.
        combine_mode = _lookup_ring_ctx(ctx_id).combine_mode
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        lse = q.new_empty(_ring_lse_shape(q.shape, combine_mode), dtype=torch.float32)
        return out, lse

    def _ring_flash4_attn_op_setup_context(ctx, inputs, output):
        q, k, v, ctx_id = inputs
        out, lse = output
        # The registry entry is still alive at forward time. A custom op can only
        # save its own inputs/outputs for backward (not arbitrary intermediates),
        # so save q, k, v, out, lse and stash the Python forward context. Backward
        # reconstructs the per-rank shards from these saved tensors.
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.fwd_ctx = _lookup_ring_ctx(ctx_id)
        # The ring backward does not pass a gradient through the LSE to FA4
        # (dlse), so mark it non-differentiable instead of silently dropping a
        # loss term that uses it. dlse then arrives as zeros and is ignored.
        ctx.mark_non_differentiable(lse)

    def _ring_flash4_attn_op_backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        fwd_ctx: _RingForwardCtx = ctx.fwd_ctx
        seq_dim = fwd_ctx.seq_dim
        process_group = fwd_ctx.process_group

        # Reconstruct the per-rank shards the backward loop needs from the saved
        # op inputs/outputs, mirroring the forward's permute -> shard sequence.
        q_local, k_local, v_local = _to_local_qkv(q, k, v, fwd_ctx)

        out_local, lse_local = out, lse
        if fwd_ctx.gather_output:
            # out/lse were gathered + restored in forward; undo the restore by
            # applying token_permutation (its inverse), then shard. Gate on the
            # permutation actually applied — token_permutation — so the check and
            # the value match (restore_permutation is its inverse pair, enforced
            # in _RingForwardCtx.__post_init__).
            if fwd_ctx.token_permutation is not None:
                out_local = apply_token_permutation(
                    out_local, fwd_ctx.token_permutation, seq_dim=seq_dim
                )
                lse_local = apply_token_permutation(
                    lse_local, fwd_ctx.token_permutation, seq_dim=seq_dim
                )
            out_local = _split_local_shard(
                out_local, dim=seq_dim, process_group=process_group
            )
            lse_local = _split_local_shard(
                lse_local, dim=seq_dim, process_group=process_group
            )

        dq, dk, dv = _run_ring_backward_from_saved(
            dout=dout,
            dlse=dlse,
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            out_local=out_local,
            lse_local=lse_local,
            fwd_ctx=fwd_ctx,
        )
        # Gradients for (q, k, v) then None for ctx_id.
        return dq, dk, dv, None

    torch.library.register_autograd(
        "usp_attention::ring_flash4_attn",
        _ring_flash4_attn_op_backward,
        setup_context=_ring_flash4_attn_op_setup_context,
    )


def ring_flash4_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    process_group: dist.ProcessGroup,
    step_meta: Flash4RingStepMeta,
    rank_plan: RankRingPlan | None = None,
    token_permutation: Tensor | None = None,
    restore_permutation: Tensor | None = None,
    seq_dim: int = 1,
    input_is_sharded: bool = False,
    gather_output: bool = False,
    use_cuda_comm_stream: bool = True,
    combine_mode: RingCombineMode = "auto",
    return_lse: bool = False,
    deterministic: bool = False,
    use_symm_mem: bool = False,
    use_symm_mem_rotator: bool = False,
    use_symm_mem_direct: bool = False,
    use_nccl4py_ring: bool = False,
    kv_comm_stream: torch.cuda.Stream | None = None,
    grad_comm_stream: torch.cuda.Stream | None = None,
    fp32_grads: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run ring attention with async KV rotation.

    Each ring step runs ``_flash_attn_fwd`` / ``_flash_attn_bwd`` as described
    by ``step_meta``, through the dispatcher-visible
    ``usp_attention::ring_flash4_attn`` custom op (forward and backward).

    When ``use_symm_mem=True``, the forward KV pipeline reads peer KV via
    NVLink-backed symmetric memory.  Three such implementations exist and the
    ``use_symm_mem_rotator`` / ``use_symm_mem_direct`` flags select between
    them (at most one may be ``True``):

    * default (both flags ``False``): each rank issues direct reads from all
      W-1 peers per call with a per-call ``hdl.barrier()`` for cross-call
      safety
      (:func:`~ring_pipeline_symm.run_async_ring_kv_pipeline_symm`).
    * ``use_symm_mem_rotator=True``: strict next/prev rotation with
      bidirectional READY/DONE signaling
      (:func:`~ring_pipeline_symm.run_async_ring_kv_pipeline_symm_rotator`),
      matching the backward grad ring's topology.
    * ``use_symm_mem_direct=True``: direct peer reads with explicit per-peer
      READY/DONE handshakes (no per-step ``staging -> buf`` rotation copy and
      no collective barrier)
      (:func:`~ring_pipeline_symm.run_async_ring_kv_pipeline_symm_direct`).

    All three require ``use_symm_mem=True``.

    ``use_nccl4py_ring=True`` selects the NCCL4Py direct-send transport for
    both forward KV and backward dk/dv rotation. It is mutually exclusive with
    ``use_symm_mem``.

    With ``return_lse=True`` the LSE is ``(B, H, S)`` for
    ``combine_mode="online"`` and ``(B, S, H)`` otherwise. It is not
    differentiable (``requires_grad=False``): the backward propagates
    gradients through the output only.

    The torch symmetric-memory backward pipeline is unaffected by either
    symmetric-memory forward variant flag and continues to use
    symmetric-memory direct reads for KV rotation plus
    :class:`~ring_pipeline_symm.SymmMemGradRing` for dk/dv rotation when
    ``use_symm_mem=True``.

    Raises:
        ValueError: If the process group world size is invalid, if q/k/v are on
            different devices, if combine_mode is unknown, if ``use_symm_mem_rotator=True`` without ``use_symm_mem=True``, if
            ``use_symm_mem_direct=True`` without ``use_symm_mem=True``, if
            both ``use_symm_mem_rotator`` and ``use_symm_mem_direct`` are set,
            or if ``use_nccl4py_ring`` is combined with ``use_symm_mem``.
        RuntimeError: If torch.distributed is not initialized.
    """
    if combine_mode not in get_args(RingCombineMode):
        raise ValueError(
            f"Unknown combine_mode {combine_mode!r}; expected one of "
            f"{get_args(RingCombineMode)}."
        )
    if use_symm_mem_rotator and not use_symm_mem:
        raise ValueError(
            "use_symm_mem_rotator=True requires use_symm_mem=True; "
            "the rotator is a symmetric-memory forward pipeline variant."
        )
    if use_symm_mem_direct and not use_symm_mem:
        raise ValueError(
            "use_symm_mem_direct=True requires use_symm_mem=True; "
            "the direct-read pipeline is a symmetric-memory forward variant."
        )
    if use_symm_mem_direct and use_symm_mem_rotator:
        raise ValueError(
            "use_symm_mem_direct and use_symm_mem_rotator are mutually "
            "exclusive forward pipeline variants; enable at most one."
        )
    if use_symm_mem and use_nccl4py_ring:
        raise ValueError(
            "use_symm_mem and use_nccl4py_ring are mutually exclusive ring "
            "transport backends."
        )

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before ring attention."
        )
    world_size = dist.get_world_size(process_group)
    if world_size < 1:
        raise ValueError(f"Invalid process group world size: {world_size=}.")
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"q/k/v must be on the same device, got {q.device=}, {k.device=}, {v.device=}."
        )

    # Route through the dispatcher-visible usp_attention::ring_flash4_attn custom op so
    # a SAC policy can see and MUST_SAVE the attention output. The op cannot
    # accept the process group / mask_mod Callable / nested step_meta as
    # arguments, so the non-tensor forward state is stashed in a transient
    # registry and only an int handle is passed. The handle is deregistered in
    # a finally block once forward (including setup_context, which copies the
    # Python objects it needs onto ctx) has run.
    if not HAS_FLASH4_CUTE:
        raise RuntimeError("flash_attn.cute is required for ring_flash4_attn.")
    fwd_ctx = _RingForwardCtx(
        step_meta=step_meta,
        process_group=process_group,
        rank_plan=rank_plan,
        token_permutation=token_permutation,
        restore_permutation=restore_permutation,
        seq_dim=seq_dim,
        input_is_sharded=input_is_sharded,
        gather_output=gather_output,
        use_cuda_comm_stream=use_cuda_comm_stream,
        combine_mode=combine_mode,
        deterministic=deterministic,
        use_symm_mem=use_symm_mem,
        use_symm_mem_rotator=use_symm_mem_rotator,
        use_symm_mem_direct=use_symm_mem_direct,
        use_nccl4py_ring=use_nccl4py_ring,
        kv_comm_stream=kv_comm_stream,
        grad_comm_stream=grad_comm_stream,
        fp32_grads=fp32_grads,
    )
    cid = _register_ring_ctx(fwd_ctx)
    try:
        out, lse = torch.ops.usp_attention.ring_flash4_attn(q, k, v, cid)
    finally:
        _deregister_ring_ctx(cid)
    if return_lse:
        return out, lse
    return out


# ---------------------------------------------------------------------------
# Alias kept for tests/distributed/profile_unified_ring_traces.py.
# ---------------------------------------------------------------------------
build_ring_flash4_plan = build_ring_plan
