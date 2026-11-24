# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Unified sequence parallelism (Ulysses x ring) over FlashAttention-4.

A packed sequence of ``S`` tokens is described by per-document lengths and, for
each document, a list of splits with an attention mode (``"causal"``, ``"full"``
or ``"noise"``). :func:`build_usp_plan` turns that description into everything a
rank needs to run attention on its local shard:

1. The FA4 per-token mask metadata (aux tensors) for the whole sequence.
2. Optionally a PTRR load-balancing permutation of 128-token blocks, so every
   ring rank receives the same amount of unmasked work.
3. Per-ring-step aux tensors and block-sparse metadata for the cross Q/KV mask.

Each CP rank owns ``S / (ring_size * ulysses_size)`` tokens of the (permuted)
sequence. :func:`usp_attention` runs a Ulysses all-to-all (sequence gathered,
heads scattered), ring attention over the ring group, and the inverse
all-to-all, returning the attention output for the local shard.

Example:
    ```python
    device = init_distributed()
    groups = USPGroups.create(ring_size=2, ulysses_size=4)
    plan = build_usp_plan(
        document_lens=[65536, 65536],
        split_lens=[1024, 64512, 2048, 63488],
        attn_modes=["causal", "full", "causal", "full"],
        groups=groups,
        config=USPConfig(),
        device=device,
    )
    q_local, k_local, v_local = (plan.shard(t) for t in (q, k, v))
    out_local = usp_attention(q_local, k_local, v_local, plan)
    out = plan.gather(out_local)
    ```
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention.distributed import USPGroups, seq_all_to_all
from usp_attention.fa4.column_ranges import column_query_ranges
from usp_attention.fa4.flash4_attn import (
    AttentionMode,
    FA4SparseMask,
    build_fa4_sparse_mask,
    flash4_attn,
    flash4_attn_dense,
)
from usp_attention.fa4.interval_mask import (
    IntervalPlan,
    build_interval_plan,
    interval_attention,
)
from usp_attention.reference import visible_key_counts
from usp_attention.ring.dense_gather import (
    DenseGatherPlan,
    build_dense_gather_plan,
    dense_gather_attention,
    dense_gather_supported,
)
from usp_attention.ring.dense_ring import (
    DenseRingPlan,
    build_dense_ring_plan,
    chunk_assignment,
    dense_ring_attention,
    dense_ring_supported,
    rank_work,
)
from usp_attention.ring.flash4_ring_attn import (
    COLUMN_RANGE_MASK_MOD,
    CROSS_QKV_SPARSE_MASK_MOD,
    POSITION_RANGE_MASK_MOD,
    Flash4RingStepMeta,
    NativeRingMask,
    _ring_step_bwd,
    _ring_step_fwd,
    build_precomputed_cross_qkv_aux_plan,
    cross_qkv_sparse_mask_mod,
    ring_flash4_attn,
    use_position_ranges,
)
from usp_attention.ring.generation_ring import (
    GenerationRingPlan,
    build_generation_ring_plan,
    generation_ring_attention,
    ring_chunk_assignment,
)
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.halo import (
    HaloPlan,
    build_halo_plan,
    halo_attention,
    needed_key_masks,
)
from usp_attention.ring.packed_layout import build_packed_layout
from usp_attention.ring.ring_combine import _normalize_step_lse
from usp_attention.ring.ring_planning import RankRingPlan, build_ring_plan
from usp_attention.sliding_window import segment_end_positions, validate_window
from usp_attention.types import ONLINE_MAX_HEAD_DIM, RingCombineMode

if TYPE_CHECKING:
    from jaxtyping import Float, Int, Shaped

Transport = Literal[
    "nccl", "symm_mem", "symm_mem_rotator", "symm_mem_direct", "nccl4py"
]

FA4_BLOCK_SIZE = 128


@dataclass(kw_only=True, slots=True)
class USPConfig:
    """Execution options for :func:`usp_attention`.

    Attributes:
        combine_mode: How per-step ring partials are merged (see
            :data:`usp_attention.types.RingCombineMode`). ``None`` picks the
            most accurate mode the GPU supports: ``"online"`` on Hopper (the
            online kernel keeps the softmax state in fp32 across ring steps),
            ``"fp32"`` on Blackwell (fp32 split-KV partials, same accuracy;
            the online kernel is SM90-only), and ``"flash"`` elsewhere. The
            online kernel covers every head_dim FA4 supports on SM90 (up to
            ``ONLINE_MAX_HEAD_DIM`` = 256), so Hopper always uses it.
        load_balance: Reorder 128-token blocks with PTRR so ring ranks receive
            equal unmasked work. Requires the sequence length to be a multiple
            of ``ring_size * ulysses_size * 128``.
        balance: How ``load_balance`` deals tokens to ring ranks. ``"ptrr"``
            scatters 128-token blocks by cost; ``"chunks"`` deals the fewest
            contiguous chunks per rank (1, 2, 4, ... up to 64) that keep every
            rank within 2.5% of the mean visible pairs. Use ``"chunks"`` for
            models with sliding-window layers: their plans
            (:func:`build_sliding_window_plan`) then exchange only a small KV
            halo per chunk instead of rotating the whole KV around the ring.
        direct_planner_block_sparse: Let PTRR classify blocks directly from the
            per-token metadata instead of building a global flex ``BlockMask``.
        direct_block_sparse: Build each ring step's block-sparse tensors
            directly from per-block summaries instead of ``create_block_mask``.
        transport: KV rotation backend. ``"nccl"`` uses async P2P send/recv,
            ``"nccl4py"`` uses one-sided NCCL RMA (NCCL >= 2.29.7, needs
            ``NCCL_CUMEM_ENABLE=1``), and the experimental ``"symm_mem*"``
            variants use torch symmetric memory inside one NVLink domain.
        use_cuda_comm_stream: Run KV rotation on a dedicated CUDA stream.
        deterministic: Use deterministic FA4 backward kernels. The native,
            varlen, interval, block-sparse (ring or single-GPU) and KV-gather
            (halo) paths pass it to FA4; the dense ring, generation ring and
            dense gather paths do not take the flag, so :func:`usp_attention`
            raises ``ValueError`` there when a backward will run (build the
            plan with ``dense_ring=False`` to avoid them).
        fp32_grads: Keep ring-attention gradients in fp32 across ring steps
            and round once at the end, instead of rounding every step's
            partial gradient. Applies to dq always and to dk / dv with
            grouped-query attention. Matches single-GPU FA4 gradient accuracy
            and is slightly faster (fewer conversion kernels per step).
        native_plain_masks: Run a layout that is one plain causal or full
            document with FA4's native masks instead of block sparsity and a
            ``mask_mod``. Without a ring this is one FA4 call, packed causal
            documents (every split causal) run as one varlen causal call over
            the document boundaries, and packed multimodal layouts whose noise
            splits end their documents can run as two dense varlen calls
            (``interval_masks``); with a ring, a
            causal document is striped over the ring ranks (token ``i`` on
            ring rank ``i % ring_size``) so every ring step is a native causal
            or strictly causal call (replacing PTRR). Same result, a few
            percent faster on long sequences.
        interval_masks: Without a ring, run packed multimodal layouts as two
            dense varlen FA4 calls merged by log-sum-exp instead of block
            sparsity (``fa4/interval_mask.py``). ``None`` enables it on
            Blackwell, where it is 13-25% faster (FA4 drops its 2-CTA
            backward and causal scheduling under block sparsity); on H100 it
            is 2-7% faster but merges bf16 partials, so output / gradient
            error is 1.06x / 1.13x a single FA4 call instead of 1.00x.
        dense_ring: With a ring, run packed multimodal layouts over contiguous
            chunks (dealt to ring ranks by work) with dense varlen FA4 calls
            per ring step instead of PTRR + block sparsity
            (``ring/dense_ring.py``), when the chunks balance the ranks within
            15% of the mean work. ``None`` enables it on
            Blackwell; step partials merge in fp32 like the ``python`` combine.
            Also selects dense varlen calls for ``kv_exchange="gather"``. Set
            it on H100 above head_dim 128, where FA4 has no 128-token block
            sparsity for training.
        dense_ring_chunks: Force this many chunks per ring rank; ``None``
            selects a balanced count automatically. Must be a positive integer.
        dense_ring_refine: Refine the exact generation ring's chunk assignment.
        dense_ring_inplace: Experimental interval dense ring: attend on query
            rows in place instead of gathering them. Off by default.
        dense_ring_dkv_overlap: Experimental interval dense ring: send remote
            dK/dV contributions as each backward step completes. Off by default.
        kv_exchange: How a ring group shares KV for packed masks. ``"ring"``
            rotates every rank's KV shard around the ring, one block-sparse
            step per rank, overlapping compute and transfer. ``"gather"``
            brings in, with one all-to-all, only the KV blocks this rank's
            queries can see (at most an all-gather) and attends over them with
            one block-sparse FA4 call; dK / dV return the same way and are
            added in fp32 (:mod:`usp_attention.ring.halo`). With the dense
            ring enabled (``dense_ring``), layouts the dense calls can express
            and whose spans reach ``dense_min_span`` gather over contiguous
            chunks and attend with dense varlen FA4 calls instead
            (:mod:`usp_attention.ring.dense_gather`). Gather avoids the
            ring's per-step launches and merges and its imbalance between
            steps; the ring keeps less remote KV in memory at a time.
        gather_regather: With ``kv_exchange="gather"`` (and sliding-window
            halos), save only the local KV for the backward and exchange the
            halo again there: the gathered KV is then not kept per layer
            (about one extra exchange per layer, much less activation memory
            for long, dense masks).
        halo_overlap: With ``kv_exchange="gather"`` on H100, hide the KV gather
            and the backward's dK / dV return behind compute: the local keys and
            the gathered halo run as two calls of the exact online kernel (fp32
            softmax state across both), so the output matches the single call.
        gather_max_keys: Dense gather: keys gathered per FA4 call. Long
            documents gather their prefix once per query sequence, so calls
            are split; smaller values lower peak memory for more calls.
        dense_min_span: Use dense gather calls only when the median full /
            noise span has at least this many tokens (each such span gathers
            its document's prefix); shorter spans use block-sparse gather.
        exact_dense_ring: With a ring on dense kernels, also run layouts the
            interval dense ring handles (noise spans ending documents) on the
            exact generation ring (fp16 forward partials, fp32 gradients,
            dK / dV rotating with the KV). Layouts with noise spans inside
            documents always use it. Layouts too short-spanned for dense calls
            take the block-sparse ring rather than the bf16-merging dense
            ring, and without a ring the generation ring's one-rank case
            replaces the interval path.
        order_only: Internal: build only the token order of a ring plan (no
            per-step block-sparse metadata), for plans that replace the ring
            exchange (KV gather) and for the base order of sliding-window
            plans.
        reorder_vision_groups: For layouts with vision groups, store each such
            document as its text followed by its image spans in reverse
            order (:mod:`usp_attention.ring.group_order`), which leaves almost
            no partially masked tiles; the mask is unchanged because it
            compares original positions.
        column_range_masks: Test ``lo[k] <= pos[q] < hi[k]`` instead of the
            position-aware predicate wherever every key's visible queries form
            one range in some token order (:mod:`usp_attention.fa4.column_ranges`:
            any layout without groups, sliding windows included, and
            vision-group layouts): without a ring on the vision-group order,
            and on every block-sparse ring step. Two or three loads per
            element of a partial tile instead of eight.
        fa3_gather: With ``kv_exchange="gather"`` on H100, run a dense gather
            that is one causal call (packed causal documents) with
            FlashAttention-3 instead of FA4: about 20% faster at head_dim 256
            (Qwen3.8 shapes), but FA3 rounds each rank's dK / dV contribution
            to bf16 before they are summed (1.4x the single-GPU dK / dV error,
            output and dq unchanged). Off by default: FA4 keeps them in fp32.
        fa3_on_hopper: Without a ring, run a native causal document with
            FlashAttention-3 on H100 when ``flash-attn-3`` is installed (its
            causal kernel is 1-3% faster than FA4's there; FA4 stays faster on
            full masks). No effect on other GPUs, with a ring, or for
            block-sparse layouts.
    """

    combine_mode: RingCombineMode | None = None
    load_balance: bool = True
    balance: Literal["ptrr", "chunks"] = "ptrr"
    direct_planner_block_sparse: bool = True
    direct_block_sparse: bool = True
    transport: Transport = "nccl"
    use_cuda_comm_stream: bool = True
    deterministic: bool = False
    fp32_grads: bool = True
    native_plain_masks: bool = True
    fa3_on_hopper: bool = True
    fa3_gather: bool = False
    interval_masks: bool | None = None
    dense_ring: bool | None = None
    dense_ring_chunks: int | None = None
    dense_ring_refine: bool = True
    dense_ring_inplace: bool = False
    dense_ring_dkv_overlap: bool = False
    kv_exchange: Literal["ring", "gather"] = "ring"
    gather_regather: bool = False
    halo_overlap: bool = False
    gather_max_keys: int = 1 << 21
    dense_min_span: int = 512
    exact_dense_ring: bool = False
    order_only: bool = False
    reorder_vision_groups: bool = True
    column_range_masks: bool = True

    def __post_init__(self) -> None:
        if self.dense_ring_chunks is not None and (
            type(self.dense_ring_chunks) is not int or self.dense_ring_chunks < 1
        ):
            raise ValueError("dense_ring_chunks must be a positive integer or None")

    def resolved_combine_mode(
        self, device: torch.device | str, head_dim: int | None = None
    ) -> RingCombineMode:
        """Resolve the configured combine mode or the device-specific default.

        Args:
            device: Device on which attention will execute.
            head_dim: Largest of the query/key and value head dims, if known.

        Returns:
            The explicit combine_mode when set; otherwise "online" for Hopper
            up to ONLINE_MAX_HEAD_DIM and "python" above it, "fp32" for newer
            CUDA GPUs, or "flash" for other devices.

        Raises:
            ValueError: If ``combine_mode="online"`` is set explicitly with a
                head_dim the online kernel cannot run.
        """
        too_wide = head_dim is not None and head_dim > ONLINE_MAX_HEAD_DIM
        if self.combine_mode is not None:
            if self.combine_mode == "online" and too_wide:
                raise ValueError(
                    f'combine_mode="online" supports head_dim <= '
                    f'{ONLINE_MAX_HEAD_DIM}, got {head_dim}. Use "python" or '
                    "leave combine_mode unset."
                )
            return self.combine_mode
        device = torch.device(device)
        if device.type != "cuda" or torch.version.hip is not None:
            return "flash"
        major = torch.cuda.get_device_capability(device)[0]
        if major == 9:
            # FA4 has no split-KV on SM90, so "fp32" is not an option here.
            return "python" if too_wide else "online"
        if major in (10, 11):
            return "fp32"
        return "flash"


def required_seq_multiple(groups: USPGroups) -> int:
    """Return the multiple the packed sequence length must be padded to.

    Args:
        groups: Ulysses and ring process-group configuration.

    Returns:
        The product of ring size, Ulysses size and FA4 block size.
    """
    return groups.ring_size * groups.ulysses_size * FA4_BLOCK_SIZE


def pad_packed_layout(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    multiple: int,
) -> tuple[list[int], list[int], list[AttentionMode]]:
    """Pad a packed layout with an isolated causal document.

    The padding tokens form their own document, so they never attend to or
    from real tokens. It is ``"causal"`` so that a layout of causal documents
    stays all-causal and keeps the packed-causal fast paths (one varlen call
    without a ring).

    Args:
        document_lens: Lengths of the packed documents, in tokens.
        split_lens: Lengths of the attention spans, in document order.
        attn_modes: Attention mode for each span; must match split_lens.
        multiple: Positive token alignment required by the parallel groups.

    Returns:
        A tuple (document_lens, split_lens, attn_modes) of new lists, with a
        separate causal padding document appended only when needed. The input
        lists are not modified.
    """
    seq_len = sum(document_lens)
    pad = (-seq_len) % multiple
    if pad == 0:
        return list(document_lens), list(split_lens), list(attn_modes)
    return (
        [*document_lens, pad],
        [*split_lens, pad],
        [*attn_modes, "causal"],
    )


@dataclass(kw_only=True, slots=True)
class USPPlan:
    """Per-rank attention plan for one packed-sequence layout.

    Build with :func:`build_usp_plan` once per batch layout and reuse it for
    every attention layer.

    Attributes:
        config: Execution options.
        groups: Process groups and coordinates.
        seq_len: Global packed sequence length.
        token_permutation: Global token order used for sharding, or ``None``
            when tokens keep their original order.
        restore_permutation: Inverse of ``token_permutation``.
        step_meta: Ring-step mask metadata (ring size > 1).
        rank_plan: PTRR schedule for this ring rank (ring size > 1 with load
            balancing).
        fa4_mask: Single-rank FA4 mask (ring size == 1), ``None`` when
            ``native_mask`` is set.
        native_mask: ``"causal"`` or ``"full"`` when the layout is one plain
            document run with FA4's native masks, or ``"causal"`` for packed
            causal documents run as varlen attention.
        native_cu_seqlens: Document boundaries (int32, ``[0, ..., seq_len]``)
            for the varlen path, ``None`` otherwise.
        native_max_seqlen: Longest document for the varlen path.
        dense_ring: Dense varlen calls per ring step over contiguous shards,
            see :mod:`usp_attention.ring.dense_ring`.
        halo: KV gather (sliding-window halo, or ``kv_exchange="gather"``)
            instead of a ring (see :mod:`usp_attention.ring.halo`).
        generation_ring: Exact ring on dense varlen calls over contiguous
            chunks, see :mod:`usp_attention.ring.generation_ring`.
        dense_gather: KV gather on dense varlen calls over contiguous chunks
            (``kv_exchange="gather"`` with a dense-ring layout), see
            :mod:`usp_attention.ring.dense_gather`.
        interval_plan: Two-call dense decomposition of a packed multimodal
            mask (no ring), see :mod:`usp_attention.fa4.interval_mask`.
        softmax_scale: Softmax scale, ``None`` for ``1/sqrt(head_dim)``.
        sliding_window: Sliding window in keys for the native and varlen
            paths (block-sparse, interval and ring plans carry it in their
            own metadata), ``None`` for none.
    """

    config: USPConfig
    groups: USPGroups
    seq_len: int
    token_permutation: Tensor | None
    restore_permutation: Tensor | None
    step_meta: Flash4RingStepMeta | None
    rank_plan: RankRingPlan | None
    fa4_mask: FA4SparseMask | None
    native_mask: Literal["causal", "full"] | None = None
    native_cu_seqlens: Tensor | None = None
    native_max_seqlen: int | None = None
    interval_plan: IntervalPlan | None = None
    dense_ring: DenseRingPlan | None = None
    halo: HaloPlan | None = None
    dense_gather: DenseGatherPlan | None = None
    generation_ring: GenerationRingPlan | None = None
    softmax_scale: float | None = None
    sliding_window: int | None = None
    _kv_comm_stream: torch.cuda.Stream | None = field(default=None, repr=False)
    _grad_comm_stream: torch.cuda.Stream | None = field(default=None, repr=False)

    @property
    def local_len(self) -> int:
        """Tokens owned by each CP rank."""
        return self.seq_len // self.groups.cp_size

    def shard(
        self, x: Shaped[Tensor, " *global_shape"], *, seq_dim: int = 1
    ) -> Shaped[Tensor, " *local_shape"]:
        """Return this rank's shard of a full packed-sequence tensor.

        Args:
            x: Full packed-sequence tensor in original token order.
            seq_dim: Dimension containing the packed token sequence.

        Returns:
            A contiguous tensor holding this rank's tokens in planned order.
            The input tensor is not modified.
        """
        if self.token_permutation is not None:
            x = torch.index_select(
                x, seq_dim, self.token_permutation.to(device=x.device)
            )
        start = self.groups.cp_rank * self.local_len
        return x.narrow(seq_dim, start, self.local_len).contiguous()

    def gather(
        self, x_local: Shaped[Tensor, " *local_shape"], *, seq_dim: int = 1
    ) -> Shaped[Tensor, " *global_shape"]:
        """All-gather local shards over the CP group and restore original order.

        Not differentiable; intended for evaluation and testing.

        Args:
            x_local: This rank's shard in planned token order; all ranks must
                participate when the CP group has more than one rank.
            seq_dim: Dimension containing the local token sequence.

        Returns:
            The full tensor in original token order, replicated on every CP
            rank. The collective is not differentiable.
        """
        if self.groups.cp_group is None:
            full = x_local
        else:
            parts = [torch.empty_like(x_local) for _ in range(self.groups.cp_size)]
            dist.all_gather(parts, x_local.contiguous(), group=self.groups.cp_group)
            full = torch.cat(parts, dim=seq_dim)
        if self.restore_permutation is not None:
            full = torch.index_select(
                full, seq_dim, self.restore_permutation.to(device=full.device)
            )
        return full

    def comm_streams(self) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        """Get cached CUDA streams for KV and gradient rotation.

        Returns:
            A tuple (kv_stream, grad_stream), lazily allocated on the current
            CUDA device and reused by subsequent calls.
        """
        if self._kv_comm_stream is None:
            self._kv_comm_stream = torch.cuda.Stream()
            self._grad_comm_stream = torch.cuda.Stream()
        assert self._grad_comm_stream is not None
        return self._kv_comm_stream, self._grad_comm_stream


def _is_blackwell(device: torch.device | str) -> bool:
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] in (
        10,
        11,
    )


def _use_dense_ring(config: USPConfig, device: torch.device | str) -> bool:
    return config.dense_ring if config.dense_ring is not None else _is_blackwell(device)


def _use_interval_masks(config: USPConfig, device: torch.device | str) -> bool:
    if config.interval_masks is not None:
        return config.interval_masks
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] in (
        10,
        11,
    )


def _execution_path(plan: USPPlan) -> str:
    """The kernel path :func:`usp_attention` takes for ``plan`` (same order)."""
    if plan.halo is not None:
        return "halo"
    if plan.generation_ring is not None:
        return "generation_ring"
    if plan.dense_gather is not None:
        return "dense_gather"
    if plan.dense_ring is not None:
        return "dense_ring"
    if plan.step_meta is not None:
        if plan.step_meta.native is not None:
            return "native_ring"
        return "block_sparse_ring" if plan.groups.ring_size > 1 else "single_step"
    if plan.interval_plan is not None:
        return "interval"
    if plan.native_mask is not None:
        return "native"
    return "fa4_mask"


# Paths whose kernels mask by position (``position_ids``): every block-sparse
# step of the position-aware cross Q/KV mask, and the KV gather built on it.
_POSITION_AWARE_PATHS = ("halo", "block_sparse_ring", "single_step")
# Paths whose backward kernels do not take FA4's ``deterministic`` flag (their
# dense varlen calls in ring/dense_ring.py run the default backward).
_NONDETERMINISTIC_PATHS = ("generation_ring", "dense_gather", "dense_ring")
# Paths that run FA4 with 128-token block sparsity.
_BLOCK_SPARSE_PATHS = ("halo", "block_sparse_ring", "single_step", "fa4_mask")


def _custom_positions(position_ids: Tensor | None, seq_len: int) -> Tensor | None:
    """``position_ids`` if they differ from the default ``arange(seq_len)``, else ``None``.

    Raises:
        ValueError: If ``position_ids`` is not a 1-D tensor of ``seq_len`` values.
    """
    if position_ids is None:
        return None
    if position_ids.shape != (seq_len,):
        raise ValueError(
            f"position_ids must have shape ({seq_len},), got {tuple(position_ids.shape)}."
        )
    if torch.equal(position_ids.cpu(), torch.arange(seq_len, dtype=position_ids.dtype)):
        return None
    return position_ids


def _check_positions_honored(plan: USPPlan, position_ids: Tensor | None) -> None:
    """Raise if non-default ``position_ids`` reached a plan that masks by index.

    Raises:
        ValueError: If ``position_ids`` is set and the plan's path ignores it.
    """
    if position_ids is None or plan.config.order_only:
        return
    path = _execution_path(plan)
    if path in _POSITION_AWARE_PATHS:
        return
    raise ValueError(
        "position_ids other than arange(seq_len) are only honored by the "
        "position-aware block-sparse plans (block-sparse ring steps, vision-group "
        f"or given token orders, KV gather); this layout and config planned the "
        f"{path} path, which masks by token index. Pass token_permutation (e.g. "
        "torch.arange(seq_len)) for the position-aware mask, or leave "
        "position_ids unset."
    )


def _fa4_lacks_block_sparsity(
    device: torch.device | str, head_dim: int, head_dim_v: int, *, backward: bool
) -> bool:
    """Whether FA4 cannot run 128-token block sparsity at these head dims here.

    On SM90 the forward's KV tile is 128 wide up to head_dim 128, and up to
    192 with a value head_dim <= 128 (80 or 112 otherwise); the backward's is
    narrower than 128 above head_dim 128 (96 at 192). FA4 rejects block-sparse
    calls whose tiles do not divide the 128-token blocks. Other GPUs are not
    checked.
    """
    widest = max(head_dim, head_dim_v)
    if widest <= 128:
        return False
    device = torch.device(device)
    if device.type != "cuda" or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability(device)[0] != 9:
        return False
    return backward or widest > 192 or head_dim_v > 128


def _check_plan_supported(
    plan: USPPlan,
    *,
    device: torch.device,
    head_dim: int,
    head_dim_v: int,
    backward: bool,
) -> None:
    """Fail fast, before any collective, on a plan its kernels cannot run as configured.

    Raises:
        ValueError: If ``deterministic`` is requested for a backward on a path
            whose kernels ignore it, or the plan needs FA4 block sparsity at a
            head_dim where this GPU has none.
    """
    path = _execution_path(plan)
    if backward and plan.config.deterministic and path in _NONDETERMINISTIC_PATHS:
        raise ValueError(
            f"USPConfig(deterministic=True) is not supported on the {path} path: "
            "its backward kernels do not take a deterministic flag. Build the "
            "plan with dense_ring=False for the block-sparse (or, without a "
            "ring, interval) paths, which pass it to FA4."
        )
    if path in _BLOCK_SPARSE_PATHS and _fa4_lacks_block_sparsity(
        device, head_dim, head_dim_v, backward=backward
    ):
        raise ValueError(
            f"The {path} path runs FA4 with 128-token block sparsity, which FA4 "
            f"does not support on this GPU (SM90) at head_dim {head_dim} / value "
            f"head_dim {head_dim_v}" + (" with a backward" if backward else "") + ". "
            "Build the plan for dense varlen kernels where the layout allows: "
            "USPConfig(dense_ring=True) with a ring (kv_exchange='gather' "
            "also needs dense_min_span=0 when the full / noise spans are "
            "short), or interval_masks=True without one."
        )


def build_usp_plan(
    *,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    groups: USPGroups,
    config: USPConfig,
    device: torch.device,
    softmax_scale: float | None = None,
    position_ids: Int[Tensor, " tokens"] | None = None,
    sliding_window: int | None = None,
    token_permutation: Int[Tensor, " tokens"] | None = None,
) -> USPPlan:
    """Build the attention plan for a packed-sequence layout.

    Args:
        document_lens: Length of each packed document (sample).
        split_lens: Length of each split, in order across all documents.
        attn_modes: Attention mode of each split.
        groups: Process groups from :meth:`USPGroups.create`.
        config: Execution options.
        device: CUDA device for the plan tensors.
        softmax_scale: Softmax scale, ``None`` for ``1/sqrt(head_dim)``.
        position_ids: Causal position of each packed token in original order;
            defaults to ``arange(seq_len)``, and passing ``arange`` is the same
            as leaving it unset. Other positions are honored only by the
            position-aware block-sparse plans (block-sparse ring steps,
            vision-group or given token orders, KV gather and halos); the
            native, varlen, interval, dense and single-GPU block-sparse
            kernels mask by token index, so a plan that would run one of them
            raises ``ValueError``. Passing ``token_permutation`` (e.g.
            ``torch.arange(seq_len)``) selects the position-aware mask.
        sliding_window: Limit causal attention to the last ``sliding_window``
            keys; full, noise and group spans still see their whole span (see
            :mod:`usp_attention.sliding_window`). For a model that mixes
            sliding and global layers, build the global plan without a window
            and derive the sliding one with :func:`build_sliding_window_plan`,
            so both shard tokens the same way.
        token_permutation: Use this token order (new index -> original index)
            instead of planning one, e.g. another plan's ``token_permutation``
            so layers with different splits (say Ulysses-32 for global layers
            and a ring with a sliding-window halo for local layers) shard
            activations identically. Every step then runs the position-aware
            block-sparse mask.

    Returns:
        The plan for the calling rank.

    Raises:
        ValueError: If the layout is inconsistent, the sequence length is not
            a multiple of :func:`required_seq_multiple`, or non-default
            ``position_ids`` would be ignored by the planned path.
    """
    position_ids = _custom_positions(position_ids, sum(document_lens))
    plan = _build_usp_plan(
        document_lens=document_lens,
        split_lens=split_lens,
        attn_modes=attn_modes,
        groups=groups,
        config=config,
        device=device,
        softmax_scale=softmax_scale,
        position_ids=position_ids,
        sliding_window=sliding_window,
        token_permutation=token_permutation,
    )
    _check_positions_honored(plan, position_ids)
    return plan


def _build_usp_plan(
    *,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    groups: USPGroups,
    config: USPConfig,
    device: torch.device,
    softmax_scale: float | None = None,
    position_ids: Int[Tensor, " tokens"] | None = None,
    sliding_window: int | None = None,
    token_permutation: Int[Tensor, " tokens"] | None = None,
) -> USPPlan:
    """:func:`build_usp_plan` without the ``position_ids`` check.

    Also builds the internal base plans (token orders) of the sliding-window
    and KV-gather plans, which may ignore positions their final plan uses.
    """
    seq_len = sum(document_lens)
    if sum(split_lens) != seq_len or len(split_lens) != len(attn_modes):
        raise ValueError(
            f"split_lens must sum to {seq_len=} and match attn_modes, got "
            f"{sum(split_lens)=}, {len(split_lens)=}, {len(attn_modes)=}."
        )
    multiple = required_seq_multiple(groups)
    if seq_len % multiple != 0:
        raise ValueError(
            f"Packed length {seq_len} must be a multiple of {multiple} "
            "(ring_size * ulysses_size * 128); see pad_packed_layout()."
        )

    if validate_window(sliding_window):
        # The global plan decides the token order; the window only changes
        # masks, so build just that order. Without causal spans the window
        # changes nothing and the base itself is returned, so build it whole.
        order_only = config.order_only or any(mode == "causal" for mode in attn_modes)
        base = _build_usp_plan(
            document_lens=document_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            groups=groups,
            config=dataclasses.replace(config, order_only=order_only),
            device=device,
            softmax_scale=softmax_scale,
            position_ids=position_ids,
            token_permutation=token_permutation,
        )
        return _build_sliding_window_plan(
            dataclasses.replace(base, config=config),
            sliding_window=sliding_window,
            document_lens=document_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            device=device,
            position_ids=position_ids,
        )

    ring_size = groups.ring_size
    if (
        ring_size > 1
        and config.kv_exchange == "ring"
        and token_permutation is None
        and len(split_lens) > 1
        and _use_dense_ring(config, device)
        and dense_gather_supported(document_lens, split_lens, attn_modes)
        and (
            config.exact_dense_ring
            or not dense_ring_supported(document_lens, split_lens, attn_modes)
        )
        and _dense_gather_worthwhile(split_lens, attn_modes, config)
        and (any(m != "causal" for m in attn_modes) or config.dense_ring is True)
    ):
        # Dense steps over contiguous chunks; exact for any noise placement.
        # Chunks balance every ring step (PTRR's criterion at chunk size).
        forced = config.dense_ring_chunks
        if forced is not None and seq_len % (ring_size * forced * FA4_BLOCK_SIZE):
            raise ValueError(
                "dense_ring_chunks must divide the sequence into chunks of "
                f"{FA4_BLOCK_SIZE} tokens (sequence={seq_len}, ring={ring_size}, chunks={forced})"
            )
        bounds = (
            {"min_chunks_per_rank": int(forced), "max_chunks_per_rank": int(forced)}
            if forced
            else {}
        )
        chunks = ring_chunk_assignment(
            document_lens,
            split_lens,
            attn_modes,
            ring_size=ring_size,
            refine=config.dense_ring_refine,
            **bounds,
        )
        perm = torch.cat([torch.arange(lo, hi) for rank in chunks for lo, hi in rank])
        perm = perm.to(device)
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=perm,
            restore_permutation=torch.argsort(perm),
            step_meta=None,
            rank_plan=None,
            fa4_mask=None,
            generation_ring=build_generation_ring_plan(
                document_lens,
                split_lens,
                attn_modes,
                ring_rank=groups.ring_rank,
                ring_size=ring_size,
                chunks=chunks,
                device=device,
                max_gathered_keys=config.gather_max_keys,
            ),
            softmax_scale=softmax_scale,
        )
    if (
        ring_size > 1
        and config.kv_exchange == "gather"
        and token_permutation is None
        and _use_dense_ring(config, device)
        and dense_gather_supported(document_lens, split_lens, attn_modes)
        and not dense_ring_supported(document_lens, split_lens, attn_modes)
        and len(split_lens) > 1
        and _dense_gather_worthwhile(split_lens, attn_modes, config)
    ):
        # Noise spans inside documents (interleaved generation): dense calls
        # over contiguous chunks, without the dense ring's interval condition.
        chunks = _chunk_ranges(document_lens, split_lens, attn_modes, ring_size)
        perm = torch.cat([torch.arange(lo, hi) for rank in chunks for lo, hi in rank])
        perm = perm.to(device)
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=perm,
            restore_permutation=torch.argsort(perm),
            step_meta=None,
            rank_plan=None,
            fa4_mask=None,
            dense_gather=build_dense_gather_plan(
                document_lens,
                split_lens,
                attn_modes,
                ring_rank=groups.ring_rank,
                ring_size=ring_size,
                chunks=chunks,
                device=device,
                allow_fa3=config.fa3_gather,
                max_gathered_keys=config.gather_max_keys,
                # USP_DENSE_GATHER_EXACT=1: one call per query row instead of
                # shared prefixes with merged fp16 partials (benchmarking).
                exact_output=os.environ.get("USP_DENSE_GATHER_EXACT", "0") == "1",
            ),
            softmax_scale=softmax_scale,
        )
    if ring_size > 1 and config.kv_exchange == "gather":
        worthwhile = _dense_gather_worthwhile(split_lens, attn_modes, config)
        base = _build_usp_plan(
            document_lens=document_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            groups=groups,
            # Spans too short for dense calls: block-sparse gather over a
            # block-sparse base (same chunks idea), so skip the dense ring.
            config=dataclasses.replace(
                config,
                kv_exchange="ring",
                order_only=True,
                **({} if worthwhile else {"dense_ring": False}),
            ),
            device=device,
            softmax_scale=softmax_scale,
            position_ids=position_ids,
            token_permutation=token_permutation,
        )
        if base.step_meta is not None and base.step_meta.native is not None:
            # One plain document: the native striped ring is already balanced.
            return dataclasses.replace(base, config=config)
        if config.order_only:
            # Only the token order (e.g. the base of a sliding-window plan).
            return USPPlan(
                config=config,
                groups=groups,
                seq_len=seq_len,
                token_permutation=base.token_permutation,
                restore_permutation=base.restore_permutation,
                step_meta=None,
                rank_plan=None,
                fa4_mask=None,
                softmax_scale=softmax_scale,
            )
        if base.dense_ring is not None and worthwhile:
            # Dense kernels (no block sparsity): gather over the same chunks.
            return dataclasses.replace(
                base,
                config=config,
                dense_ring=None,
                dense_gather=build_dense_gather_plan(
                    document_lens,
                    split_lens,
                    attn_modes,
                    ring_rank=groups.ring_rank,
                    ring_size=ring_size,
                    chunks=base.dense_ring.chunks,
                    device=device,
                    allow_fa3=config.fa3_gather,
                    max_gathered_keys=config.gather_max_keys,
                    exact_output=os.environ.get("USP_DENSE_GATHER_EXACT", "0") == "1",
                ),
            )
        gathered = _halo_plan(
            dataclasses.replace(base, config=config),
            window=None,
            document_lens=document_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            device=device,
            position_ids=position_ids,
            max_halo_fraction=1.0,
        )
        assert gathered is not None
        return gathered

    if token_permutation is not None:
        return _build_group_ordered_plan(
            token_permutation,
            document_lens=document_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            groups=groups,
            config=config,
            device=device,
            softmax_scale=softmax_scale,
            position_ids=position_ids,
            rebalance=False,
        )
    group_perm = (
        vision_group_order(document_lens, split_lens, attn_modes)
        if config.reorder_vision_groups
        else None
    )
    if group_perm is not None:
        return _build_group_ordered_plan(
            group_perm,
            document_lens=document_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            groups=groups,
            config=config,
            device=device,
            softmax_scale=softmax_scale,
            position_ids=position_ids,
        )

    if (
        ring_size == 1
        and config.native_plain_masks
        and len(document_lens) > 1
        and all(mode == "causal" for mode in attn_modes)
    ):
        # Packed causal documents: varlen attention on FA4's dense kernels
        # (no block sparsity or mask_mod; 2-CTA backward on Blackwell).
        bounds = torch.tensor([0, *document_lens], dtype=torch.int64).cumsum(0)
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=None,
            restore_permutation=None,
            step_meta=None,
            rank_plan=None,
            fa4_mask=None,
            native_mask="causal",
            native_cu_seqlens=bounds.to(device=device, dtype=torch.int32),
            native_max_seqlen=max(document_lens),
            softmax_scale=softmax_scale,
        )

    if (
        ring_size == 1
        and config.exact_dense_ring
        and config.native_plain_masks
        and len(split_lens) > 1
        and any(m != "causal" for m in attn_modes)
        and _use_dense_ring(config, device)
        and dense_gather_supported(document_lens, split_lens, attn_modes)
    ):
        # Exact dense calls without a ring: the generation ring's one-rank
        # case (fp16 partials merged in fp32, fp32 dK / dV), in place of the
        # interval path's bf16 merges.
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=None,
            restore_permutation=None,
            step_meta=None,
            rank_plan=None,
            fa4_mask=None,
            generation_ring=build_generation_ring_plan(
                document_lens,
                split_lens,
                attn_modes,
                ring_rank=0,
                ring_size=1,
                chunks=[[(0, seq_len)]],
                device=device,
                max_gathered_keys=config.gather_max_keys,
            ),
            softmax_scale=softmax_scale,
        )

    interval = (
        build_interval_plan(document_lens, split_lens, attn_modes, device=device)
        if ring_size == 1
        and config.native_plain_masks
        and len(split_lens) > 1
        and _use_interval_masks(config, device)
        else None
    )
    if interval is not None:
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=None,
            restore_permutation=None,
            step_meta=None,
            rank_plan=None,
            fa4_mask=None,
            interval_plan=interval,
            softmax_scale=softmax_scale,
        )

    if (
        ring_size == 1
        and config.native_plain_masks
        and len(split_lens) == 1
        and attn_modes[0] in ("causal", "full")
    ):
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=None,
            restore_permutation=None,
            step_meta=None,
            rank_plan=None,
            fa4_mask=None,
            native_mask=attn_modes[0],
            softmax_scale=softmax_scale,
        )

    if (
        ring_size > 1
        and config.native_plain_masks
        and len(split_lens) == 1
        and attn_modes[0] in ("causal", "full")
    ):
        token_permutation = restore_permutation = None
        if attn_modes[0] == "causal":
            # Stripe r holds tokens r, r + ring_size, ... in order.
            token_permutation = (
                torch.arange(seq_len, device=device).view(-1, ring_size).T.reshape(-1)
            )
            restore_permutation = torch.argsort(token_permutation)
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=token_permutation,
            restore_permutation=restore_permutation,
            step_meta=Flash4RingStepMeta(
                softmax_scale=softmax_scale,
                native=NativeRingMask(
                    mode="causal_striped" if attn_modes[0] == "causal" else "full",
                    ring_rank=groups.ring_rank,
                    ring_size=ring_size,
                ),
            ),
            rank_plan=None,
            fa4_mask=None,
            native_mask=attn_modes[0],
            softmax_scale=softmax_scale,
        )

    if (
        ring_size > 1
        and len(split_lens) > 1
        # Packed causal documents take the dense ring only when asked for
        # explicitly (e.g. head_dim 256 on H100, where FA4 has no 128-token
        # block sparsity); by default they run block-sparse with PTRR.
        and (any(m != "causal" for m in attn_modes) or config.dense_ring is True)
        and _use_dense_ring(config, device)
        # This ring merges bf16 partials; exact plans take the block-sparse ring.
        and not config.exact_dense_ring
        and dense_ring_supported(document_lens, split_lens, attn_modes)
    ):
        # Chunks dealt to ring ranks by work (snake order): the fewest per
        # rank that balance the ranks within 2.5% (more chunks mean smaller
        # calls and more merges), else the best balance available. On 8x B300
        # the threshold picks the fastest measured count for the Qwen2.5-7B-layout
        # (28 query / 4 KV heads) and 32-head mixed masks from 256k to 1M tokens.
        best = None
        forced = config.dense_ring_chunks
        for per_rank in (int(forced),) if forced else (1, 2, 4, 8, 16):
            if seq_len % (ring_size * per_rank * FA4_BLOCK_SIZE):
                break
            chunks = chunk_assignment(
                document_lens,
                split_lens,
                attn_modes,
                ring_size=ring_size,
                chunks_per_rank=per_rank,
            )
            work = rank_work(document_lens, split_lens, attn_modes, ring_size, chunks)
            imbalance = max(work) / (sum(work) / len(work))
            if best is None or imbalance < best[0] - 1e-9:
                best = (imbalance, per_rank, chunks, work)
            if imbalance <= 1.025:
                break
        if best is None:
            raise ValueError(
                "dense_ring_chunks must divide the sequence into chunks of "
                f"{FA4_BLOCK_SIZE} tokens (sequence={seq_len}, ring={ring_size}, "
                f"chunks={forced})"
            )
        _, per_rank, chunks, work = best
        if max(work) <= 1.15 * (sum(work) / len(work)):
            token_permutation = restore_permutation = None
            if per_rank > 1:
                token_permutation = torch.cat(
                    [torch.arange(lo, hi) for rank in chunks for lo, hi in rank]
                ).to(device)
                restore_permutation = torch.argsort(token_permutation)
            return USPPlan(
                config=config,
                groups=groups,
                seq_len=seq_len,
                token_permutation=token_permutation,
                restore_permutation=restore_permutation,
                step_meta=None,
                rank_plan=None,
                fa4_mask=None,
                dense_ring=build_dense_ring_plan(
                    document_lens,
                    split_lens,
                    attn_modes,
                    ring_rank=groups.ring_rank,
                    ring_size=ring_size,
                    device=device,
                    chunks=chunks,
                    inplace=config.dense_ring_inplace,
                    dkv_overlap=config.dense_ring_dkv_overlap,
                ),
                softmax_scale=softmax_scale,
            )

    use_ptrr = ring_size > 1 and config.load_balance
    # With a ring only the aux tensors are needed; PTRR without the direct
    # planner builds its reference BlockMask from the same aux tensors.
    fa4_mask = build_fa4_sparse_mask(
        document_lens,
        split_lens,
        attn_modes,
        device=device,
        build_block_sparse=ring_size == 1,
    )

    if ring_size == 1:
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=None,
            restore_permutation=None,
            step_meta=None,
            rank_plan=None,
            fa4_mask=fa4_mask,
            softmax_scale=softmax_scale,
        )

    given_positions = position_ids
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device, dtype=torch.int64)
    position_ids = position_ids.to(device=device, dtype=torch.int64)

    token_permutation = restore_permutation = None
    rank_plan = None
    ring_aux_tensors = fa4_mask.aux_tensors
    if use_ptrr and config.balance == "chunks":
        token_permutation = _balanced_chunks(
            document_lens, split_lens, attn_modes, ring_size, device
        )
        restore_permutation = torch.argsort(token_permutation)
        ring_aux_tensors = [
            aux[token_permutation].contiguous() for aux in ring_aux_tensors
        ]
        position_ids = position_ids[token_permutation].contiguous()
    elif use_ptrr:
        layout = build_packed_layout(
            document_lens,
            split_lens,
            attn_modes,
            device=device,
            aux_tensors=fa4_mask.aux_tensors,
        )
        ring_plan = build_ring_plan(
            layout,
            world_size=ring_size,
            device=device,
            use_direct_planner_block_sparse=config.direct_planner_block_sparse,
            # The per-step plan only labels profiler ranges: build this rank's
            # alone, and none for an order-only plan.
            ranks=() if config.order_only else (groups.ring_rank,),
        )
        token_permutation = ring_plan.token_permutation
        restore_permutation = ring_plan.restore_permutation
        ring_aux_tensors = [
            aux[token_permutation].contiguous() for aux in ring_aux_tensors
        ]
        position_ids = position_ids[token_permutation].contiguous()
        # Very long sequences skip the (diagnostic) per-step plans.
        rank_plan = ring_plan.rank_plans[0] if ring_plan.rank_plans else None

    if config.order_only:
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=token_permutation,
            restore_permutation=restore_permutation,
            step_meta=None,
            rank_plan=None,
            fa4_mask=None,
            softmax_scale=softmax_scale,
        )
    precomputed_aux_plan = build_precomputed_cross_qkv_aux_plan(
        global_aux_tensors=ring_aux_tensors,
        rank=groups.ring_rank,
        world_size=ring_size,
        local_len=seq_len // ring_size,
        global_position_ids=position_ids,
        use_direct_block_sparse=config.direct_block_sparse,
    )
    mask_mod = CROSS_QKV_SPARSE_MASK_MOD
    ranges = (
        _position_ranges(
            document_lens,
            split_lens,
            attn_modes,
            token_permutation,
            given_positions,
            device=device,
        )
        if config.column_range_masks
        else None
    )
    if ranges is not None:
        use_position_ranges(precomputed_aux_plan, *ranges)
        mask_mod = POSITION_RANGE_MASK_MOD
    step_meta = Flash4RingStepMeta(
        mask_mod=mask_mod,
        precomputed_aux_plan=precomputed_aux_plan,
        softmax_scale=softmax_scale,
    )
    return USPPlan(
        config=config,
        groups=groups,
        seq_len=seq_len,
        token_permutation=token_permutation,
        restore_permutation=restore_permutation,
        step_meta=step_meta,
        rank_plan=rank_plan,
        fa4_mask=None,
        softmax_scale=softmax_scale,
    )


def _build_group_ordered_plan(
    group_perm: Tensor,
    *,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    groups: USPGroups,
    config: USPConfig,
    device: torch.device,
    softmax_scale: float | None,
    position_ids: Tensor | None,
    rebalance: bool = True,
) -> USPPlan:
    """Plan for a layout stored in a given token order (e.g. tile-friendly vision groups).

    Tokens are first reordered by :func:`vision_group_order`, then (with a
    ring) balanced with PTRR over that order. Every ring step, and the single
    step without a ring, runs the position-aware cross Q/KV mask.
    """
    seq_len = sum(document_lens)
    ring_size = groups.ring_size
    aux = build_fa4_sparse_mask(
        document_lens, split_lens, attn_modes, device=device, build_block_sparse=False
    ).aux_tensors
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device, dtype=torch.int64)
    position_ids = position_ids.to(device=device, dtype=torch.int64)
    perm = group_perm.to(device)
    if (
        rebalance
        and ring_size > 1
        and config.load_balance
        and config.balance == "chunks"
    ):
        # Contiguous chunks of the reordered tokens (small KV gathers / halos).
        counts = visible_key_counts(document_lens, split_lens, attn_modes)
        perm = perm[
            _balanced_chunks(
                document_lens,
                split_lens,
                attn_modes,
                ring_size,
                device,
                counts=counts[perm.cpu()],
            )
        ]
    elif rebalance and ring_size > 1 and config.load_balance:
        layout = build_packed_layout(
            document_lens, split_lens, attn_modes, device=device, aux_tensors=aux
        ).reordered(perm)
        ring_plan = build_ring_plan(
            layout,
            world_size=ring_size,
            device=device,
            use_direct_planner_block_sparse=True,
            position_ids=position_ids[perm],
        )
        perm = perm[ring_plan.token_permutation.to(device)]
    if config.order_only:
        return USPPlan(
            config=config,
            groups=groups,
            seq_len=seq_len,
            token_permutation=perm,
            restore_permutation=torch.argsort(perm),
            step_meta=None,
            rank_plan=None,
            fa4_mask=None,
            softmax_scale=softmax_scale,
        )
    aux_plan = build_precomputed_cross_qkv_aux_plan(
        global_aux_tensors=[t[perm].contiguous() for t in aux],
        rank=groups.ring_rank,
        world_size=ring_size,
        local_len=seq_len // ring_size,
        global_position_ids=position_ids[perm].contiguous(),
        use_direct_block_sparse=config.direct_block_sparse,
    )
    mask_mod = CROSS_QKV_SPARSE_MASK_MOD
    if ring_size == 1 and config.column_range_masks:
        # One call over the whole reordered sequence: if the order is the
        # vision-group order with default positions, every key's visible
        # queries are one range and the cheap range predicate is exact.
        ranges = _vision_group_column_ranges(
            perm, document_lens, split_lens, attn_modes, position_ids, device
        )
        if ranges is not None:
            aux_plan.steps[0].aux_tensors = list(ranges)
            mask_mod = COLUMN_RANGE_MASK_MOD
    elif config.column_range_masks:
        ranges = _position_ranges(
            document_lens, split_lens, attn_modes, perm, position_ids, device=device
        )
        if ranges is not None:
            use_position_ranges(aux_plan, *ranges)
            mask_mod = POSITION_RANGE_MASK_MOD
    step_meta = Flash4RingStepMeta(
        mask_mod=mask_mod,
        precomputed_aux_plan=aux_plan,
        softmax_scale=softmax_scale,
    )
    return USPPlan(
        config=config,
        groups=groups,
        seq_len=seq_len,
        token_permutation=perm,
        restore_permutation=torch.argsort(perm),
        step_meta=step_meta,
        rank_plan=None,
        fa4_mask=None,
        softmax_scale=softmax_scale,
    )


def _position_ranges(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    perm: Tensor | None,
    position_ids: Tensor | None,
    *,
    device: torch.device,
    sliding_window: int | None = None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Column-range tracks for ring steps, in token order ``perm`` (new -> original).

    Each token's index in mask order (vision-group order for documents with
    groups, else its position) and its key's visible query range ``[lo, hi)``
    in that order, as int32 per token in ``perm``'s order. The mask is the
    same in every order, so ``lo <= pos < hi`` is exact for any permutation
    (:func:`~usp_attention.ring.flash4_ring_attn.use_position_ranges`).
    ``None`` when positions are not the default ``arange`` or some key has no
    single range (:func:`column_query_ranges`).
    """
    seq_len = sum(document_lens)
    if position_ids is not None and not torch.equal(
        position_ids.cpu(), torch.arange(seq_len, dtype=position_ids.dtype)
    ):
        return None
    try:
        lo, hi = column_query_ranges(
            document_lens, split_lens, attn_modes, sliding_window=sliding_window
        )
    except ValueError:
        return None
    group_perm = vision_group_order(document_lens, split_lens, attn_modes)
    order = (  # original index -> index in mask order
        torch.arange(seq_len) if group_perm is None else torch.argsort(group_perm.cpu())
    )
    if perm is not None:
        order = order[perm.cpu()]
    lo_t, hi_t = torch.from_numpy(lo)[order], torch.from_numpy(hi)[order]
    return tuple(
        t.to(device=device, dtype=torch.int32).contiguous() for t in (order, lo_t, hi_t)
    )


def _vision_group_column_ranges(
    perm: Tensor,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    position_ids: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    """Per-key visible query ranges when ``perm`` is the vision-group order.

    ``None`` when the order differs, positions are not the default ``arange``
    (the range form assumes it), or a document mixes groups with full or
    noise spans.
    """
    seq_len = sum(document_lens)
    group_perm = vision_group_order(document_lens, split_lens, attn_modes)
    if group_perm is None or not torch.equal(perm.cpu(), group_perm.cpu()):
        return None
    if not torch.equal(
        position_ids.cpu(), torch.arange(seq_len, dtype=position_ids.dtype)
    ):
        return None
    try:
        lo, hi = column_query_ranges(document_lens, split_lens, attn_modes)
    except ValueError:
        return None
    return (
        torch.from_numpy(lo).to(device),
        torch.from_numpy(hi).to(device),
    )


def _dense_gather_worthwhile(
    split_lens: list[int], attn_modes: list[AttentionMode], config: USPConfig
) -> bool:
    """Whether dense calls pay for their gathered prefixes on this layout.

    A full or noise span's queries are answered over the document's prefix,
    gathered once per span: long spans (generated frames, 1k+ tokens) do
    enough attention work per copied key, short ones (256-token image
    crops) mostly copy. Depends on the layout only, so every rank agrees.
    """
    spans = sorted(n for n, m in zip(split_lens, attn_modes) if m in ("full", "noise"))
    return not spans or spans[len(spans) // 2] >= config.dense_min_span


def _chunk_ranges(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    ring_size: int,
    counts: Tensor | None = None,
) -> list[list[tuple[int, int]]]:
    """The fewest contiguous chunks per ring rank that balance ring ranks."""
    seq_len = sum(document_lens)
    best = None
    per_rank = 1
    while per_rank <= 64 and seq_len % (ring_size * per_rank * FA4_BLOCK_SIZE) == 0:
        chunks = chunk_assignment(
            document_lens,
            split_lens,
            attn_modes,
            ring_size=ring_size,
            chunks_per_rank=per_rank,
            counts=counts,
        )
        work = rank_work(
            document_lens, split_lens, attn_modes, ring_size, chunks, counts=counts
        )
        imbalance = max(work) / (sum(work) / len(work))
        if best is None or imbalance < best[0] - 1e-9:
            best = (imbalance, chunks)
        if imbalance <= 1.025:
            break
        per_rank *= 2
    assert best is not None
    return best[1]


def _balanced_chunks(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    ring_size: int,
    device: torch.device,
    counts: Tensor | None = None,
) -> Tensor:
    """Token order dealing the fewest contiguous chunks that balance ring ranks.

    ``counts`` are the stored tokens' visible keys when the tokens are already
    reordered (e.g. by :func:`vision_group_order`); the result then indexes the
    stored order.
    """
    chunks = _chunk_ranges(document_lens, split_lens, attn_modes, ring_size, counts)
    return torch.cat([torch.arange(lo, hi) for rank in chunks for lo, hi in rank]).to(
        device
    )


def _sharded_aux(
    base: USPPlan,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    device: torch.device,
    position_ids: Tensor | None,
) -> tuple[list[Tensor], Tensor, Tensor]:
    """Mask metadata, positions and segment ends in ``base``'s token order."""
    aux = build_fa4_sparse_mask(
        document_lens, split_lens, attn_modes, device=device, build_block_sparse=False
    ).aux_tensors
    if position_ids is None:
        position_ids = torch.arange(base.seq_len, device=device, dtype=torch.int64)
    position_ids = position_ids.to(device=device, dtype=torch.int64)
    segment_end = segment_end_positions(aux[0], position_ids)
    perm = base.token_permutation
    if perm is not None:
        aux = [t[perm].contiguous() for t in aux]
        position_ids = position_ids[perm].contiguous()
        segment_end = segment_end[perm].contiguous()
    return aux, position_ids, segment_end


def _halo_plan(
    base: USPPlan,
    *,
    window: int | None,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    device: torch.device,
    position_ids: Tensor | None,
    max_halo_fraction: float,
) -> USPPlan | None:
    """A KV-gather plan in ``base``'s token order, or ``None`` if the halo is too big."""
    groups = base.groups
    aux, positions, segment_end = _sharded_aux(
        base, document_lens, split_lens, attn_modes, device, position_ids
    )
    ranges = (
        _position_ranges(
            document_lens,
            split_lens,
            attn_modes,
            base.token_permutation,
            position_ids,
            device=device,
            sliding_window=window,
        )
        if base.config.column_range_masks
        else None
    )
    halo = build_halo_plan(
        global_aux_tensors=aux,
        global_position_ids=positions,
        global_segment_end=segment_end,
        ring_rank=groups.ring_rank,
        ring_size=groups.ring_size,
        local_len=base.seq_len // groups.ring_size,
        sliding_window=window,
        mask_mod=(
            cross_qkv_sparse_mask_mod(window)
            if ranges is None
            else POSITION_RANGE_MASK_MOD
        ),
        position_ranges=ranges,
        max_halo_fraction=max_halo_fraction,
        overlap=base.config.halo_overlap,
        # Default positions: needed keys follow from the layout's intervals.
        needed_masks=(
            needed_key_masks(
                document_lens,
                split_lens,
                attn_modes,
                ring_size=groups.ring_size,
                token_permutation=base.token_permutation,
                sliding_window=window,
            )
            if position_ids is None
            else None
        ),
    )
    if halo is None:
        return None
    return USPPlan(
        config=base.config,
        groups=groups,
        seq_len=base.seq_len,
        softmax_scale=base.softmax_scale,
        sliding_window=window,
        token_permutation=base.token_permutation,
        restore_permutation=base.restore_permutation,
        step_meta=None,
        rank_plan=None,
        fa4_mask=None,
        halo=halo,
    )


def build_sliding_window_plan(
    base: USPPlan,
    *,
    sliding_window: int,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    device: torch.device,
    position_ids: Int[Tensor, " tokens"] | None = None,
) -> USPPlan:
    """Plan a sliding-window layer that shards tokens exactly like ``base``.

    Models such as Gemma 3 interleave sliding-window and global attention
    layers over the same activations, so every layer must use one token order.
    Build ``base`` for the global layers with :func:`build_usp_plan`, then
    derive the sliding layers' plan from it here. Without a ring, a ``base`` in
    natural token order uses FA4's local attention when the layout allows
    (packed causal documents, or the two-call interval path when no span is
    longer than the window) and the windowed block-sparse mask otherwise; a
    reordered ``base`` (vision groups, or a given ``token_permutation``) runs
    the position-aware windowed mask in ``base``'s order. With a ring, every ring step runs
    block-sparse over ``base``'s token order; steps and blocks the window
    empties are skipped. When the KV that the window can reach from each ring
    rank is at most half of what a ring rotates (``base`` built with
    ``USPConfig(balance="chunks")``), a halo exchange replaces the ring: one
    all-to-all brings in just those KV blocks and one FA4 call attends over
    them (:mod:`usp_attention.ring.halo`).

    Args:
        base: Plan of the global layers for the same layout.
        sliding_window: Window in keys (a token sees itself and the
            ``sliding_window - 1`` keys before it, within its document).
        document_lens: Same layout as ``base``.
        split_lens: Same layout as ``base``.
        attn_modes: Same layout as ``base``.
        device: CUDA device for the plan tensors.
        position_ids: As given to ``base``. Non-default positions need a
            position-aware plan, as in :func:`build_usp_plan`; without a ring
            that means a reordered ``base``.

    Returns:
        The sliding-window plan for the calling rank.

    Raises:
        ValueError: The requested sliding-window size is not a positive
            integer, or non-default ``position_ids`` would be ignored by the
            planned path.
    """
    position_ids = _custom_positions(position_ids, base.seq_len)
    plan = _build_sliding_window_plan(
        base,
        sliding_window=sliding_window,
        document_lens=document_lens,
        split_lens=split_lens,
        attn_modes=attn_modes,
        device=device,
        position_ids=position_ids,
    )
    _check_positions_honored(plan, position_ids)
    return plan


def _build_sliding_window_plan(
    base: USPPlan,
    *,
    sliding_window: int,
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    device: torch.device,
    position_ids: Int[Tensor, " tokens"] | None = None,
) -> USPPlan:
    """:func:`build_sliding_window_plan` without the ``position_ids`` check."""
    window = validate_window(sliding_window)
    groups, config = base.groups, base.config
    seq_len = base.seq_len
    given_positions = position_ids
    common = dict(
        config=config,
        groups=groups,
        seq_len=seq_len,
        softmax_scale=base.softmax_scale,
        sliding_window=window,
    )
    if all(mode != "causal" for mode in attn_modes):
        # Every token sees its whole span: the window changes nothing.
        return base

    if groups.ring_size == 1 and base.token_permutation is None:
        # Natural token order: FA4's local attention or the windowed mask in
        # sequence order. A reordered base (vision groups, or a given
        # token_permutation) takes the position-aware path below instead.
        if config.native_plain_masks and all(m == "causal" for m in attn_modes):
            bounds = torch.tensor([0, *document_lens], dtype=torch.int64).cumsum(0)
            return USPPlan(
                **common,
                token_permutation=None,
                restore_permutation=None,
                step_meta=None,
                rank_plan=None,
                fa4_mask=None,
                native_mask="causal",
                native_cu_seqlens=(
                    bounds.to(device=device, dtype=torch.int32)
                    if len(document_lens) > 1
                    else None
                ),
                native_max_seqlen=max(document_lens),
            )
        interval = (
            build_interval_plan(
                document_lens,
                split_lens,
                attn_modes,
                device=device,
                sliding_window=window,
            )
            if config.native_plain_masks
            and len(split_lens) > 1
            and _use_interval_masks(config, device)
            else None
        )
        if interval is not None:
            return USPPlan(
                **common,
                token_permutation=None,
                restore_permutation=None,
                step_meta=None,
                rank_plan=None,
                fa4_mask=None,
                interval_plan=interval,
            )
        return USPPlan(
            **common,
            token_permutation=None,
            restore_permutation=None,
            step_meta=None,
            rank_plan=None,
            fa4_mask=build_fa4_sparse_mask(
                document_lens,
                split_lens,
                attn_modes,
                device=device,
                sliding_window=window,
            ),
        )

    if groups.ring_size > 1:
        halo = _halo_plan(
            base,
            window=window,
            document_lens=document_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            device=device,
            position_ids=position_ids,
            max_halo_fraction=1.0 if config.kv_exchange == "gather" else 0.5,
        )
        if halo is not None:
            return halo
    aux, position_ids, segment_end = _sharded_aux(
        base, document_lens, split_lens, attn_modes, device, position_ids
    )
    perm = base.token_permutation
    aux_plan = build_precomputed_cross_qkv_aux_plan(
        global_aux_tensors=aux,
        rank=groups.ring_rank,
        world_size=groups.ring_size,
        local_len=seq_len // groups.ring_size,
        global_position_ids=position_ids,
        use_direct_block_sparse=True,
        sliding_window=window,
        global_segment_end=segment_end,
    )
    mask_mod = cross_qkv_sparse_mask_mod(window)
    ranges = (
        _position_ranges(
            document_lens,
            split_lens,
            attn_modes,
            perm,
            given_positions,
            device=device,
            sliding_window=window,
        )
        if base.config.column_range_masks
        else None
    )
    if ranges is not None:
        use_position_ranges(aux_plan, *ranges)
        mask_mod = POSITION_RANGE_MASK_MOD
    step_meta = Flash4RingStepMeta(
        mask_mod=mask_mod,
        precomputed_aux_plan=aux_plan,
        softmax_scale=base.softmax_scale,
    )
    return USPPlan(
        **common,
        token_permutation=perm,
        restore_permutation=base.restore_permutation,
        step_meta=step_meta,
        rank_plan=None,
        fa4_mask=None,
    )


class _SingleStepAttention(torch.autograd.Function):
    """The one step of a ring of size one: a position-aware block-sparse FA4 call."""

    @staticmethod
    def forward(ctx, q, k, v, step_meta, deterministic):
        out, lse = _ring_step_fwd(q, k, v, 0, step_meta)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.step_meta, ctx.deterministic = step_meta, deterministic
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _ring_step_bwd(
            q,
            k,
            v,
            out,
            dout.contiguous(),
            lse,
            0,
            ctx.step_meta,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None


_SYMM_MEM_WARNED = False


def _warn_symm_mem_once() -> None:
    global _SYMM_MEM_WARNED
    if not _SYMM_MEM_WARNED:
        _SYMM_MEM_WARNED = True
        warnings.warn(
            "The symm_mem* ring transports are experimental: all three variants "
            "deadlock reproducibly in the backward at ring 2 x ulysses 4 with 128k "
            "tokens on 8x H100. Use transport='nccl' (default) or 'nccl4py'; they "
            "measured as fast.",
            stacklevel=3,
        )


def usp_attention(
    q: Float[Tensor, " batch local_tokens num_heads head_dim"],
    k: Float[Tensor, " batch local_tokens kv_heads head_dim"],
    v: Float[Tensor, " batch local_tokens kv_heads head_dim"],
    plan: USPPlan,
    *,
    return_lse: bool = False,
) -> (
    Float[Tensor, " batch local_tokens num_heads head_dim"]
    | tuple[
        Float[Tensor, " batch local_tokens num_heads head_dim"],
        Float[Tensor, " batch local_tokens num_heads"],
    ]
):
    """Run Ulysses x ring FA4 attention on local shards.

    Args:
        q: Local query shard, ``(batch, plan.local_len, heads, head_dim)``.
        k: Local key shard, ``(batch, plan.local_len, kv_heads, head_dim)``.
        v: Local value shard, same shape as ``k``.
        plan: Plan from :func:`build_usp_plan`.
        return_lse: Also return the log-sum-exp, ``(batch, local_len, heads)``.
            The LSE is not differentiable on the ring paths (block-sparse or
            native striped ring steps), the single position-aware step, the
            KV gather (halo) or the interval path: a loss through it gets no
            gradient there. The single-GPU block-sparse path passes its
            gradient to FA4 when the installed FA4's backward takes one. The
            dense ring, generation ring and dense gather paths do not return
            an LSE.

    Returns:
        The attention output for the local shard in the input layout, and the
        LSE when ``return_lse`` is set.

    Raises:
        ValueError: If ``plan.config.deterministic`` is set, a backward will
            run, and the plan's path cannot run a deterministic backward
            (dense ring, generation ring, dense gather); or the plan needs FA4
            block sparsity at a head_dim where the GPU has none (above 128 on
            H100 with a backward).
    """
    groups = plan.groups
    config = plan.config
    ulysses = groups.ulysses_group
    if config.transport.startswith("symm_mem"):
        _warn_symm_mem_once()
    _check_plan_supported(
        plan,
        device=q.device,
        head_dim=q.shape[-1],
        head_dim_v=v.shape[-1],
        backward=torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)),
    )

    # Sequence-sharded, all heads -> ring-local sequence, heads / ulysses_size.
    q, k, v = (
        seq_all_to_all(t, ulysses, scatter_dim=2, gather_dim=1) for t in (q, k, v)
    )

    if plan.halo is not None:
        result = halo_attention(
            q,
            k,
            v,
            plan.halo,
            group=groups.ring_group,
            softmax_scale=plan.softmax_scale,
            deterministic=config.deterministic,
            return_lse=return_lse,
            regather=config.gather_regather,
            overlap=config.halo_overlap,
        )
    elif plan.generation_ring is not None:
        if return_lse:
            raise NotImplementedError(
                "return_lse is not supported with generation_ring plans."
            )
        result = generation_ring_attention(
            q,
            k,
            v,
            group=groups.ring_group,
            plan=plan.generation_ring,
            softmax_scale=plan.softmax_scale,
        )
    elif plan.dense_gather is not None:
        if return_lse:
            raise NotImplementedError(
                "return_lse is not supported with dense_gather plans."
            )
        result = dense_gather_attention(
            q,
            k,
            v,
            group=groups.ring_group,
            plan=plan.dense_gather,
            softmax_scale=plan.softmax_scale,
            regather=config.gather_regather,
        )
    elif plan.dense_ring is not None:
        if return_lse:
            raise NotImplementedError(
                "return_lse is not supported with dense_ring plans."
            )
        result = dense_ring_attention(
            q,
            k,
            v,
            group=groups.ring_group,
            plan=plan.dense_ring,
            softmax_scale=plan.softmax_scale,
        )
    elif plan.step_meta is not None and groups.ring_size == 1:
        # One position-aware step (vision groups stored in tile-friendly order).
        out, lse = _SingleStepAttention.apply(
            q, k, v, plan.step_meta, config.deterministic
        )
        result = (out, lse) if return_lse else out
    elif plan.step_meta is not None:
        # Without overlap, transports run on the compute stream (serialized).
        kv_stream, grad_stream = (
            plan.comm_streams() if config.use_cuda_comm_stream else (None, None)
        )
        transport = config.transport
        result = ring_flash4_attn(
            q,
            k,
            v,
            process_group=groups.ring_group,
            step_meta=plan.step_meta,
            rank_plan=plan.rank_plan,
            input_is_sharded=True,
            gather_output=False,
            use_cuda_comm_stream=config.use_cuda_comm_stream,
            combine_mode=config.resolved_combine_mode(
                q.device, head_dim=max(q.shape[-1], v.shape[-1])
            ),
            return_lse=return_lse,
            deterministic=config.deterministic,
            use_symm_mem=transport.startswith("symm_mem"),
            use_symm_mem_rotator=transport == "symm_mem_rotator",
            use_symm_mem_direct=transport == "symm_mem_direct",
            use_nccl4py_ring=transport == "nccl4py",
            kv_comm_stream=kv_stream,
            grad_comm_stream=grad_stream,
            fp32_grads=config.fp32_grads,
        )
    elif plan.interval_plan is not None:
        result = interval_attention(
            q,
            k,
            v,
            plan.interval_plan,
            softmax_scale=plan.softmax_scale,
            deterministic=config.deterministic,
            return_lse=return_lse,
        )
    elif plan.native_mask is not None:
        result = flash4_attn_dense(
            q,
            k,
            v,
            causal=plan.native_mask == "causal",
            softmax_scale=plan.softmax_scale,
            deterministic=config.deterministic,
            return_lse=return_lse,
            use_fa3_on_hopper=config.fa3_on_hopper,
            cu_seqlens=plan.native_cu_seqlens,
            max_seqlen=plan.native_max_seqlen,
            sliding_window=plan.sliding_window,
        )
    else:
        assert plan.fa4_mask is not None
        result = flash4_attn(
            q,
            k,
            v,
            plan.fa4_mask,
            softmax_scale=plan.softmax_scale,
            return_lse=return_lse,
            deterministic=config.deterministic,
        )

    if return_lse:
        out, lse = result
        # Kernels disagree on (B, H, S) vs (B, S, H); align with out's layout.
        lse = _normalize_step_lse(out, lse)
        out = seq_all_to_all(out, ulysses, scatter_dim=1, gather_dim=2)
        lse = seq_all_to_all(lse, ulysses, scatter_dim=1, gather_dim=2)
        return out, lse
    return seq_all_to_all(result, ulysses, scatter_dim=1, gather_dim=2)
