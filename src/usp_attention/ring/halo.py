# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Mask-aware KV gather: bring in only the remote KV blocks a rank's queries see.

A ring rotates every other rank's KV shard past every rank, and an all-gather
(FlashMask's context parallelism) copies all of it. A packed mask usually needs
much less: a sliding-window query only sees keys within the window (and its
own bidirectional segment), and a document's queries only see that document.
At plan time every rank classifies, for each other ring rank, which 128-token
KV blocks its queries can see (the cross Q/KV block classification, windowed
by :mod:`usp_attention.sliding_window` for local layers), and which of its own
blocks the others need. At run time one variable-size all-to-all over the ring
group brings in this halo, and a single block-sparse FA4 call attends over
``local KV ++ halo``: no per-step merges, so the result has single-GPU
accuracy. The backward returns the halo's dK / dV to their owners with the
reverse all-to-all and adds them in fp32.

For sliding-window layers with a contiguous-chunk token order
(``USPConfig(balance="chunks")``) the halo is about one window per chunk. For
global layers it is at most the all-gather, and a single kernel pass replaces
the ring's per-step calls (``USPConfig(kv_exchange="gather")``).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention.fa4.flash4_block_sparse import (
    sparse_kv_block_factor,
    sparse_q_block_factor,
)
from usp_attention.ring.block_sparse import build_block_sparse_from_cross_qkv_aux
from usp_attention.ring.fp32_grad import Fp32GradState, convert_dkv, fp32_grad_step
from usp_attention.ring.ring_planning import build_cross_qkv_step_aux_tensors
from usp_attention.sliding_window import BLOCK_EMPTY, windowed_cross_blocks

# Never equal to a real document id (those start at 1): padding keys stay masked.
_PAD_DOCUMENT = -1


@dataclass(kw_only=True, slots=True)
class HaloPlan:
    """Per-rank metadata for :func:`halo_attention`.

    Attributes:
        local_len: Tokens per ring rank.
        send_index: Local token indices sent to other ring ranks, grouped by
            destination in ring-rank order.
        send_counts: Tokens sent to each ring rank.
        recv_counts: Tokens received from each ring rank.
        pad: Masked padding keys appended after the halo.
        aux_tensors: The kernel's aux tensors, keys = local ++ halo ++ pad: the
            8 cross Q/KV tensors, or ``[q_pos, kv_lo, kv_hi]`` for a
            column-range ``mask_mod`` (``position_ranges``).
        block_sparse: Forward FA4 block sparsity.
        block_sparse_bwd: Backward FA4 block sparsity.
        mask_mod: Windowed cross Q/KV ``mask_mod``.
        steps: With ``overlap``, the same keys as two online-kernel calls:
            ``(local, halo)``, each with its own padding, aux tensors and block
            sparsity, so the local call can run while the halo arrives.
    """

    local_len: int
    send_index: Tensor
    send_counts: list[int]
    recv_counts: list[int]
    pad: int
    aux_tensors: list[Tensor]
    block_sparse: object
    block_sparse_bwd: object
    mask_mod: Callable
    steps: "tuple[HaloStep, HaloStep] | None" = None

    @property
    def halo_len(self) -> int:
        """Remote keys this rank receives."""
        return sum(self.recv_counts)


@dataclass(kw_only=True, slots=True)
class HaloStep:
    """One online-kernel call of the overlapped halo path: queries over a KV range.

    Attributes:
        pad: Masked padding keys appended after the range's keys.
        aux_tensors: Kernel aux tensors for keys = range ++ pad.
        block_sparse: Forward FA4 block sparsity.
        block_sparse_bwd: Backward FA4 block sparsity.
    """

    pad: int
    aux_tensors: list[Tensor]
    block_sparse: object
    block_sparse_bwd: object


def _visible_kv_blocks(
    aux: list[Tensor],
    positions: Tensor,
    segment_end: Tensor | None,
    *,
    q_rank: int,
    kv_rank: int,
    local_len: int,
    window: int | None,
    block_size: int,
) -> Tensor:
    """KV blocks of ``kv_rank``'s shard that any query of ``q_rank`` can see."""
    step = build_cross_qkv_step_aux_tensors(
        global_aux_tensors=aux,
        q_start=q_rank * local_len,
        q_len=local_len,
        kv_start=kv_rank * local_len,
        kv_len=local_len,
        global_position_ids=positions,
    )
    result = build_block_sparse_from_cross_qkv_aux(
        step, block_size=block_size, types_only=True
    )
    if window is not None:
        kv_seg = segment_end[kv_rank * local_len : (kv_rank + 1) * local_len]
        result = windowed_cross_blocks(result, step, kv_seg, window)
    return (result.block_type != BLOCK_EMPTY).any(dim=0).nonzero().flatten()


def needed_key_masks(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[str],
    *,
    ring_size: int,
    token_permutation: Tensor | None,
    sliding_window: int | None,
) -> list[np.ndarray]:
    """Per ring rank, the stored-order keys any of its queries can see.

    From the layout alone, O(seq_len) per rank: each query's visible keys are
    intervals (its document's prefix, limited by the window for causal pairs;
    the rest of its full / noise span; every group span of its document),
    unioned with a difference array; other noise spans are removed. Exact up
    to the block granularity the exchange uses, and much cheaper than
    classifying every (rank, rank) block pair.
    """
    seq_len = sum(document_lens)
    local = seq_len // ring_size
    window = sliding_window or seq_len
    lens = np.asarray(split_lens, dtype=np.int64)
    modes = np.asarray(attn_modes)
    split_start = np.cumsum(lens) - lens
    # Document of each split: splits never straddle documents.
    doc_end = np.cumsum(np.asarray(document_lens, dtype=np.int64))
    split_doc = np.searchsorted(doc_end, split_start, side="right")
    split_d0 = np.concatenate([[0], doc_end[:-1]])[split_doc]
    tok_split = np.repeat(np.arange(len(lens)), lens)
    q = np.arange(seq_len, dtype=np.int64)
    d0 = split_d0[tok_split]
    start = np.maximum(d0, q - window + 1)
    is_causal = np.isin(modes, ("causal", "group"))[tok_split]
    # Causal and group: windowed prefix; full / noise: plus the whole span.
    lo = np.where(is_causal, start, np.minimum(start, split_start[tok_split]))
    hi = np.where(is_causal, q + 1, (split_start + lens)[tok_split])
    noise = (modes == "noise")[tok_split]
    group = (modes == "group")[tok_split]
    split_noise = modes == "noise"
    split_group = modes == "group"
    doc = split_doc[tok_split]
    perm = (
        np.arange(seq_len)
        if token_permutation is None
        else token_permutation.detach().cpu().numpy()
    )

    def rank_mask(r: int) -> np.ndarray:
        qs = perm[r * local : (r + 1) * local]
        cover = np.bincount(lo[qs], minlength=seq_len + 1) - np.bincount(
            hi[qs], minlength=seq_len + 1
        )
        # Noise spans this rank's queries sit in, and every group span of the
        # documents its group queries belong to, are visible whole.
        own = np.zeros(len(lens), bool)
        own[tok_split[qs[noise[qs]]]] = True
        docs_with_groups = np.zeros(len(document_lens), bool)
        docs_with_groups[doc[qs[group[qs]]]] = True
        own |= split_group & docs_with_groups[split_doc]
        spans = np.flatnonzero(own & (split_noise | split_group))
        cover[split_start[spans]] += 1 << 32
        np.subtract.at(cover, split_start[spans] + lens[spans], 1 << 32)
        covered = np.cumsum(cover[:-1])
        need = ((covered & ((1 << 32) - 1)) > 0) & ~noise
        need |= (covered >> 32) > 0
        return need[perm]

    # Ranks are independent and NumPy releases the GIL on these arrays.
    with ThreadPoolExecutor(max_workers=min(ring_size, 8)) as pool:
        return list(pool.map(rank_mask, range(ring_size)))


def _needed_blocks(
    masks: list[np.ndarray],
    *,
    q_rank: int,
    kv_rank: int,
    local_len: int,
    block_size: int,
) -> Tensor:
    shard = masks[q_rank][kv_rank * local_len : (kv_rank + 1) * local_len]
    return torch.from_numpy(np.flatnonzero(shard.reshape(-1, block_size).any(axis=1)))


def _block_tokens(blocks: Tensor, block_size: int) -> Tensor:
    offsets = torch.arange(block_size, device=blocks.device)
    return (blocks[:, None] * block_size + offsets[None, :]).flatten()


def build_halo_plan(
    *,
    global_aux_tensors: list[Tensor],
    global_position_ids: Tensor,
    global_segment_end: Tensor | None,
    ring_rank: int,
    ring_size: int,
    local_len: int,
    sliding_window: int | None,
    mask_mod: Callable,
    max_halo_fraction: float = 0.5,
    block_size: int = 128,
    needed_masks: list[np.ndarray] | None = None,
    position_ranges: tuple[Tensor, Tensor, Tensor] | None = None,
    overlap: bool = False,
) -> HaloPlan | None:
    """Plan the halo exchange for this ring rank, or ``None`` if it does not pay.

    Every rank derives the same send / receive lists from the global layout, so
    planning needs no communication.

    Args:
        global_aux_tensors: ``[fn_seq, noise_seq, doc_seq]`` in sharded token
            order (rank ``r`` owns ``[r * local_len, (r + 1) * local_len)``).
        global_position_ids: Token positions in the same order.
        global_segment_end: :func:`~usp_attention.sliding_window.segment_end_positions`
            in the same order (unused without a window).
        ring_rank: This rank in the ring group.
        ring_size: Ring group size.
        local_len: Tokens per ring rank (a multiple of ``block_size``).
        sliding_window: Window in keys, ``None`` for a global layer.
        mask_mod: Cross Q/KV ``mask_mod`` (windowed for a sliding window).
        max_halo_fraction: Decline when some rank would receive more than this
            fraction of the ``(ring_size - 1) * local_len`` keys a ring rotates.
        block_size: KV block size of the exchange.
        needed_masks: :func:`needed_key_masks` for this layout and token
            order; skips the per-(rank, rank) block classification.
        position_ranges: Per-token ``(pos, lo, hi)`` int32 in the same order
            (``usp._position_ranges``): the kernel then tests
            ``lo[k] <= pos[q] < hi[k]`` with ``mask_mod`` (a column-range
            ``mask_mod``); padding keys get the empty range. Block sparsity is
            still classified from ``global_aux_tensors``.
        overlap: Also plan the local and halo keys as separate online-kernel
            calls (``HaloPlan.steps``) for :func:`halo_attention`'s overlapped path.

    Returns:
        The plan, or ``None`` when the largest halo exceeds ``max_halo_fraction``.
    """
    kwargs = dict(local_len=local_len, window=sliding_window, block_size=block_size)
    aux, pos, seg = global_aux_tensors, global_position_ids, global_segment_end
    if needed_masks is not None:
        need = [
            [
                _needed_blocks(
                    needed_masks, q_rank=r, kv_rank=o, local_len=local_len,
                    block_size=block_size,
                ).to(aux[0].device)
                if r != o
                else None
                for o in range(ring_size)
            ]
            for r in range(ring_size)
        ]  # fmt: skip
    else:
        need = [
            [
                _visible_kv_blocks(aux, pos, seg, q_rank=r, kv_rank=o, **kwargs)
                if r != o
                else None
                for o in range(ring_size)
            ]
            for r in range(ring_size)
        ]
    largest = max(
        sum(int(b.numel()) for b in row if b is not None) * block_size for row in need
    )
    if largest > max_halo_fraction * (ring_size - 1) * local_len:
        return None

    device = aux[0].device
    empty = torch.empty(0, dtype=torch.int64, device=device)
    recv_tokens = [
        _block_tokens(need[ring_rank][o], block_size) if o != ring_rank else empty
        for o in range(ring_size)
    ]
    send_tokens = [
        _block_tokens(need[o][ring_rank], block_size) if o != ring_rank else empty
        for o in range(ring_size)
    ]
    local_ids = torch.arange(
        ring_rank * local_len, (ring_rank + 1) * local_len, device=device
    )
    halo_ids = torch.cat([o * local_len + t for o, t in enumerate(recv_tokens)])
    kv_multiple = block_size * sparse_kv_block_factor(device)
    q = slice(ring_rank * local_len, (ring_rank + 1) * local_len)
    fn, noise, doc = aux

    # [MLR-02.08] the step helpers import flash_attn.cute, an optional dependency.
    from usp_attention.ring.flash4_ring_attn import (
        PrecomputedStepAux,
        _coarsen_step_block_sparse,
        _step_bwd_block_sparse,
        _step_fwd_block_sparse,
        _StepBlockSparse,
    )

    def kv_step(kv_global: Tensor) -> HaloStep:
        """Aux tensors and block sparsity for this rank's queries over ``kv_global``."""
        pad = (-kv_global.numel()) % kv_multiple

        def keys(t: Tensor, fill: float) -> Tensor:
            return torch.cat([t[kv_global], t.new_full((pad,), fill)]).contiguous()

        cross = [
            fn[q].contiguous(),
            keys(fn, -1),
            noise[q].contiguous(),
            keys(noise, -1),
            doc[q].contiguous(),
            keys(doc, _PAD_DOCUMENT),
            pos[q].contiguous(),
            keys(pos, 0),
        ]
        result = build_block_sparse_from_cross_qkv_aux(
            cross,
            block_size=block_size,
            compact=True,
            with_types=sliding_window is not None,
        )
        if sliding_window is not None:
            result = windowed_cross_blocks(result, cross, keys(seg, -1), sliding_window)

        if position_ranges is not None:
            r_pos, r_lo, r_hi = position_ranges

            def range_keys(t: Tensor) -> Tensor:
                return torch.cat([t[kv_global], t.new_zeros(pad)]).contiguous()

            kernel_aux = [r_pos[q].contiguous(), range_keys(r_lo), range_keys(r_hi)]
        else:
            kernel_aux = cross

        sparse = _StepBlockSparse(
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
        factor = sparse_q_block_factor(device)
        kv_factor = sparse_kv_block_factor(device)
        if factor > 1 or kv_factor > 1:
            sparse = _coarsen_step_block_sparse(sparse, factor, kv_factor)
        step = PrecomputedStepAux(
            step=0,
            kv_owner_rank=ring_rank,
            kv_start=0,
            kv_len=int(kv_global.numel()) + pad,
            aux_tensors=cross,
            block_size=sparse.block_size,
            full_block_cnt=sparse.full_block_cnt,
            full_block_idx=sparse.full_block_idx,
            mask_block_cnt=sparse.mask_block_cnt,
            mask_block_idx=sparse.mask_block_idx,
            bwd_full_block_cnt=sparse.bwd_full_block_cnt,
            bwd_full_block_idx=sparse.bwd_full_block_idx,
            bwd_mask_block_cnt=sparse.bwd_mask_block_cnt,
            bwd_mask_block_idx=sparse.bwd_mask_block_idx,
        )
        return HaloStep(
            pad=pad,
            aux_tensors=kernel_aux,
            block_sparse=_step_fwd_block_sparse(step),
            block_sparse_bwd=_step_bwd_block_sparse(step),
        )

    combined = kv_step(torch.cat([local_ids, halo_ids]))
    return HaloPlan(
        local_len=local_len,
        send_index=torch.cat(send_tokens),
        send_counts=[int(t.numel()) for t in send_tokens],
        recv_counts=[int(t.numel()) for t in recv_tokens],
        pad=combined.pad,
        aux_tensors=combined.aux_tensors,
        block_sparse=combined.block_sparse,
        block_sparse_bwd=combined.block_sparse_bwd,
        mask_mod=mask_mod,
        steps=(kv_step(local_ids), kv_step(halo_ids)) if overlap else None,
    )


def _exchange(
    tensors: tuple[Tensor, ...],
    *,
    send_counts: list[int],
    recv_counts: list[int],
    group: dist.ProcessGroup,
) -> Tensor:
    """All-to-all sequence-first ``(S, ...)`` tensors; returns ``(n, S_recv, ...)``."""
    send = torch.stack(tensors, dim=1).contiguous()
    recv = send.new_empty((sum(recv_counts), *send.shape[1:]))
    dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        group=group,
    )
    return recv.transpose(0, 1)


def _with_halo(local: Tensor, halo: Tensor, pad: int) -> Tensor:
    """``(B, L, ...)`` local ++ ``(S, B, ...)`` halo ++ ``pad`` zero keys."""
    parts = [local, halo.transpose(0, 1)]
    if pad:
        parts.append(local.new_zeros((local.shape[0], pad, *local.shape[2:])))
    return torch.cat(parts, dim=1)


class _HaloAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, plan: HaloPlan, group, softmax_scale, deterministic, regather
    ):
        # [MLR-02.08] flash_attn.cute is an optional, platform-specific dependency.
        from flash_attn.cute.interface import _flash_attn_fwd

        halo = _exchange(
            tuple(t[:, plan.send_index].transpose(0, 1) for t in (k, v)),
            send_counts=plan.send_counts,
            recv_counts=plan.recv_counts,
            group=group,
        )
        k_all = _with_halo(k, halo[0], plan.pad)
        v_all = _with_halo(v, halo[1], plan.pad)
        out, lse, *_ = _flash_attn_fwd(
            q,
            k_all,
            v_all,
            softmax_scale=softmax_scale,
            mask_mod=plan.mask_mod,
            block_sparse_tensors=plan.block_sparse,
            aux_tensors=plan.aux_tensors,
            return_lse=True,
        )
        if regather:
            # Keep only the local KV; the backward exchanges the halo again.
            ctx.save_for_backward(q, k, v, out, lse)
        else:
            ctx.save_for_backward(q, k_all, v_all, out, lse)
        ctx.plan, ctx.group, ctx.regather = plan, group, regather
        ctx.softmax_scale, ctx.deterministic = softmax_scale, deterministic
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        # [MLR-02.08] flash_attn.cute is an optional, platform-specific dependency.
        from flash_attn.cute.interface import _flash_attn_bwd

        q, k_all, v_all, out, lse = ctx.saved_tensors
        plan: HaloPlan = ctx.plan
        if ctx.regather:
            halo = _exchange(
                tuple(t[:, plan.send_index].transpose(0, 1) for t in (k_all, v_all)),
                send_counts=plan.send_counts,
                recv_counts=plan.recv_counts,
                group=ctx.group,
            )
            k_all = _with_halo(k_all, halo[0], plan.pad)
            v_all = _with_halo(v_all, halo[1], plan.pad)
        state = Fp32GradState()
        # With GQA, FA4 accumulates dK / dV in fp32: keep the raw accumulators
        # so the halo's share is exchanged and added in fp32, rounded once.
        with fp32_grad_step(state, is_first=True, is_last=True):
            dq, dk_all, dv_all = _flash_attn_bwd(
                q,
                k_all,
                v_all,
                out,
                dout.contiguous(),
                lse,
                ctx.softmax_scale,
                causal=False,
                softcap=0.0,
                deterministic=ctx.deterministic,
                mask_mod=plan.mask_mod,
                aux_tensors=plan.aux_tensors,
                block_sparse_tensors=plan.block_sparse_bwd,
            )
        n = plan.local_len
        halo = slice(n, n + plan.halo_len)
        batch, _, kv_heads, head_dim = k_all.shape
        if state.dk_raw is not None:
            # Raw accumulators are (B, H_kv, S, D), unscaled.
            raw = [
                t.view(batch, kv_heads, -1, head_dim)
                for t in (state.dk_raw, state.dv_raw)
            ]
            back = _exchange(
                tuple(t[:, :, halo].permute(2, 0, 1, 3) for t in raw),
                send_counts=plan.recv_counts,
                recv_counts=plan.send_counts,
                group=ctx.group,
            )
            local = []
            for t, part in zip(raw, back):
                acc = t[:, :, :n].clone()
                acc.index_add_(2, plan.send_index, part.permute(1, 2, 0, 3))
                local.append(acc.reshape(batch, kv_heads, -1))
            dk = k_all.new_empty((batch, n, kv_heads, head_dim))
            dv = v_all.new_empty((batch, n, kv_heads, v_all.shape[-1]))
            convert_dkv(state, local[0], local[1], dk, dv)
            return dq, dk, dv, None, None, None, None, None
        # MHA: FA4 writes dK / dV in the input dtype; add the halo's in fp32.
        back = _exchange(
            tuple(t[:, halo].transpose(0, 1) for t in (dk_all, dv_all)),
            send_counts=plan.recv_counts,
            recv_counts=plan.send_counts,
            group=ctx.group,
        )
        grads = []
        for full, part in ((dk_all, back[0]), (dv_all, back[1])):
            acc = full[:, :n].float()
            acc.index_add_(1, plan.send_index, part.transpose(0, 1).float())
            grads.append(acc.to(full.dtype))
        return dq, *grads, None, None, None, None, None


def _exchange_async(
    tensors: tuple[Tensor, ...],
    *,
    send_counts: list[int],
    recv_counts: list[int],
    group: dist.ProcessGroup,
) -> Callable[[], Tensor]:
    """Start :func:`_exchange`; the returned callable waits and returns its result."""
    send = torch.stack(tensors, dim=1).contiguous()
    recv = send.new_empty((sum(recv_counts), *send.shape[1:]))
    work = dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=recv_counts,
        input_split_sizes=send_counts,
        group=group,
        async_op=True,
    )

    def wait() -> Tensor:
        work.wait()
        assert send.numel() >= 0  # the closure keeps the send buffer alive until done
        return recv.transpose(0, 1)

    return wait


def _padded(t: Tensor, pad: int) -> Tensor:
    """``(B, S, ...)`` with ``pad`` zero keys appended."""
    if not pad:
        return t
    return torch.cat([t, t.new_zeros((t.shape[0], pad, *t.shape[2:]))], dim=1)


class _HaloOverlapAttention(torch.autograd.Function):
    """Halo attention with the exchange hidden behind compute (Hopper online kernel).

    Forward: start the halo all-to-all, attend over the local keys with the online
    kernel (first call), then over the arrived halo (last call). The online kernel
    keeps the softmax state in fp32 across both calls, so the output matches the
    single-call path. Backward: run the halo keys' call first, send their dK / dV
    home while the local call runs, and add them in fp32 as before.
    """

    @staticmethod
    def forward(ctx, q, k, v, plan: HaloPlan, group, softmax_scale, deterministic):
        # [MLR-02.08] flash_attn.cute and the online kernel are optional dependencies.
        from usp_attention.online_flash import online_flash_attn_func
        from usp_attention.ring.ring_combine import _ensure_lse_bwd_layout

        local, halo_step = plan.steps
        wait = _exchange_async(
            tuple(t[:, plan.send_index].transpose(0, 1) for t in (k, v)),
            send_counts=plan.send_counts,
            recv_counts=plan.recv_counts,
            group=group,
        )
        k_loc, v_loc = _padded(k, local.pad), _padded(v, local.pad)
        out, _, max_, sum_ = online_flash_attn_func(
            q,
            k_loc,
            v_loc,
            softmax_scale=softmax_scale,
            block_sparse_tensors=local.block_sparse,
            mask_mod=plan.mask_mod,
            aux_tensors=local.aux_tensors,
            is_online_first=True,
        )
        halo = wait()
        k_halo = _padded(halo[0].transpose(0, 1).contiguous(), halo_step.pad)
        v_halo = _padded(halo[1].transpose(0, 1).contiguous(), halo_step.pad)
        out, lse, _, _ = online_flash_attn_func(
            q,
            k_halo,
            v_halo,
            out=out,
            max_=max_,
            sum_=sum_,
            softmax_scale=softmax_scale,
            block_sparse_tensors=halo_step.block_sparse,
            mask_mod=plan.mask_mod,
            aux_tensors=halo_step.aux_tensors,
            is_online_last=True,
        )
        lse = _ensure_lse_bwd_layout(lse, out)
        ctx.save_for_backward(q, k_loc, v_loc, k_halo, v_halo, out, lse)
        ctx.plan, ctx.group = plan, group
        ctx.softmax_scale, ctx.deterministic = softmax_scale, deterministic
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        # [MLR-02.08] flash_attn.cute is an optional, platform-specific dependency.
        from flash_attn.cute.interface import _flash_attn_bwd

        from usp_attention.ring.flash4_ring_attn import _add_step_dq

        q, k_loc, v_loc, k_halo, v_halo, out, lse = ctx.saved_tensors
        plan: HaloPlan = ctx.plan
        local, halo_step = plan.steps
        n, halo_len = plan.local_len, plan.halo_len
        dout = dout.contiguous()
        batch, _, kv_heads, head_dim = k_loc.shape
        state = Fp32GradState()

        def bwd(k_s, v_s, step: HaloStep, *, is_first: bool, is_last: bool):
            with fp32_grad_step(state, is_first=is_first, is_last=is_last):
                return _flash_attn_bwd(
                    q,
                    k_s,
                    v_s,
                    out,
                    dout,
                    lse,
                    ctx.softmax_scale,
                    causal=False,
                    softcap=0.0,
                    deterministic=ctx.deterministic,
                    mask_mod=plan.mask_mod,
                    aux_tensors=step.aux_tensors,
                    block_sparse_tensors=step.block_sparse_bwd,
                )

        # Halo keys first: their gradients travel home while the local call runs.
        dq_h, dk_h, dv_h = bwd(k_halo, v_halo, halo_step, is_first=True, is_last=False)
        dq = _add_step_dq(None, dq_h, state, is_last=False)
        gqa = state.dk_raw is not None
        if gqa:
            # Raw fp32 accumulators are (B, H_kv, S, D), unscaled.
            parts = tuple(
                t.view(batch, kv_heads, -1, head_dim)[:, :, :halo_len].permute(
                    2, 0, 1, 3
                )
                for t in (state.dk_raw, state.dv_raw)
            )
        else:
            parts = tuple(t[:, :halo_len].transpose(0, 1) for t in (dk_h, dv_h))
        wait = _exchange_async(
            parts, send_counts=plan.recv_counts, recv_counts=plan.send_counts,
            group=ctx.group,
        )  # fmt: skip
        dq_l, dk_l, dv_l = bwd(k_loc, v_loc, local, is_first=False, is_last=True)
        dq = _add_step_dq(dq, dq_l, state, is_last=True)
        back = wait()
        if gqa:
            grads = []
            for t, part in zip((state.dk_raw, state.dv_raw), back):
                acc = t.view(batch, kv_heads, -1, head_dim)[:, :, :n].clone()
                acc.index_add_(2, plan.send_index, part.permute(1, 2, 0, 3))
                grads.append(acc.reshape(batch, kv_heads, -1))
            dk = k_loc.new_empty((batch, n, kv_heads, head_dim))
            dv = v_loc.new_empty((batch, n, kv_heads, v_loc.shape[-1]))
            convert_dkv(state, grads[0], grads[1], dk, dv)
        else:
            out_grads = []
            for full, part in ((dk_l, back[0]), (dv_l, back[1])):
                acc = full[:, :n].float()
                acc.index_add_(1, plan.send_index, part.transpose(0, 1).float())
                out_grads.append(acc.to(full.dtype))
            dk, dv = out_grads
        return dq.to(q.dtype), dk, dv, None, None, None, None


def halo_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    plan: HaloPlan,
    *,
    group: dist.ProcessGroup,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
    regather: bool = False,
    overlap: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Attention of a ring rank's local shard over its local KV and gathered halo.

    Args:
        q: ``(batch, local_len, heads, head_dim)`` local queries.
        k: ``(batch, local_len, kv_heads, head_dim)`` local keys.
        v: Local values, same shape as ``k``.
        plan: From :func:`build_halo_plan`.
        group: The ring process group.
        softmax_scale: Defaults to ``1 / sqrt(head_dim)``.
        deterministic: Use FA4's deterministic backward.
        return_lse: Also return the ``(batch, heads, local_len)`` log-sum-exp.
        regather: Save only the local KV for the backward and exchange the
            halo again there, instead of keeping the gathered KV per layer.
        overlap: Hide the exchanges behind compute with two online-kernel
            calls (needs a plan built with ``overlap=True``, Hopper, and no
            ``regather``); otherwise the single-call path runs.

    Returns:
        The local attention output, and the LSE when ``return_lse`` is set.
    """
    if (
        overlap
        and plan.steps is not None
        and plan.halo_len > 0
        and not regather
        and torch.cuda.get_device_capability(q.device)[0] == 9  # online kernel: SM90
    ):
        out, lse = _HaloOverlapAttention.apply(
            q, k, v, plan, group, softmax_scale, deterministic
        )
        return (out, lse) if return_lse else out
    out, lse = _HaloAttention.apply(
        q, k, v, plan, group, softmax_scale, deterministic, regather
    )
    return (out, lse) if return_lse else out
