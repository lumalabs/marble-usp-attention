# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SPDX-FileCopyrightText: 2022 the FlashAttention contributors, as shown by third_party_licenses/flash-attention-AUTHORS
# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Online Flash Attention interface (SM90).

Thin wrapper around `OnlineFlashAttentionForwardSm90` that reuses as much of the
upstream `flash_attn.cute` plumbing (tensor conversion, dtype map, head-dim
validation, block-sparse normalization, regular-FA4 reference path) as possible.
Only the online-specific compile/invocation path and a handful of SM90
tile-sizing constants are spelled out locally, because older installs of
flash-attn-4 do not expose `_tile_size_fwd_sm90` / `FwdConfig` / `cache_utils`
as public symbols.

Scope of this module: SM90 dense attention with BSHD layout, optionally with
block sparsity and an optional CuTe DSL `mask_mod`. Varlen, paged-KV, split-KV,
score_mod, learnable sinks, and GQA packing are still out of scope — the kernel itself
supports most of them through inheritance from `FlashAttentionForwardSm90`;
plumb them through this wrapper if you need them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cutlass.cute as cute
import torch
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_config,
    to_cute_block_sparse_tensors,
)
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.interface import (
    _flash_attn_fwd as _upstream_flash_attn_fwd,
)
from flash_attn.cute.interface import (
    _get_device_arch,
    _validate_tensor,
    maybe_contiguous,
    torch2cute_dtype_map,
)

try:
    from flash_attn.cute.block_sparsity import (
        get_sparse_q_block_size,  # type: ignore[attr-defined]
    )
except ImportError:  # older flash-attn-4 does not export this helper
    from flash_attn.cute.block_sparsity import (
        ceildiv,  # older versions still have ceildiv
    )

    def get_sparse_q_block_size(
        tensors: Optional[BlockSparseTensorsTorch],
        seqlen_q: int,
    ) -> Optional[int]:
        if tensors is None:
            return None
        if tensors.block_size is not None:
            return tensors.block_size[0]
        num_m_blocks = tensors.mask_block_idx.shape[2]
        min_block_size = ceildiv(seqlen_q, num_m_blocks)
        max_block_size = (
            seqlen_q if num_m_blocks == 1 else (seqlen_q - 1) // (num_m_blocks - 1)
        )
        if min_block_size != max_block_size:
            return None
        return min_block_size


try:
    from flash_attn.cute.testing import is_fake_mode  # type: ignore[attr-defined]
except ImportError:  # older flash-attn-4 has no fake-mode tracing

    def is_fake_mode() -> bool:
        return False


from usp_attention.online_flash.online_flash_fwd_sm90 import (
    OnlineFlashAttentionForwardSm90,
)
from usp_attention.types import ONLINE_MAX_HEAD_DIM

# ---------------------------------------------------------------------------
# Local SM90 tile-sizing (vendored because older flash-attn-4 doesn't export
# _tile_size_fwd_sm90). Values mirror the SM90 branch of upstream FwdConfig.
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class _OnlineFwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _tile_size_online_sm90(
    head_dim: int,
    head_dim_v: int,
    causal: bool,
    *,
    local: bool = False,
    sparse_block_size_q: Optional[int] = None,
) -> _OnlineFwdConfig:
    # SM90. The online kernel inherits from FlashAttentionForwardSm90, so we pick
    # upstream's SM90 defaults here (narrower tile_n for local attention above
    # head_dim 128, as upstream does). Head-dim 128 is the common case;
    # wider heads narrow tile_n to fit shared memory; smaller head_dims can use
    # a larger tile_m.
    del causal  # not currently used for sizing, kept for signature parity
    if max(head_dim, head_dim_v) > 192:
        # Upstream FA4 SM90 values (tile_size_fwd_sm90): fits 227 KiB with
        # 2 K/V stages now that the O state bypasses shared memory.
        cfg = _OnlineFwdConfig(
            m_block_size=128,
            n_block_size=64 if local else 80,
            mma_pv_is_rs=True,
            intra_wg_overlap=True,
        )
    elif max(head_dim, head_dim_v) > 128:
        cfg = _OnlineFwdConfig(
            m_block_size=128,
            n_block_size=96 if local else (128 if head_dim_v <= 128 else 112),
            mma_pv_is_rs=True,
            intra_wg_overlap=True,
        )
    elif max(head_dim, head_dim_v) >= 128:
        cfg = _OnlineFwdConfig(
            m_block_size=128,
            n_block_size=128,
            mma_pv_is_rs=True,
            intra_wg_overlap=True,
        )
    else:
        cfg = _OnlineFwdConfig(
            m_block_size=192,
            n_block_size=128,
            mma_pv_is_rs=True,
            intra_wg_overlap=True,
        )
    # If the caller's sparse Q block size is known and smaller than our default
    # tile_m, shrink tile_m to match — the kernel requires tile_m to divide the
    # sparse q block so block-sparse metadata indexing lines up.
    if sparse_block_size_q is not None and sparse_block_size_q < cfg.m_block_size:
        cfg = _OnlineFwdConfig(
            m_block_size=sparse_block_size_q,
            n_block_size=cfg.n_block_size,
            mma_pv_is_rs=cfg.mma_pv_is_rs,
            intra_wg_overlap=cfg.intra_wg_overlap,
        )
    return cfg


def _validate_online_head_dims(head_dim: int, head_dim_v: int, alignment: int) -> None:
    # Multiples of 8, like SM90 FA4; at most ONLINE_MAX_HEAD_DIM (shared memory).
    assert head_dim % alignment == 0, f"head_dim must be a multiple of {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be a multiple of {alignment}"
    if max(head_dim, head_dim_v) > ONLINE_MAX_HEAD_DIM:
        raise ValueError(
            f"The online kernel supports head_dim <= {ONLINE_MAX_HEAD_DIM}, got "
            f"{head_dim=} {head_dim_v=}."
        )


# ---------------------------------------------------------------------------
# Compile cache. Upstream uses `flash_attn.cute.cache_utils.get_jit_cache("fwd")`,
# which persists across processes; a plain dict is fine for our use case
# (per-process) and avoids depending on a helper that older installs do not ship.
# ---------------------------------------------------------------------------
_compile_cache: dict = {}


def _online_flash_attn_fwd_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[torch.Tensor],
    max_: Optional[torch.Tensor],
    sum_: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    is_online_first: bool,
    is_online_last: bool,
    return_lse: bool,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch],
    mask_mod=None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """SM90 online flash attention forward.

    Returns ``(out, lse, max_, sum_)``. ``lse`` is populated on the final call
    and ``None`` otherwise.
    """
    if return_lse and not is_online_last:
        # The kernel writes the LSE only when finalizing; earlier calls write
        # the running max / sum instead.
        raise ValueError(
            "return_lse=True requires is_online_last=True: the online kernel "
            "only writes the LSE on the last call (use max_ and sum_ for the "
            "running state of earlier calls)."
        )
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]

    assert q.dim() == 4, "q must be BSHD"
    batch_size, seqlen_q, num_head, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype
    assert num_head % num_head_kv == 0
    qhead_per_kvhead = num_head // num_head_kv

    arch = _get_device_arch()
    assert arch // 10 == 9, (
        f"OnlineFlashAttentionForwardSm90 only supports SM 9.x, got arch={arch}"
    )

    alignment = 16 // q.element_size()
    _validate_online_head_dims(head_dim, head_dim_v, alignment)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    device = q.device
    out_shape = (batch_size, seqlen_q, num_head, head_dim_v)
    lse_shape = (batch_size, num_head, seqlen_q)

    # The accumulation buffer is always fp32; the last call reads it and writes the
    # normalized output in q.dtype to a new tensor (see o_final below).
    if out is None:
        out = torch.empty(*out_shape, dtype=torch.float32, device=device)
    else:
        _validate_tensor(out, "out", out_shape, torch.float32, device)

    if is_online_first:
        assert max_ is None and sum_ is None, (
            "max_/sum_ must be None on the first online call; they are allocated internally"
        )
        max_ = torch.empty(lse_shape, dtype=torch.float32, device=device)
        sum_ = torch.empty(lse_shape, dtype=torch.float32, device=device)
    else:
        assert max_ is not None and sum_ is not None, (
            "max_/sum_ must be provided on non-first online calls"
        )
        _validate_tensor(max_, "max_", lse_shape, torch.float32, device)
        _validate_tensor(sum_, "sum_", lse_shape, torch.float32, device)

    if is_online_last:
        if lse is None:
            lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
        else:
            _validate_tensor(lse, "lse", lse_shape, torch.float32, device)
    else:
        lse = None

    dtype = torch2cute_dtype_map[q.dtype]

    local = window_size_left is not None or window_size_right is not None

    use_block_sparsity = block_sparse_tensors is not None
    sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
    fwd_cfg = _tile_size_online_sm90(
        head_dim,
        head_dim_v,
        bool(causal),
        local=local,
        sparse_block_size_q=sparse_q,
    )
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    intra_wg_overlap = fwd_cfg.intra_wg_overlap
    num_threads = 384

    # Block-sparse normalization — mirrors the SM90 branch of upstream
    # `_flash_attn_fwd`. `q_stage=1` on SM90 because q_stage != 1 is an SM100-only path.
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if use_block_sparsity:
        assert block_sparse_tensors is not None  # narrow for type checker
        normalized = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=1,
        )
        # FA4 <= 4.0.0b21 returns a plain 3-tuple; later releases a NamedTuple
        # with an extra kv_subtile_factor field.
        if hasattr(normalized, "tensors"):
            normalized_block_sparse_tensors = normalized.tensors
            block_sparse_broadcast_pattern = normalized.broadcast_pattern
            q_subtile_factor = normalized.q_subtile_factor
        else:
            (
                normalized_block_sparse_tensors,
                block_sparse_broadcast_pattern,
                q_subtile_factor,
            ) = normalized

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # The last call stores the normalized output directly in q.dtype; a resumed
    # last call reads the fp32 state from `out` (mO_state).
    o_final = (
        torch.empty(out_shape, dtype=q.dtype, device=device) if is_online_last else out
    )
    o_state = out if is_online_last and not is_online_first else None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        bool(causal),
        lse is None,
        is_online_first,
        is_online_last,
        window_size_left is not None,
        window_size_right is not None,
        tile_m,
        tile_n,
        num_threads,
        arch,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        q_subtile_factor,
        mask_mod,
        aux_tensors is not None,
    )

    if compile_key not in _compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, o_final)
        ]
        o_state_tensor = to_cute_tensor(o_state) if o_state is not None else None
        lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None
        max_tensor = to_cute_tensor(max_, assumed_align=4)
        sum_tensor = to_cute_tensor(sum_, assumed_align=4)

        cute_sparse_tensors = (
            to_cute_block_sparse_tensors(normalized_block_sparse_tensors)
            if normalized_block_sparse_tensors is not None
            else None
        )

        cute_aux_tensors = (
            [to_cute_tensor(t, assumed_align=4) for t in aux_tensors]
            if aux_tensors is not None
            else None
        )

        fa_fwd = OnlineFlashAttentionForwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=bool(causal),
            is_local=local,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=2,
            num_threads=num_threads,
            Q_in_regs=False,
            intra_wg_overlap=intra_wg_overlap,
            mma_pv_is_rs=mma_pv_is_rs,
            mask_mod=mask_mod,
            score_mod=None,
            has_aux_tensors=aux_tensors is not None,
            q_subtile_factor=q_subtile_factor,
            paged_kv_non_tma=False,
        )

        _compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            max_tensor,
            sum_tensor,
            is_online_first,
            is_online_last,
            softmax_scale,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # page_table
            window_size_left,
            window_size_right,
            None,  # learnable_sink
            cute_sparse_tensors,
            cute_aux_tensors,
            o_state_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        # Kernel expects the BlockSparseTensors fields of the normalized
        # torch-side struct at invocation time (flash-attn-4 4.0.0b21 pattern).
        sparse_invoke = (
            (
                normalized_block_sparse_tensors.mask_block_cnt,
                normalized_block_sparse_tensors.mask_block_idx,
                normalized_block_sparse_tensors.full_block_cnt,
                normalized_block_sparse_tensors.full_block_idx,
                normalized_block_sparse_tensors.cu_total_m_blocks,
                normalized_block_sparse_tensors.cu_block_idx_offsets,
                normalized_block_sparse_tensors.dq_write_order,
                normalized_block_sparse_tensors.dq_write_order_full,
            )
            if normalized_block_sparse_tensors is not None
            else None
        )
        _compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            o_final.detach(),
            lse,
            max_,
            sum_,
            softmax_scale,
            None,
            None,
            None,
            None,
            None,
            window_size_left,
            window_size_right,
            None,
            sparse_invoke,
            aux_tensors,
            o_state.detach() if o_state is not None else None,
        )

    return o_final, lse, max_, sum_


def online_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    max_: Optional[torch.Tensor] = None,
    sum_: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[Optional[int], Optional[int]] = (None, None),
    is_online_first: bool = False,
    is_online_last: bool = False,
    return_lse: bool = False,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    mask_mod=None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Online flash attention forward (SM90).

    On the first call, pass ``is_online_first=True`` and leave ``out``/``max_``/``sum_``
    as ``None`` — they are allocated internally and returned. On subsequent calls,
    thread the returned tensors back in as ``out=...``, ``max_=...``, ``sum_=...``.
    On the last call, pass ``is_online_last=True`` to trigger LSE finalization and
    output normalization; the returned ``lse`` is then populated.

    Args:
        q: Query tensor, shape (batch, seqlen_q, num_head, head_dim), fp16/bf16.
        k, v: Key/value tensors, shape (batch, seqlen_k, num_head_kv, head_dim[_v]).
        out: Running output accumulator. ``None`` on the first call.
        max_: Running row-max accumulator (fp32). ``None`` on the first call.
        sum_: Running row-sum accumulator (fp32). ``None`` on the first call.
        lse: Optional pre-allocated LSE buffer (fp32), written on the last
            call and ignored on earlier ones.
        softmax_scale: Scale applied to QK^T before softmax. Defaults to 1/sqrt(head_dim).
        causal: If True, apply a causal mask.
        window_size: Optional (left, right) sliding-window size.
        is_online_first: Mark this as the first KV chunk (allocates max/sum).
        is_online_last: Mark this as the last KV chunk (finalizes output and LSE).
        return_lse: Kept for parity with ``flash_attn_func``; the last call
            always returns the LSE. Must be False on earlier calls, which
            write only the running max / sum.
        block_sparse_tensors: Optional block-sparse descriptor. Same
            ``BlockSparseTensorsTorch`` used by the upstream non-online interface.
            Must be consistent across online chunks (compile cache is keyed on
            the sparse broadcast pattern and tile dims).

    Returns:
        (out, lse, max_, sum_). ``lse`` is ``None`` unless ``is_online_last``
        is set.

    Raises:
        ValueError: If ``return_lse`` is set on a call that is not the last.
    """
    return _online_flash_attn_fwd_sm90(
        q,
        k,
        v,
        out=out,
        max_=max_,
        sum_=sum_,
        lse=lse,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        is_online_first=is_online_first,
        is_online_last=is_online_last,
        return_lse=return_lse,
        block_sparse_tensors=block_sparse_tensors,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
    )


def online_flash_attn_chunks(
    q: torch.Tensor,
    k_chunks,
    v_chunks,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[Optional[int], Optional[int]] = (None, None),
    return_lse: bool = False,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Convenience: run online flash attention over a pre-split list of KV chunks.

    Walks the chunks in order, threading ``out``/``max_``/``sum_`` between calls
    and marking the first/last chunks appropriately. Returns ``(out, lse)``.

    ``block_sparse_tensors``, when provided, is reused on every chunk — it is
    the caller's responsibility to ensure the sparse mask describes the full
    (packed) Q x K space consistently across chunks.
    """
    k_chunks = list(k_chunks)
    v_chunks = list(v_chunks)
    assert len(k_chunks) == len(v_chunks) and len(k_chunks) >= 1
    num_chunks = len(k_chunks)

    out: Optional[torch.Tensor] = None
    max_: Optional[torch.Tensor] = None
    sum_: Optional[torch.Tensor] = None
    lse_out: Optional[torch.Tensor] = None
    for i, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
        is_first = i == 0
        is_last = i == num_chunks - 1
        out, lse_out, max_, sum_ = online_flash_attn_func(
            q,
            k_chunk,
            v_chunk,
            out=out,
            max_=max_,
            sum_=sum_,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            is_online_first=is_first,
            is_online_last=is_last,
            return_lse=return_lse and is_last,
            block_sparse_tensors=block_sparse_tensors,
        )
    assert out is not None  # loop runs at least once (len >= 1)
    return out, lse_out


def offline_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[Optional[int], Optional[int]] = (None, None),
    return_lse: bool = True,
):
    """Reference: regular (non-online) FA4 forward via the upstream interface.

    Provided here so callers can diff online-chunked output against a single
    full-sequence FA4 call without reaching into `flash_attn.cute` directly.
    Returns whatever `_flash_attn_fwd` returns, normalized to `(out, lse)` —
    older installs return `(out, lse)`, newer installs may return extra values.
    """
    result = _upstream_flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        return_lse=return_lse,
    )
    out, lse = result[0], result[1]
    return out, lse
