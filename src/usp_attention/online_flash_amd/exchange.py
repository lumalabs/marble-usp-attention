# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Direct Q/Ulysses and KV/CP redistribution in a single variable-size exchange."""

from __future__ import annotations

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from usp_attention.distributed import USPGroups


@triton.jit
def _pack(
    Q,
    K,
    V,
    Send,
    BS: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    U: tl.constexpr,
    RANK_RING: tl.constexpr,
    BLOCK: tl.constexpr,
    INDEX64: tl.constexpr = False,
):
    peer = tl.program_id(1)
    has_q = peer // U == RANK_RING
    qh = tl.where(has_q, HQ, 0)
    heads = qh + 2 * HK
    prior_q = tl.minimum(tl.maximum(peer - RANK_RING * U, 0), U)
    start = (peer * 2 * HK + prior_q * HQ) * BS * 128
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if INDEX64:
        start = (peer * 2 * HK + prior_q * HQ).to(tl.int64) * BS * 128
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < BS * heads * 128
    d = i % 128
    h = i // 128 % heads
    bs = i // (heads * 128)
    u = peer % U
    is_q = h < qh
    is_k = (h >= qh) & (h < qh + HK)
    is_v = h >= qh + HK
    q = tl.load(Q + (bs * (HQ * U) + u * HQ + h) * 128 + d, valid & is_q, 0)
    k = tl.load(K + (bs * (HK * U) + u * HK + h - qh) * 128 + d, valid & is_k, 0)
    v = tl.load(V + (bs * (HK * U) + u * HK + h - qh - HK) * 128 + d, valid & is_v, 0)
    tl.store(Send + start + i, tl.where(is_q, q, tl.where(is_k, k, v)), valid)


@triton.jit
def _unpack(
    Recv,
    Q,
    K,
    V,
    B: tl.constexpr,
    S: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    U: tl.constexpr,
    WORLD: tl.constexpr,
    RANK_RING: tl.constexpr,
    BLOCK: tl.constexpr,
    INDEX64: tl.constexpr = False,
):
    peer = tl.program_id(1)
    has_q = peer // U == RANK_RING
    qh = tl.where(has_q, HQ, 0)
    heads = qh + 2 * HK
    prior_q = tl.minimum(tl.maximum(peer - RANK_RING * U, 0), U)
    start = (peer * 2 * HK + prior_q * HQ) * B * S * 128
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if INDEX64:
        start = (peer * 2 * HK + prior_q * HQ).to(tl.int64) * B * S * 128
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < B * S * heads * 128
    d = i % 128
    h = i // 128 % heads
    bs = i // (heads * 128)
    b, s = bs // S, bs % S
    x = tl.load(Recv + start + i, valid, 0)
    qi = ((b * (U * S) + (peer % U) * S + s) * HQ + h) * 128 + d
    ki = ((b * (WORLD * S) + peer * S + s) * HK + h - qh) * 128 + d
    vi = ((b * (WORLD * S) + peer * S + s) * HK + h - qh - HK) * 128 + d
    tl.store(Q + qi, x, valid & (h < qh))
    tl.store(K + ki, x, valid & (h >= qh) & (h < qh + HK))
    tl.store(V + vi, x, valid & (h >= qh + HK))


def gather_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, groups: USPGroups
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BSHD shards to Ulysses Q and full-CP KV, preserving source-rank order.

    Raises:
        ValueError: Inputs are not compatible bf16 BSHD tensors with D=128.
    """
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("Q/K/V must be four-dimensional BSHD tensors")
    if (
        any(x.device != q.device or x.dtype != torch.bfloat16 for x in (q, k, v))
        or not q.is_cuda
    ):
        raise ValueError("Q/K/V must be bf16 on the same GPU")
    if (
        min(q.shape) < 1
        or min(k.shape) < 1
        or q.shape[-1] != 128
        or k.shape[:2] != q.shape[:2]
        or k.shape[-1] != 128
        or k.shape != v.shape
        or q.shape[2] % k.shape[2]
        or q.shape[2] % groups.ulysses_size
        or k.shape[2] % groups.ulysses_size
    ):
        raise ValueError(
            "Q/K/V must share batch, sequence and D=128, with divisible GQA/Ulysses heads"
        )
    q, k, v = (x.contiguous() for x in (q, k, v))
    b, s, heads, _ = q.shape
    u, world = groups.ulysses_size, groups.cp_size
    hq, hk = heads // u, k.shape[2] // u
    splits = [
        b * s * 128 * (2 * hk + (hq if peer // u == groups.ring_rank else 0))
        for peer in range(world)
    ]
    send = torch.empty(sum(splits), dtype=q.dtype, device=q.device)
    recv = torch.empty_like(send)
    grid = (triton.cdiv(b * s * (hq + 2 * hk) * 128, 4096), world)
    index64 = max(q.numel(), k.numel(), sum(splits)) >= 2**31
    _pack[grid](
        q, k, v, send, b * s, hq, hk, u, groups.ring_rank, 4096, INDEX64=index64
    )
    dist.all_to_all_single(recv, send, splits, splits, group=groups.cp_group)
    qo = torch.empty((b, u * s, hq, 128), dtype=q.dtype, device=q.device)
    ko = torch.empty((b, world * s, hk, 128), dtype=k.dtype, device=k.device)
    vo = torch.empty_like(ko)
    _unpack[grid](
        recv,
        qo,
        ko,
        vo,
        b,
        s,
        hq,
        hk,
        u,
        world,
        groups.ring_rank,
        4096,
        INDEX64=index64,
    )
    return qo, ko, vo
