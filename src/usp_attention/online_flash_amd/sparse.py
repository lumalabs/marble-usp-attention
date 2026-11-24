# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Conservative BCSR plans for the AMD kernel's 384 x 64 tiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from usp_attention.ring.block_sparse import build_block_sparse_from_cross_qkv_aux


@dataclass(kw_only=True, slots=True)
class SparsePlan:
    """Trusted device plan, shared by all heads/batches with this packed layout.

    Construct with build_sparse_plan; mutating counts/indices can cause invalid
    device reads. The plan is independent of Q/K/V and reusable across layers.
    """

    counts: torch.Tensor
    indices: torch.Tensor
    types: torch.Tensor
    q_aux: torch.Tensor
    kv_aux: torch.Tensor
    mask_offsets: torch.Tensor
    mask_bits: torch.Tensor
    q_order: torch.Tensor

    @property
    def tensors(self) -> list[torch.Tensor]:
        return [
            self.counts,
            self.indices,
            self.types,
            self.q_aux,
            self.kv_aux,
            self.mask_offsets,
            self.mask_bits,
            self.q_order,
        ]


@triton.jit
def _partial_mask_bits(
    Q, K, Tiles, Bits, NQ: tl.constexpr, NK: tl.constexpr, KB: tl.constexpr
):
    tile = tl.program_id(0)
    qr = tl.program_id(1) * 32 + tl.arange(0, 32)
    key = tl.arange(0, 64)
    flat = tl.load(Tiles + tile)
    qi = (flat // KB) * 384 + qr
    ki = (flat % KB) * 64 + key
    qs = tl.load(Q + qi, qi < NQ, -1)
    qn = tl.load(Q + NQ + qi, qi < NQ, -1)
    qd = tl.load(Q + 2 * NQ + qi, qi < NQ, -1)
    qp = tl.load(Q + 3 * NQ + qi, qi < NQ, -1)
    ks = tl.load(K + ki, ki < NK, -1)
    kn = tl.load(K + NK + ki, ki < NK, -1)
    kd = tl.load(K + 2 * NK + ki, ki < NK, -1)
    kp = tl.load(K + 3 * NK + ki, ki < NK, -1)
    legal = (qp[:, None] >= kp[None, :]) | (
        (qs[:, None] >= 0) & (qs[:, None] == ks[None, :])
    )
    legal &= (kn[None, :] < 0) | (qn[:, None] == kn[None, :])
    legal &= qd[:, None] == kd[None, :]
    legal &= (qi[:, None] < NQ) & (ki[None, :] < NK)
    words = tl.where(legal, 1 << (key[None, :] % 32), 0).to(tl.uint32)
    lo = tl.sum(tl.where(key[None, :] < 32, words, 0), 1)
    hi = tl.sum(tl.where(key[None, :] >= 32, words, 0), 1)
    tl.store(Bits + tile * 768 + qr * 2, lo)
    tl.store(Bits + tile * 768 + qr * 2 + 1, hi)


def build_sparse_plan(q_aux: torch.Tensor, kv_aux: torch.Tensor) -> SparsePlan:
    """Build from int32 (4, tokens) segment/noise/document/original-position tracks.

    Empty tiles are omitted; full tiles bypass token predicates. Partial tiles cache exact bitmaps (3 KiB per partial tile), shared across
    heads and reused across layers. The planning predicate is (causal OR same nonnegative segment) AND noise isolation AND document
    equality. Original positions make nonmonotonic PTRR and ring windows safe.
    """
    for aux in (q_aux, kv_aux):
        if aux.ndim != 2 or aux.shape[0] != 4 or aux.shape[1] == 0:
            raise ValueError("metadata must have shape (4, positive token count)")
        if aux.dtype != torch.int32 or aux.device != q_aux.device:
            raise ValueError("metadata must be int32 on the same device")
    q_aux, kv_aux = q_aux.contiguous(), kv_aux.contiguous()
    cross = [track for pair in zip(q_aux, kv_aux) for track in pair]
    classified = build_block_sparse_from_cross_qkv_aux(
        cross, block_size=(384, 64), types_only=True
    )
    types = classified.block_type.to(torch.int32).contiguous()
    active = types != 0
    counts = active.sum(-1, dtype=torch.int32)
    # Fixed-width BCSR rows, in ascending KV order. Padding entries are never read.
    indices = torch.argsort(
        active.to(torch.int32), dim=-1, descending=True, stable=True
    )
    partial = (types.flatten() == 1).nonzero().flatten().int()
    mask_offsets = torch.full_like(types, -1)
    mask_offsets.flatten()[partial.long()] = torch.arange(
        partial.numel(), device=types.device, dtype=torch.int32
    )
    mask_bits = torch.empty(
        (partial.numel(), 384, 2), device=types.device, dtype=torch.int32
    )
    if partial.numel():
        _partial_mask_bits[(partial.numel(), 12)](
            q_aux,
            kv_aux,
            partial,
            mask_bits,
            q_aux.shape[1],
            kv_aux.shape[1],
            types.shape[-1],
            num_warps=4,
        )
    return SparsePlan(
        counts=counts,
        indices=indices.int(),
        types=types,
        q_aux=q_aux,
        kv_aux=kv_aux,
        mask_offsets=mask_offsets,
        mask_bits=mask_bits,
        q_order=torch.argsort(counts, descending=True, stable=True).int(),
    )
