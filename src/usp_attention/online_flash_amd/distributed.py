# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Forward-only AMD sparse USP; explicit entry point, no autograd integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist

from usp_attention.distributed import USPGroups, seq_all_to_all
from usp_attention.online_flash_amd.exchange import gather_qkv
from usp_attention.online_flash_amd.interface import online_flash_attn_func
from usp_attention.online_flash_amd.sparse import SparsePlan, build_sparse_plan
from usp_attention.ring.block_sparse import build_block_sparse_from_cross_qkv_aux
from usp_attention.ring.ring_planning import _ptrr_scheduling


def _qkv_all_to_all(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, groups: USPGroups
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack each peer's Q/K/V head slices into one Ulysses exchange."""
    u = groups.ulysses_size
    if u == 1:
        return q, k, v
    heads = [x.shape[2] // u for x in (q, k, v)]
    send = torch.cat([x.unflatten(2, (u, -1)).movedim(2, 0) for x in (q, k, v)], dim=3)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=groups.ulysses_group)
    gathered = recv.movedim(0, 1).flatten(1, 2)
    q_out, k_out, v_out = (x.contiguous() for x in gathered.split(heads, dim=2))
    return q_out, k_out, v_out


@dataclass(kw_only=True, slots=True)
class AMDForwardPlan:
    groups: USPGroups
    permutation: torch.Tensor
    steps: list[SparsePlan]
    query_head_factor: int = 1
    kv_exchange: Literal["ring", "gather"] = "ring"

    def shard(self, x: torch.Tensor) -> torch.Tensor:
        """Permute BSHD input and take this CP rank's sequence slice."""
        g = self.groups
        return (
            x[:, self.permutation]
            .chunk(g.ring_size * g.ulysses_size, 1)[g.cp_rank]
            .contiguous()
        )


def build_forward_plan(
    aux: torch.Tensor,
    groups: USPGroups,
    *,
    balance: bool = True,
    query_head_factor: int = 1,
    kv_exchange: Literal["ring", "gather"] = "ring",
) -> AMDForwardPlan:
    """Plan once per layout; caller pads to a multiple of 384 * CP size.

    Uses the existing PTRR snake scheduler on 384-token query blocks. Attention
    tiles use 64-token KV blocks; only partial tiles cache token-level bitmaps.
    The scheduler balances all CP shards before each Ulysses group merges them.

    ``kv_exchange="ring"`` rotates KV with overlapped RCCL P2P. ``"gather"``
    gathers full KV along the ring axis and computes one sparse attention call.
    With Ulysses, one variable-size CP all-to-all routes Q to its Ulysses peers
    and KV to every rank needing that KV head. Gather uses more KV/BCSR memory.

    With one KV head per rank after Ulysses, set ``query_head_factor`` to the
    number of query heads per rank to fold those heads into the query sequence.
    Original metadata is repeated in the same order; K/V and output sharding
    are unchanged. This selects the contiguous-head HIP specialization without
    transposing Q. Both BCSR storage and partial bitmaps can grow with the factor.
    """
    if query_head_factor < 1:
        raise ValueError(f"query_head_factor must be positive, got {query_head_factor}")
    if kv_exchange not in ("ring", "gather"):
        raise ValueError(f"unknown KV exchange {kv_exchange!r}")
    n = aux.shape[1]
    multiple = 384 * groups.ring_size * groups.ulysses_size
    if n % multiple:
        raise ValueError(f"sequence length must be divisible by {multiple}")
    perm = torch.arange(n, device=aux.device)
    if balance and groups.ring_size > 1:
        cross = [track for a in aux for track in (a, a)]
        types = build_block_sparse_from_cross_qkv_aux(
            cross, block_size=(384, 64), types_only=True
        ).block_type
        costs = (types != 0).sum(-1)
        blocks = _ptrr_scheduling(costs, groups.cp_size).reshape(-1)
        perm = (blocks[:, None] * 384 + torch.arange(384, device=aux.device)).reshape(
            -1
        )
    permuted_aux = aux[:, perm].contiguous()
    windows = permuted_aux.chunk(groups.ring_size, 1)
    kv_windows = (
        [permuted_aux]
        if kv_exchange != "ring"
        else [
            windows[(groups.ring_rank - s) % groups.ring_size]
            for s in range(groups.ring_size)
        ]
    )
    q_aux = windows[groups.ring_rank].repeat_interleave(query_head_factor, dim=1)
    steps = [build_sparse_plan(q_aux, kv_aux) for kv_aux in kv_windows]
    return AMDForwardPlan(
        groups=groups,
        permutation=perm,
        steps=steps,
        query_head_factor=query_head_factor,
        kv_exchange=kv_exchange,
    )


def sparse_usp_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: AMDForwardPlan,
    *,
    round_mode: str = "rtne",
) -> torch.Tensor:
    """BSHD local shards -> BSHD local output. Ring KV uses overlapped RCCL P2P.

    Includes both Ulysses exchanges and every online kernel call. No backward is
    registered; reject grad-enabled inputs rather than silently losing gradients.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError("AMD sparse USP is forward-only; use torch.no_grad()")
    g = plan.groups
    if q.shape[2] % g.ulysses_size or k.shape[2] % g.ulysses_size:
        raise ValueError("query and KV heads must divide the Ulysses degree")
    direct = plan.kv_exchange == "gather" and g.ulysses_size > 1
    q, k, v = gather_qkv(q, k, v, g) if direct else _qkv_all_to_all(q, k, v, g)
    q_shape = q.shape
    head_first = plan.query_head_factor != 1 or g.ulysses_size > 1
    if plan.query_head_factor != 1:
        if q.shape[2] != plan.query_head_factor or k.shape[2] != 1:
            raise ValueError(
                "packed queries require one KV head and the planned Q head count"
            )
        q = q.reshape(q.shape[0], 1, -1, q.shape[3])
    elif head_first:
        q = q.transpose(1, 2).contiguous()
    if head_first:
        k, v = (x.transpose(1, 2).contiguous() for x in (k, v))
    if plan.kv_exchange == "gather" and g.ring_size > 1 and not direct:
        kv = torch.stack((k, v))
        gathered = torch.empty(
            (g.ring_size * 2, *k.shape), device=k.device, dtype=k.dtype
        )
        dist.all_gather_into_tensor(gathered, kv, group=g.ring_group)
        gathered = gathered.unflatten(0, (g.ring_size, 2))
        seq_dim = 3 if head_first else 2
        gathered = gathered.movedim(0, seq_dim).flatten(seq_dim, seq_dim + 1)
        k, v = gathered.unbind(0)
    out = state = None
    for s, step in enumerate(plan.steps):
        work = []
        if s + 1 < len(plan.steps):
            next_k, next_v = torch.empty_like(k), torch.empty_like(v)
            send = dist.get_global_rank(g.ring_group, (g.ring_rank + 1) % g.ring_size)
            recv = dist.get_global_rank(g.ring_group, (g.ring_rank - 1) % g.ring_size)
            work = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, k, send, g.ring_group),
                    dist.P2POp(dist.irecv, next_k, recv, g.ring_group),
                    dist.P2POp(dist.isend, v, send, g.ring_group),
                    dist.P2POp(dist.irecv, next_v, recv, g.ring_group),
                ]
            )
        out, state = online_flash_attn_func(
            q,
            k,
            v,
            out=out,
            state=state,
            is_online_first=s == 0,
            is_online_last=s == len(plan.steps) - 1,
            sparse_plan=step,
            round_mode=round_mode,
            layout="bhsd" if head_first else "bshd",
        )
        for request in work:
            request.wait()
        if work:
            k, v = next_k, next_v
    if plan.query_head_factor != 1:
        out = out.reshape(q_shape)
    elif head_first:
        out = out.transpose(1, 2).contiguous()
    return seq_all_to_all(out, g.ulysses_group, scatter_dim=1, gather_dim=2)
