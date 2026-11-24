# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention.ring.ring_planning import RankRingPlan, RingStepPlan

RingStepCallback = Callable[[int, Tensor, Tensor, RingStepPlan | None], None]
RingStepKernel = Callable[
    [Tensor, Tensor, Tensor, int, RingStepPlan | None], tuple[Tensor, Tensor]
]


@dataclass(kw_only=True, slots=True)
class PendingKVExchange:
    """In-flight asynchronous KV exchange state."""

    recv_key: Tensor
    recv_value: Tensor
    work_handles: list
    comm_stream: torch.cuda.Stream | None = None


@dataclass(kw_only=True, slots=True)
class PendingGradExchange:
    """In-flight asynchronous dk/dv gradient exchange (reverse direction)."""

    recv_dk: Tensor
    recv_dv: Tensor
    work_handles: list
    comm_stream: torch.cuda.Stream | None = None


def _launch_async_kv_rotation(
    *,
    curr_key: Tensor,
    curr_value: Tensor,
    recv_key: Tensor,
    recv_value: Tensor,
    process_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream | None,
) -> PendingKVExchange:
    """Start asynchronous ring KV exchange for one step.

    Raises:
        ValueError: If world size is <= 1, or send/recv buffer shapes or dtypes
            don't match.
    """
    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        raise ValueError(
            f"Expected world_size > 1 for ring exchange, got {world_size=}."
        )
    rank = dist.get_rank(process_group)
    next_local = (rank + 1) % world_size
    prev_local = (rank - 1 + world_size) % world_size
    next_rank = dist.get_global_rank(process_group, next_local)
    prev_rank = dist.get_global_rank(process_group, prev_local)

    if curr_key.shape != recv_key.shape or curr_value.shape != recv_value.shape:
        raise ValueError(
            f"Send/recv buffer shape mismatch: "
            f"{curr_key.shape=} vs {recv_key.shape=}, "
            f"{curr_value.shape=} vs {recv_value.shape=}."
        )
    if curr_key.dtype != recv_key.dtype or curr_value.dtype != recv_value.dtype:
        raise ValueError(
            f"Send/recv dtype mismatch: "
            f"{curr_key.dtype=} vs {recv_key.dtype=}, "
            f"{curr_value.dtype=} vs {recv_value.dtype=}."
        )

    def _enqueue_exchange() -> list:
        ops = [
            dist.P2POp(dist.irecv, recv_key, prev_rank, process_group),
            dist.P2POp(dist.irecv, recv_value, prev_rank, process_group),
            dist.P2POp(dist.isend, curr_key, next_rank, process_group),
            dist.P2POp(dist.isend, curr_value, next_rank, process_group),
        ]
        return dist.batch_isend_irecv(ops)

    with torch.autograd.profiler.record_function("ring_nccl_p2p_launch_kv"):
        if comm_stream is not None and curr_key.is_cuda:
            current_stream = torch.cuda.current_stream(device=curr_key.device)
            comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(comm_stream):
                work_handles = _enqueue_exchange()
        else:
            work_handles = _enqueue_exchange()

    return PendingKVExchange(
        recv_key=recv_key,
        recv_value=recv_value,
        work_handles=work_handles,
        comm_stream=comm_stream,
    )


def _wait_pending_kv_exchange(pending: PendingKVExchange) -> tuple[Tensor, Tensor]:
    """Wait for in-flight KV exchange and return received KV buffers."""
    with torch.autograd.profiler.record_function("ring_nccl_p2p_wait_kv"):
        for handle in pending.work_handles:
            handle.wait()
        if pending.comm_stream is not None and pending.recv_key.is_cuda:
            torch.cuda.current_stream(device=pending.recv_key.device).wait_stream(
                pending.comm_stream
            )
    return pending.recv_key, pending.recv_value


def run_async_ring_kv_pipeline(
    *,
    initial_key: Tensor,
    initial_value: Tensor,
    process_group: dist.ProcessGroup,
    step_callback: RingStepCallback,
    rank_plan: RankRingPlan | None = None,
    use_cuda_comm_stream: bool = True,
    comm_stream: torch.cuda.Stream | None = None,
) -> None:
    """Rotate KV asynchronously and run per-step compute callback.

    Communication is launched before each step's compute so the next KV shard can
    be in-flight while current-step attention runs.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialized.
        ValueError: If the process group world size is invalid, or ``rank_plan``
            step count doesn't match world size.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before ring pipeline."
        )

    world_size = dist.get_world_size(process_group)
    if world_size <= 0:
        raise ValueError(f"Invalid process group world size: {world_size=}.")
    if world_size == 1:
        step_plan = rank_plan.steps[0] if rank_plan is not None else None
        step_callback(0, initial_key, initial_value, step_plan)
        return

    if rank_plan is not None and len(rank_plan.steps) != world_size:
        raise ValueError(
            f"rank_plan must have one step per rank: {len(rank_plan.steps)=}, "
            f"{world_size=}."
        )

    if comm_stream is None:
        comm_stream = (
            torch.cuda.Stream(device=initial_key.device)
            if use_cuda_comm_stream and initial_key.is_cuda
            else None
        )

    current_key = initial_key
    current_value = initial_value
    recv_key_buffers = [torch.empty_like(initial_key), torch.empty_like(initial_key)]
    recv_value_buffers = [
        torch.empty_like(initial_value),
        torch.empty_like(initial_value),
    ]

    for step in range(world_size):
        pending: PendingKVExchange | None = None
        if step < world_size - 1:
            recv_idx = step % 2
            with torch.autograd.profiler.record_function(
                f"ring_pipeline_nccl_p2p_prefetch/step={step + 1}"
            ):
                pending = _launch_async_kv_rotation(
                    curr_key=current_key,
                    curr_value=current_value,
                    recv_key=recv_key_buffers[recv_idx],
                    recv_value=recv_value_buffers[recv_idx],
                    process_group=process_group,
                    comm_stream=comm_stream,
                )

        step_plan = rank_plan.steps[step] if rank_plan is not None else None
        step_callback(step, current_key, current_value, step_plan)

        if pending is not None:
            with torch.autograd.profiler.record_function(
                f"ring_pipeline_nccl_p2p_wait/step={step + 1}"
            ):
                current_key, current_value = _wait_pending_kv_exchange(pending)


def _launch_async_grad_rotation(
    *,
    dk: Tensor,
    dv: Tensor,
    recv_dk: Tensor,
    recv_dv: Tensor,
    process_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream | None,
) -> PendingGradExchange:
    """Start asynchronous ring dk/dv exchange in the same direction as KV.

    dk/dv accumulators travel with the KV shard they belong to: send to
    next_rank, recv from prev_rank (same as KV rotation). After W-1 hops
    the accumulator returns to the KV-owning rank with all contributions.
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    next_local = (rank + 1) % world_size
    prev_local = (rank - 1 + world_size) % world_size
    next_rank = dist.get_global_rank(process_group, next_local)
    prev_rank = dist.get_global_rank(process_group, prev_local)

    def _enqueue() -> list:
        ops = [
            dist.P2POp(dist.irecv, recv_dk, prev_rank, process_group),
            dist.P2POp(dist.irecv, recv_dv, prev_rank, process_group),
            dist.P2POp(dist.isend, dk, next_rank, process_group),
            dist.P2POp(dist.isend, dv, next_rank, process_group),
        ]
        return dist.batch_isend_irecv(ops)

    with torch.autograd.profiler.record_function("ring_nccl_p2p_launch_grad"):
        if comm_stream is not None and dk.is_cuda:
            current_stream = torch.cuda.current_stream(device=dk.device)
            comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(comm_stream):
                work_handles = _enqueue()
        else:
            work_handles = _enqueue()

    return PendingGradExchange(
        recv_dk=recv_dk,
        recv_dv=recv_dv,
        work_handles=work_handles,
        comm_stream=comm_stream,
    )


def _wait_pending_grad_exchange(pending: PendingGradExchange) -> tuple[Tensor, Tensor]:
    """Wait for in-flight dk/dv exchange and return received buffers."""
    with torch.autograd.profiler.record_function("ring_nccl_p2p_wait_grad"):
        for handle in pending.work_handles:
            handle.wait()
        if pending.comm_stream is not None and pending.recv_dk.is_cuda:
            torch.cuda.current_stream(device=pending.recv_dk.device).wait_stream(
                pending.comm_stream
            )
    return pending.recv_dk, pending.recv_dv
