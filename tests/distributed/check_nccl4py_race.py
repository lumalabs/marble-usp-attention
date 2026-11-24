# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Race check for the one-sided ring transports (NCCL4Py and symmetric memory).

Run with torchrun on one node (NCCL4Py: NCCL >= 2.29.7, ``NCCL_CUMEM_ENABLE=1``)::

    NCCL_CUMEM_ENABLE=1 torchrun --nproc-per-node 8 \
        tests/distributed/check_nccl4py_race.py --calls 100
    torchrun --nproc-per-node 8 tests/distributed/check_nccl4py_race.py \
        --transport symm_mem_direct --calls 100

Every rank publishes tensors whose values encode (owner rank, call), then makes
each step a slow consumer (``torch.cuda._sleep``) before reading what arrived.
A push that lands in a receive slot the consumer is still reading shows up as a
step that sees another owner's or call's data. Three paths are checked:

- ``fwd_kv``: the transport's forward driver
  (``run_async_ring_kv_pipeline_nccl4py_direct`` or the selected symmetric-memory
  forward pipeline).
- ``bwd_kv``: the direct KV ring with tag ``bwd_direct`` driven in the
  backward's order: prefetch step ``s + 1`` before step ``s`` computes.
- ``grad``: the dk/dv grad ring rotating an accumulator W - 1 hops plus the
  final hop home, as the ring backward does.

Before the first call of each path, odd ranks queue ``--alloc-skew-cycles`` of
GPU spin, so their transport buffers are created while their GPU is behind
their peers'. This reproduces the symmetric-memory backward deadlock, where a
fast peer's first signal landed in a new signal pad before the slow rank's GPU
had zeroed it. A path that does not finish within ``--timeout`` seconds dumps
every thread's stack and exits non-zero.

Each path prints one JSON line prefixed with ``RESULT``; the script exits
non-zero when any step read the wrong data.
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention import init_distributed
from usp_attention.ring import ring_pipeline_nccl4py as nccl4py_ring
from usp_attention.ring import ring_pipeline_symm as symm_ring

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from provenance import result_json  # noqa: E402

# Values stay exact in fp32: owner < 100, call < 1000.
_OWNER = 1000.0
_V_OFFSET = 0.5


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--calls", type=int, default=100)
    p.add_argument(
        "--sleep-cycles",
        type=int,
        default=2_000_000,
        help="GPU spin per step before reading what arrived (the slow consumer).",
    )
    p.add_argument("--numel", type=int, default=1 << 20)
    p.add_argument(
        "--paths",
        nargs="+",
        default=["fwd_kv", "bwd_kv", "grad"],
        choices=["fwd_kv", "bwd_kv", "grad"],
    )
    p.add_argument(
        "--transport",
        default="nccl4py",
        choices=sorted(_FWD_PIPELINES),
        help="nccl4py, or a symm_mem* transport (its forward variant; the backward "
        "KV and grad rings are shared by all three).",
    )
    p.add_argument(
        "--alloc-skew-cycles",
        type=int,
        default=1_000_000_000,
        help="GPU spin queued on odd ranks before each path's first call, so their "
        "transport buffers are created while their GPU lags (0 disables).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Seconds per path before the check dumps stacks and fails as hung.",
    )
    return p.parse_args()


_FWD_PIPELINES = {
    "nccl4py": nccl4py_ring.run_async_ring_kv_pipeline_nccl4py_direct,
    "symm_mem": symm_ring.run_async_ring_kv_pipeline_symm,
    "symm_mem_rotator": symm_ring.run_async_ring_kv_pipeline_symm_rotator,
    "symm_mem_direct": symm_ring.run_async_ring_kv_pipeline_symm_direct,
}


def _ring_module(args: argparse.Namespace):
    return nccl4py_ring if args.transport == "nccl4py" else symm_ring


def _grad_ring(args: argparse.Namespace, shape: torch.Size, group, device):
    """Return the grad ring the ring backward would use for this call."""
    if args.transport == "nccl4py":
        return nccl4py_ring._get_or_create_grad_ring(
            shape, torch.float32, device, group, None
        )
    # The ring backward builds a new SymmMemGradRing every call.
    return symm_ring.SymmMemGradRing(
        grad_shape=shape, dtype=torch.float32, device=device, process_group=group
    )


def _skew_first_call(args: argparse.Namespace, call: int, rank: int) -> None:
    """Make odd ranks' GPUs lag before the first call creates any buffers."""
    if call == 0 and rank % 2 == 1 and args.alloc_skew_cycles > 0:
        torch.cuda._sleep(args.alloc_skew_cycles)


def _code(owner: int, call: int) -> float:
    return owner * _OWNER + call


def _record(seen: Tensor, call: int, step: int, k: Tensor, v: Tensor) -> None:
    """Store first, last and mean of K and V; a partial overwrite moves one."""
    seen[call, step, 0] = k[0]
    seen[call, step, 1] = k[-1]
    seen[call, step, 2] = k.double().mean()
    seen[call, step, 3] = v[0]
    seen[call, step, 4] = v[-1]
    seen[call, step, 5] = v.double().mean()


def _kv_expected(calls: int, rank: int, world: int, device: torch.device) -> Tensor:
    expected = torch.empty(calls, world, 6, dtype=torch.float64, device=device)
    for call in range(calls):
        for step in range(world):
            code = _code((rank - step) % world, call)
            expected[call, step, :3] = code
            expected[call, step, 3:] = code + _V_OFFSET
    return expected


def _check_fwd_kv(args: argparse.Namespace, group, device: torch.device) -> Tensor:
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    seen = torch.zeros(args.calls, world, 6, dtype=torch.float64, device=device)
    for call in range(args.calls):
        _skew_first_call(args, call, rank)
        k = torch.full((args.numel,), _code(rank, call), device=device)
        v = k + _V_OFFSET

        def step_callback(step, sk, sv, _plan, call=call):
            if step > 0:
                torch.cuda._sleep(args.sleep_cycles)
            _record(seen, call, step, sk, sv)

        _FWD_PIPELINES[args.transport](
            initial_key=k,
            initial_value=v,
            process_group=group,
            step_callback=step_callback,
        )
    torch.cuda.synchronize()
    return seen != _kv_expected(args.calls, rank, world, device)


def _check_bwd_kv(args: argparse.Namespace, group, device: torch.device) -> Tensor:
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    seen = torch.zeros(args.calls, world, 6, dtype=torch.float64, device=device)
    shape = torch.Size([args.numel])
    for call in range(args.calls):
        _skew_first_call(args, call, rank)
        k = torch.full(shape, _code(rank, call), device=device)
        v = k + _V_OFFSET
        direct = _ring_module(args)._get_or_create_direct(
            shape, k.dtype, device, group, None, tag="bwd_direct"
        )
        direct.initial_publish(k, v)
        current_k, current_v = k, v
        # Same order as the ring backward: prefetch step + 1, compute, wait.
        for step in range(world):
            if step < world - 1:
                ready = direct.prepare_prefetch(step + 1)
                direct.start_prefetch(step + 1, ready)
            if step > 0:
                torch.cuda._sleep(args.sleep_cycles)
            _record(seen, call, step, current_k, current_v)
            if step < world - 1:
                direct.wait_prefetch()
                direct.ack_prefetch(step + 1)
                current_k, current_v = direct.staging(step + 1)
        direct.finalize()
    torch.cuda.synchronize()
    return seen != _kv_expected(args.calls, rank, world, device)


def _check_grad(args: argparse.Namespace, group, device: torch.device) -> Tensor:
    """Rotate dk/dv accumulators; each rank adds a contribution per hop.

    Rank ``r`` contributes ``r + 1 + 100 * shard`` to the accumulator for the
    KV shard it holds at that step (shard ``(r - step) % W``). After W - 1 hops
    and the final hop home, rank ``o`` holds the sum over all ranks for shard
    ``o``; a skipped, repeated or misrouted hop changes that sum.
    """
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    shape = torch.Size([args.numel])
    got = torch.zeros(args.calls, 4, dtype=torch.float64, device=device)
    for call in range(args.calls):
        _skew_first_call(args, call, rank)
        grad_ring = _grad_ring(args, shape, group, device)
        dk = dv = None
        for step in range(world):
            if step > 0:
                ready = grad_ring.prepare_read_prev(step - 1)
                grad_ring.start_read_prev(step - 1, ready)
            torch.cuda._sleep(args.sleep_cycles)
            shard = (rank - step) % world
            contrib = float(rank + 1 + 100 * shard + 10_000 * call)
            if step > 0:
                dk, dv = grad_ring.wait_read()
                grad_ring.ack_read_prev(step - 1)
                dk = dk + contrib
                dv = dv + 2 * contrib
            else:
                dk = torch.full(shape, contrib, device=device)
                dv = torch.full(shape, 2 * contrib, device=device)
            if step == 0:
                grad_ring.initial_publish(dk, dv)
            elif step < world - 1:
                grad_ring.republish(dk, dv, step)
        final = world - 1
        grad_ring.republish(dk, dv, final)
        ready = grad_ring.prepare_read_prev(final)
        grad_ring.start_read_prev(final, ready)
        dk_home, dv_home = grad_ring.wait_read()
        grad_ring.ack_read_prev(final)
        grad_ring.finalize(final)
        got[call] = torch.stack(
            [dk_home.min(), dk_home.max(), dv_home.min(), dv_home.max()]
        ).double()
    torch.cuda.synchronize()
    expected = torch.empty_like(got)
    for call in range(args.calls):
        total = sum(r + 1 + 100 * rank + 10_000 * call for r in range(world))
        expected[call, :2] = total
        expected[call, 2:] = 2 * total
    return (got != expected)[:, :, None]


_CHECKS = {"fwd_kv": _check_fwd_kv, "bwd_kv": _check_bwd_kv, "grad": _check_grad}


def main() -> None:
    args = _parse_args()
    device = init_distributed()
    if args.transport == "nccl4py":
        nccl4py_ring.require_nccl4py_one_sided()
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        raise SystemExit("Run with torchrun and at least 2 ranks.")

    failed = False
    for path in args.paths:
        faulthandler.dump_traceback_later(args.timeout, exit=True)
        bad = _CHECKS[path](args, group, device).flatten(1).any(dim=1)
        faulthandler.cancel_dump_traceback_later()
        bad_calls = bad.nonzero().flatten().tolist()
        per_rank: list[list[int] | None] = [None] * world
        dist.all_gather_object(per_rank, bad_calls)
        if rank == 0:
            n_bad = sum(len(calls) for calls in per_rank)
            row = {
                "check": "nccl4py_race",
                "transport": args.transport,
                "path": path,
                "world": world,
                "calls": args.calls,
                "sleep_cycles": args.sleep_cycles,
                "numel": args.numel,
                "alloc_skew_cycles": args.alloc_skew_cycles,
                "bad_rank_calls": n_bad,
                "bad_ranks": [r for r, calls in enumerate(per_rank) if calls],
                "first_bad_calls": sorted({c for calls in per_rank for c in calls})[:5],
                "passed": n_bad == 0,
            }
            print("RESULT " + result_json(row), flush=True)
            failed = failed or n_bad > 0
        dist.barrier()

    if args.transport == "nccl4py":
        nccl4py_ring.destroy_all_nccl4py_ring_communicators()
    status = torch.tensor([int(failed)], device=device)
    dist.broadcast(status, src=0)
    dist.destroy_process_group()
    if status.item():
        sys.exit(1)


if __name__ == "__main__":
    main()
