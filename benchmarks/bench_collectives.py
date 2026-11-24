# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Collective bandwidth check for the process groups USP uses.

Measures bus bandwidth of all-reduce over all ranks, all-to-all within each
node (the Ulysses group) and across nodes, and a ring-style P2P exchange
between corresponding ranks of neighbouring nodes. Use it to confirm a
multi-node run is on RDMA before trusting its attention timings::

    torchrun --nnodes 2 --nproc-per-node 8 ... benchmarks/bench_collectives.py
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch
import torch.distributed as dist
from provenance import result_json

from usp_attention.distributed import init_distributed


def _time(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        dist.barrier()
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e-3)
    return statistics.median(times)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--label", default=None, help="Row label (sweep.py passes one).")
    p.add_argument("--mib", type=int, default=512, help="Per-rank message size in MiB.")
    p.add_argument("--local-size", type=int, default=8, help="Ranks per node.")
    return p.parse_args()


def main() -> None:
    """Measure distributed communication from sys.argv and emit timing rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__
    args = _parse_args()
    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    local = args.local_size
    nodes = world // local
    numel = args.mib * 2**20 // 2
    x = torch.randn(numel, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()
    results = {}

    t = _time(lambda: dist.all_reduce(x))
    results["all_reduce_world_busbw_gbps"] = nbytes * 2 * (world - 1) / world / t / 1e9

    intra = inter = None
    for n in range(nodes):
        g = dist.new_group(list(range(n * local, (n + 1) * local)))
        if rank // local == n:
            intra = g
    for i in range(local):
        g = dist.new_group(list(range(i, world, local))) if nodes > 1 else None
        if rank % local == i:
            inter = g

    t = _time(lambda: dist.all_to_all_single(y, x, group=intra))
    results["all_to_all_intra_node_busbw_gbps"] = nbytes * (local - 1) / local / t / 1e9

    if inter is not None:
        t = _time(lambda: dist.all_to_all_single(y, x, group=inter))
        results["all_to_all_inter_node_busbw_gbps"] = (
            nbytes * (nodes - 1) / nodes / t / 1e9
        )
        nxt, prv = (rank + local) % world, (rank - local) % world

        def ring_step() -> None:
            ops = [dist.P2POp(dist.isend, x, nxt), dist.P2POp(dist.irecv, y, prv)]
            for req in dist.batch_isend_irecv(ops):
                req.wait()

        t = _time(ring_step)
        results["ring_p2p_inter_node_gbps_per_rank"] = nbytes / t / 1e9

    if rank == 0:
        print(
            "RESULT "
            + result_json(
                {"bench": "collectives", "world": world, "mib": args.mib, **results}
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
