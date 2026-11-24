# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Training-loop check of the automatic strategy: choice, lock-in and overhead.

:class:`PlanPrefetcher` worker processes (``--threads``: threads) run
:meth:`AutoStrategy.plan` (cost model and plan building on CPU) for the next
steps while the GPUs run the current one, as in training; the main loop only
attaches the plan and runs ``--layers`` attention
forward + backward passes. Prints one ``RESULT`` row per step (strategy,
attention time, CPU planning time in the planner, time the step waited for its
plan, attach time) and a summary. With ``--explicit R,U,exchange,balance``
it runs that one strategy with a plan built once, for the overhead comparison.

    torchrun --nproc-per-node 8 --nnodes 4 ... benchmarks/bench_auto.py \\
        --layout file:benchmarks/layouts/packs/bagel_video_2048k.json#0 \\
        --seq-len 2097152 --heads 28 --kv-heads 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
import torch.distributed as dist
from provenance import result_json

from usp_attention import USPConfig, build_usp_plan, init_distributed
from usp_attention.auto import (
    AutoStrategy,
    PlanPrefetcher,
    auto_attention,
    regime_signature,
    strategy_config,
)
from usp_attention.layouts import build_layout
from usp_attention.strategy import Strategy


def _parse_strategy(text: str) -> Strategy:
    r, u, exchange, balance = text.split(",")
    return Strategy(
        ring_size=int(r), ulysses_size=int(u), kv_exchange=exchange, balance=balance
    )


def main() -> None:
    """Run the training-loop simulation and print per-step and summary rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--layout", required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--sliding-window", type=int, default=None)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--trial-steps", type=int, default=2)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--depth", type=int, default=2, help="steps planned ahead")
    p.add_argument("--workers", type=int, default=2, help="planner processes")
    p.add_argument(
        "--threads", action="store_true", help="plan in threads, not processes"
    )
    p.add_argument(
        "--explicit",
        default=None,
        help="R,U,exchange,balance: run only this strategy, planned per step like auto",
    )
    p.add_argument(
        "--static",
        action="store_true",
        help="with --explicit: one plan built up front, no per-step planning",
    )
    p.add_argument("--label", default="auto")
    args = p.parse_args()

    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    kv_heads = args.kv_heads or args.heads
    layout = build_layout(args.layout, args.seq_len)
    auto = AutoStrategy(
        world=world,
        heads=args.heads,
        kv_heads=kv_heads,
        head_dim=args.head_dim,
        config=USPConfig(),
        top_k=args.top_k,
        trial_steps=args.trial_steps,
        horizon=args.horizon,
    )
    auto.create_groups()
    gen = torch.Generator(device=device).manual_seed(0)

    def run(plan, strategy: Strategy) -> float:
        # Local shards only (values do not affect timing; full copies would not fit).
        q, k, v, dout = (
            torch.randn(
                (1, plan.local_len, h, args.head_dim),
                device=device,
                dtype=torch.bfloat16,
                generator=gen,
            )
            for h in (args.heads, kv_heads, kv_heads, args.heads)
        )
        q, k, v = (t.requires_grad_(True) for t in (q, k, v))
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        dist.barrier()
        start.record()
        for _ in range(args.layers):
            auto_attention(q, k, v, plan, strategy).backward(dout)
        end.record()
        end.synchronize()
        ms = torch.tensor([start.elapsed_time(end)], device=device)
        dist.all_reduce(ms, op=dist.ReduceOp.MAX)
        return float(ms.item())

    rows = []
    shortlist = None
    if args.explicit and not args.static:
        # Same per-step planning thread as auto, with the strategy fixed.
        strategy = _parse_strategy(args.explicit)
        regime = regime_signature(*layout, sliding_window=args.sliding_window)
        auto._shortlists[regime] = [strategy]
        auto.locks[regime] = (strategy, 0)
    if args.static:
        strategy = _parse_strategy(args.explicit)
        plan = build_usp_plan(
            document_lens=layout[0],
            split_lens=layout[1],
            attn_modes=layout[2],
            groups=auto.groups[(strategy.ring_size, strategy.ulysses_size)],
            config=strategy_config(strategy, auto.config, hopper=auto.hopper),
            device=device,
        )
        for step in range(args.steps):
            rows.append(
                {"step": step, "strategy": strategy.label, "ms": run(plan, strategy)}
            )
    else:
        it = PlanPrefetcher(
            auto,
            ((i, *layout) for i in range(args.steps)),
            cp_rank=rank,
            sliding_window=args.sliding_window,
            depth=args.depth,
            workers=args.workers,
            processes=not args.threads,
        )
        for step in range(args.steps):
            t0 = time.perf_counter()
            batch = next(it)
            plan_ms = batch.plan_ms
            wait_ms = (time.perf_counter() - t0) * 1e3
            t1 = time.perf_counter()
            plan, _ = auto.attach(batch, device)
            shortlist = shortlist or [s.label for s in batch.shortlist]
            attach_ms = (time.perf_counter() - t1) * 1e3
            ms = run(plan, batch.strategy)
            auto.record(batch, ms)
            rows.append(
                {
                    "step": step,
                    "strategy": batch.strategy.label,
                    "ms": ms,
                    "plan_ms": plan_ms,
                    "wait_ms": wait_ms,
                    "attach_ms": attach_ms,
                }
            )
    if rank == 0:
        for r in rows:
            print("STEP " + result_json(dict(r, label=args.label)), flush=True)
        tail = rows[len(rows) // 2 :]
        locked = sorted({r["strategy"] for r in tail})
        summary = {
            "bench": "auto",
            "label": args.label,
            "layout": args.layout,
            "seq_len": args.seq_len,
            "world": world,
            "layers": args.layers,
            "mode": "static"
            if args.static
            else "explicit"
            if args.explicit
            else "auto",
            "shortlist": shortlist,
            "locked": locked,
            "steady_ms_median": statistics.median(r["ms"] for r in tail),
            "per_layer_ms": statistics.median(r["ms"] for r in tail) / args.layers,
        }
        if not args.static:
            summary.update(
                plan_ms_median=statistics.median(r["plan_ms"] for r in rows[1:]),
                wait_ms_median_after_first=statistics.median(
                    r["wait_ms"] for r in rows[1:]
                ),
                wait_ms_max_after_first=max(r["wait_ms"] for r in rows[1:]),
                attach_ms_median=statistics.median(r["attach_ms"] for r in rows),
                trial={
                    s: statistics.median(r["ms"] for r in rows if r["strategy"] == s)
                    for s in {r["strategy"] for r in rows}
                },
            )
        print("RESULT " + result_json(summary), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
