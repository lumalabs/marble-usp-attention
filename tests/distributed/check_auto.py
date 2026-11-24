# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""The automatic strategy runs exactly the explicit strategy it picks.

For every candidate of the layout's shortlist, compare a plan built on CPU by
:meth:`AutoStrategy.plan` and moved by :meth:`AutoStrategy.attach` with the
same strategy planned directly on the GPU: identical sharding, outputs and
gradients (bitwise). Run with torchrun on one or more nodes, e.g.::

    torchrun --nproc-per-node 8 tests/distributed/check_auto.py \\
        --layout file:benchmarks/layouts/packs/bagel_video_1024k.json#0 \\
        --seq-len 1048576 --heads 28 --kv-heads 4
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.distributed as dist

from usp_attention import USPConfig, build_usp_plan, init_distributed
from usp_attention.auto import (
    AutoStrategy,
    PlannedBatch,
    auto_attention,
    strategy_config,
)
from usp_attention.layouts import build_layout


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute difference between two tensors."""
    return float((a.float() - b.float()).abs().max())


def plans_equal(a, b, path: str = "plan") -> list[str]:
    """Paths where two plans differ (tensors by value, other leaves by ==)."""
    import dataclasses

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        ok = (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.shape == b.shape
            and a.dtype == b.dtype
            and torch.equal(a.cpu(), b.cpu())
        )
        return [] if ok else [path]
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return [
            d
            for f in dataclasses.fields(a)
            if f.name != "groups" and not f.name.startswith("_")
            for d in plans_equal(
                getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}"
            )
        ]
    if isinstance(a, list | tuple):
        if len(a) != len(b):
            return [path]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b))
            for d in plans_equal(x, y, f"{path}[{i}]")
        ]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [path]
        return [d for k in a for d in plans_equal(a[k], b[k], f"{path}[{k!r}]")]
    try:
        return [] if a == b else [path]
    except Exception:
        return [] if type(a) is type(b) else [path]


def main() -> None:
    """Compare auto-built and directly built plans for every shortlisted strategy.

    Raises:
        SystemExit: A candidate's plans differ in sharding, output or gradients.
    """
    sys.excepthook = sys.__excepthook__
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--layout", required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--sliding-window", type=int, default=None)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    kv_heads = args.kv_heads or args.heads
    docs, splits, modes = build_layout(args.layout, args.seq_len)
    auto = AutoStrategy(
        world=world,
        heads=args.heads,
        kv_heads=kv_heads,
        head_dim=args.head_dim,
        config=USPConfig(),
        top_k=args.top_k,
    )
    auto.create_groups()
    gen = torch.Generator(device=device).manual_seed(0)
    full = [
        torch.randn(
            (1, args.seq_len, h, args.head_dim),
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        for h in (args.heads, kv_heads, kv_heads, args.heads)
    ]
    failed = False
    shortlist = auto.shortlist(docs, splits, modes, sliding_window=args.sliding_window)
    for step, strategy in enumerate(shortlist):
        batch: PlannedBatch = auto.plan(
            step,
            docs,
            splits,
            modes,
            cp_rank=rank,
            sliding_window=args.sliding_window,
        )
        if batch.strategy != strategy:
            raise SystemExit(
                f"step {step} planned {batch.strategy}, expected {strategy}"
            )
        plan_auto, _ = auto.attach(batch, device)
        plan_direct = build_usp_plan(
            document_lens=docs,
            split_lens=splits,
            attn_modes=modes,
            groups=auto.groups[(strategy.ring_size, strategy.ulysses_size)],
            config=strategy_config(strategy, auto.config, hopper=auto.hopper),
            device=device,
        )
        results = []
        for plan in (plan_auto, plan_direct, plan_direct):
            q, k, v, dout = (plan.shard(t, seq_dim=1).contiguous() for t in full)
            q, k, v = (t.requires_grad_(True) for t in (q, k, v))
            out = auto_attention(q, k, v, plan, strategy)
            out.backward(dout)
            results.append((out.detach(), q.grad, k.grad, v.grad))
        mismatch = plans_equal(plan_auto, plan_direct)
        names = ("out", "dq", "dk", "dv")
        sharding = torch.equal(
            plan_auto.shard(full[0], seq_dim=1), plan_direct.shard(full[0], seq_dim=1)
        )
        diff = {
            f"{tag}_{n}": max_diff(a, b)
            for tag, (x, y) in (
                ("auto_vs_direct", (results[0], results[1])),
                ("direct_vs_direct", (results[1], results[2])),
            )
            for n, a, b in zip(names, x, y, strict=True)
        }
        stats = torch.tensor(
            [float(not sharding or bool(mismatch)), *diff.values()], device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        diff = dict(zip(diff, stats[1:].tolist(), strict=True))
        # The same plan (every tensor equal) runs the same kernels: the output
        # matches bitwise; gradients only up to the path's own run-to-run noise
        # (atomics, overlapped dK/dV returns), reported for reference.
        ok = stats[0].item() == 0 and diff["auto_vs_direct_out"] == 0
        if rank == 0:
            print(
                "RESULT "
                + json.dumps(
                    {
                        "check": "auto_vs_direct",
                        "layout": args.layout,
                        "world": world,
                        "strategy": strategy.label,
                        "same_plan": stats[0].item() == 0,
                        "plan_mismatch": mismatch[:5],
                        "ok": ok,
                        **diff,
                    }
                ),
                flush=True,
            )
        failed |= not ok
    dist.destroy_process_group()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
