# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Layer-level benchmark for usp_attention on packed sparse layouts.

Run with torchrun; every rank must see the same arguments::

    torchrun --nproc-per-node 8 benchmarks/bench_attention.py \
        --layout mixed --seq-len 262144 --ring 2 --ulysses 4 --combine online

One ``RESULT {json}`` line is printed per run with:

- step time (forward + backward unless ``--forward-only``) per iteration,
  taken as the maximum over ranks, summarized as median / p10 / p90 over all
  timed iterations of all trials, plus the per-trial medians;
- sparsity-aware throughput: FLOPs count only unmasked (query, key) pairs,
  ``4 * B * H * D * pairs`` forward and 2.5x that for backward;
- ring load balance: unmasked pairs per ring rank (max / mean) for the
  executed token order, and for contiguous sharding for comparison;
- plan build time and peak memory.

With ``--profile-dir`` rank 0 also writes a torch profiler trace of one extra
iteration.
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from provenance import result_json

from usp_attention import (
    USPConfig,
    USPGroups,
    build_usp_plan,
    init_distributed,
    usp_attention,
)
from usp_attention.layouts import LAYOUT_NAMES, build_layout
from usp_attention.reference import visible_key_counts

if TYPE_CHECKING:
    from jaxtyping import Int


def _layout_label(layout: str) -> str:
    """A layout name, or ``<file stem>#<index>`` for a layout file."""
    if not layout.startswith("file:"):
        return layout
    path, _, index = layout.removeprefix("file:").partition("#")
    name = Path(path).name.removesuffix(".gz").removesuffix(".json")
    return f"{name}#{index or 0}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--ring", type=int, required=True)
    p.add_argument("--ulysses", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument(
        "--layout",
        default="mixed",
        help=f"one of {LAYOUT_NAMES} or file:<path>[#index] (a packed-layout JSON)",
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument(
        "--combine",
        default=None,
        choices=["online", "flash", "python", "fp32"],
        help="ring combine mode (default: resolved per GPU and head_dim; see USPConfig.combine_mode)",
    )
    p.add_argument("--no-load-balance", action="store_true")
    p.add_argument(
        "--no-group-reorder",
        action="store_true",
        help="Keep vision-group documents in natural token order.",
    )
    p.add_argument(
        "--no-column-ranges",
        action="store_true",
        help="without a ring, keep the position-aware predicate instead of the "
        "column-range one for vision-group layouts",
    )
    p.add_argument(
        "--balance",
        choices=["ptrr", "chunks"],
        default="ptrr",
        help="Ring load balancing: PTRR 128-token blocks or contiguous chunks.",
    )
    p.add_argument(
        "--kv-exchange",
        choices=["ring", "gather"],
        default="ring",
        help="Ring groups rotate KV (ring) or gather only the visible KV blocks once (gather).",
    )
    p.add_argument(
        "--fa3-gather",
        action="store_true",
        help="FlashAttention-3 for single-call dense gathers (H100; dK / dV round per rank).",
    )
    p.add_argument(
        "--halo-overlap",
        action="store_true",
        help="Gather: hide the halo exchanges behind two online-kernel calls (H100).",
    )
    p.add_argument(
        "--gather-regather",
        action="store_true",
        help="Gather: keep only local KV for the backward and exchange the halo again.",
    )
    p.add_argument(
        "--exact-dense-ring",
        action="store_true",
        help="Run interval layouts on the exact generation ring too.",
    )
    p.add_argument("--gather-max-keys", type=int, default=1 << 21)
    p.add_argument(
        "--dense-gather-per-row",
        action="store_true",
        help="Dense gather: one call per query row (no partial merges) instead of "
        "shared prefixes with fp16 partials merged in fp32.",
    )
    p.add_argument(
        "--dense-ring-chunks",
        type=int,
        default=None,
        help="Dense rings: force this many chunks per rank.",
    )
    p.add_argument(
        "--no-dense-ring-refine",
        action="store_true",
        help="Dense rings: keep the plain snake / zigzag deals.",
    )
    p.add_argument("--dense-ring-inplace", action="store_true")
    p.add_argument("--dense-ring-dkv-overlap", action="store_true")
    p.add_argument(
        "--no-native-plain-masks",
        action="store_true",
        help="Use block sparsity even for one plain causal or full document.",
    )
    p.add_argument(
        "--dense-ring",
        choices=["auto", "on", "off"],
        default="auto",
        help="Dense varlen ring steps for packed multimodal masks (auto = Blackwell only).",
    )
    p.add_argument(
        "--interval-masks",
        choices=["auto", "on", "off"],
        default="auto",
        help="Dense two-call path for packed multimodal masks (auto = Blackwell only).",
    )
    p.add_argument(
        "--no-fa3",
        action="store_true",
        help="Keep FA4 for native-mask Ulysses on H100 even if flash-attn-3 is installed.",
    )
    p.add_argument(
        "--bf16-grads",
        action="store_true",
        help="Round each ring step's gradients to bf16 (pre-fp32-accumulation behavior).",
    )
    p.add_argument(
        "--symm-mem-backend",
        default="cuda",
        choices=["cuda", "nccl"],
        help="torch symmetric-memory backend for symm_mem* transports. 'nccl' "
        "(copy engine) needs torch's NCCL headers to match the NCCL runtime.",
    )
    p.add_argument(
        "--transport",
        default="nccl",
        choices=["nccl", "symm_mem", "symm_mem_rotator", "symm_mem_direct", "nccl4py"],
    )
    p.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="Sliding-window attention over the last N keys (bidirectional spans exempt).",
    )
    p.add_argument("--forward-only", action="store_true")
    p.add_argument(
        "--replicate-kv",
        type=int,
        default=1,
        help="Repeat each KV head this many times before attention, so Ulysses can "
        "shard more ways than there are KV heads (DeepSpeed-Ulysses style).",
    )
    p.add_argument(
        "--no-comm-stream",
        action="store_true",
        help="Serialize KV transport with attention on one stream (no overlap).",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="")
    p.add_argument("--profile-dir", default=None)
    return p.parse_args()


def ring_imbalance(
    row_counts: Int[torch.Tensor, " tokens"],
    order: Int[torch.Tensor, " tokens"] | None,
    ring: int,
) -> float:
    """Measure the maximum-to-mean visible-pair load across ring shards.

    Args:
        row_counts: One-dimensional tensor of visible-key counts per query
            token.
        order: Optional token permutation; None keeps the current order.
        ring: Number of equal-length ring shards.

    Returns:
        The largest per-shard visible-pair count divided by the mean.
    """
    rows = row_counts if order is None else row_counts[order.to(row_counts.device)]
    per_rank = rows.view(ring, -1).sum(dim=1).double()
    return (per_rank.max() / per_rank.mean()).item()


def main() -> None:
    # CuTe DSL installs its own sys.excepthook on import, which hides
    # uncaught exceptions from torchrun workers; restore the default.
    """Benchmark packed attention from sys.argv and emit stamped result rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__
    if os.environ.get("SYMM_RING_DEBUG", "0") != "0":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parse_args()
    if args.dense_gather_per_row:
        os.environ["USP_DENSE_GATHER_EXACT"] = "1"
    device = init_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    groups = USPGroups.create(
        ring_size=args.ring,
        ulysses_size=args.ulysses,
        symm_mem_copy_engine=args.transport.startswith("symm_mem")
        and args.symm_mem_backend == "nccl",
    )
    docs, splits, modes = build_layout(
        args.layout, args.seq_len, ring_size=args.ring, seed=args.seed
    )
    config = USPConfig(
        combine_mode=args.combine,  # type: ignore[arg-type]
        load_balance=not args.no_load_balance,
        balance=args.balance,
        kv_exchange=args.kv_exchange,
        fa3_gather=args.fa3_gather,
        reorder_vision_groups=not args.no_group_reorder,
        column_range_masks=not args.no_column_ranges,
        native_plain_masks=not args.no_native_plain_masks,
        fa3_on_hopper=not args.no_fa3,
        interval_masks={"auto": None, "on": True, "off": False}[args.interval_masks],
        dense_ring={"auto": None, "on": True, "off": False}[args.dense_ring],
        dense_ring_chunks=args.dense_ring_chunks,
        dense_ring_refine=not args.no_dense_ring_refine,
        dense_ring_inplace=args.dense_ring_inplace,
        dense_ring_dkv_overlap=args.dense_ring_dkv_overlap,
        exact_dense_ring=args.exact_dense_ring,
        gather_regather=args.gather_regather,
        halo_overlap=args.halo_overlap,
        gather_max_keys=args.gather_max_keys,
        fp32_grads=not args.bf16_grads,
        transport=args.transport,  # type: ignore[arg-type]
        use_cuda_comm_stream=not args.no_comm_stream,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    plan = build_usp_plan(
        document_lens=docs,
        split_lens=splits,
        attn_modes=modes,
        groups=groups,
        config=config,
        device=device,
        sliding_window=args.sliding_window,
    )
    torch.cuda.synchronize()
    plan_ms = (time.perf_counter() - t0) * 1e3

    kv_heads = args.kv_heads or args.heads
    local = plan.local_len
    gen = torch.Generator(device=device).manual_seed(args.seed + rank)
    q = torch.randn(
        args.batch,
        local,
        args.heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    k = torch.randn(
        args.batch,
        local,
        kv_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    q.requires_grad_(not args.forward_only)
    k.requires_grad_(not args.forward_only)
    v.requires_grad_(not args.forward_only)

    def step() -> None:
        if args.replicate_kv > 1:
            out = usp_attention(
                q,
                k.repeat_interleave(args.replicate_kv, dim=2),
                v.repeat_interleave(args.replicate_kv, dim=2),
                plan,
            )
        else:
            out = usp_attention(q, k, v, plan)
        if not args.forward_only:
            out.backward(dout)
            q.grad = k.grad = v.grad = None

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times_ms: list[float] = []
    trial_medians: list[float] = []
    for _ in range(args.trials):
        trial: list[float] = []
        for _ in range(args.iters):
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step()
            end.record()
            end.synchronize()
            t = torch.tensor([start.elapsed_time(end)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            trial.append(t.item())
        times_ms += trial
        trial_medians.append(statistics.median(trial))
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    if args.profile_dir and rank == 0:
        Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
    if args.profile_dir:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            dist.barrier()
            step()
            torch.cuda.synchronize()
        if rank == 0:
            name = (
                args.label
                or f"{args.layout}_{args.seq_len}_R{args.ring}U{args.ulysses}_{config.resolved_combine_mode(device, head_dim=args.head_dim)}"
            )
            prof.export_chrome_trace(str(Path(args.profile_dir) / f"{name}.json"))

    times_ms.sort()
    median = statistics.median(times_ms)

    if rank == 0:
        row_counts = visible_key_counts(
            docs, splits, modes, sliding_window=args.sliding_window
        ).to(device)
        pairs = int(row_counts.sum().item())
        fwd_flops = 4 * args.batch * args.heads * args.head_dim * pairs
        flops = fwd_flops * (1.0 if args.forward_only else 3.5)
        # Ranks outside one CP group are data-parallel replicas, each running
        # its own sequence in the same step.
        replicas = world // (args.ring * args.ulysses)
        flops *= replicas
        row = {
            "bench": "attention",
            "label": args.label,
            "layout": _layout_label(args.layout),
            "sliding_window": args.sliding_window,
            "seq_len": args.seq_len,
            "ring": args.ring,
            "ulysses": args.ulysses,
            "world": world,
            "dp_replicas": replicas,
            "combine": config.resolved_combine_mode(device, head_dim=args.head_dim),
            "load_balance": config.load_balance,
            "fp32_grads": config.fp32_grads,
            "native_mask": plan.native_mask,
            "fa3_on_hopper": not args.no_fa3,
            "interval_masks": args.interval_masks,
            "interval": plan.interval_plan is not None,
            "generation_ring": plan.generation_ring is not None,
            "dense_ring": plan.dense_ring is not None,
            "usp_config": asdict(config),
            "halo_len": (
                plan.halo.halo_len
                if plan.halo is not None
                else plan.dense_gather.halo_len
                if plan.dense_gather is not None
                else None
            ),
            "balance": args.balance,
            "kv_exchange": args.kv_exchange,
            "fa3_gather": args.fa3_gather,
            "gather_regather": args.gather_regather,
            "halo_overlap": args.halo_overlap,
            "exact_dense_ring": args.exact_dense_ring,
            "gather_max_keys": args.gather_max_keys,
            "dense_gather_per_row": args.dense_gather_per_row,
            "dense_ring_chunks": args.dense_ring_chunks,
            "dense_ring_refine": config.dense_ring_refine,
            "dense_ring_inplace": config.dense_ring_inplace,
            "dense_ring_dkv_overlap": config.dense_ring_dkv_overlap,
            "group_reorder": not args.no_group_reorder,
            "transport": args.transport,
            "batch": args.batch,
            "heads": args.heads,
            "kv_heads": kv_heads,
            "replicate_kv": args.replicate_kv,
            "head_dim": args.head_dim,
            "forward_only": args.forward_only,
            "comm_stream": not args.no_comm_stream,
            "median_ms": median,
            "p10_ms": times_ms[len(times_ms) // 10],
            "p90_ms": times_ms[(len(times_ms) * 9) // 10],
            "trial_median_ms": trial_medians,
            "tflops_per_gpu": flops / (median * 1e-3) / world / 1e12,
            "mask_density": pairs / args.seq_len**2,
            "ring_imbalance": ring_imbalance(
                row_counts, plan.token_permutation, args.ring
            )
            if args.ring > 1
            else 1.0,
            "ring_imbalance_contiguous": ring_imbalance(row_counts, None, args.ring)
            if args.ring > 1
            else 1.0,
            "plan_ms": plan_ms,
            "peak_mem_gib": peak_gib,
        }
        print("RESULT " + result_json(row), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
