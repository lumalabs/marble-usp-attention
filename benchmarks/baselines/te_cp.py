# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""TransformerEngine context-parallel attention baseline (what Megatron-LM uses).

Runs ``transformer_engine.pytorch.DotProductAttention`` with context
parallelism forward + backward on local shards and prints a ``RESULT {json}``
line with the same timing and FLOP accounting as ``bench_attention.py``.

``--comm`` selects TE's ``cp_comm_type``: ``p2p`` (ring), ``a2a`` (Ulysses),
``all_gather`` or ``a2a+p2p`` (Ulysses inside ``--ulysses`` ranks, ring across).
Layouts: ``causal`` / ``full`` (BSHD) and ``docs`` (the shared packed-document
layout, THD with ``cu_seqlens``; document lengths are rounded to multiples of
``2 * cp`` as TE requires). Self-contained so it runs in a separate
environment::

    torchrun --nproc-per-node 8 benchmarks/baselines/te_cp.py \
        --comm a2a+p2p --ulysses 4 --layout causal --seq-len 131072
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layouts import (  # noqa: E402
    dense_mask,
    dense_reference,
    docs_layout,
    mean_rel_errors,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import result_json  # noqa: E402

if TYPE_CHECKING:
    from jaxtyping import Int


def _docs(seq_len: int, multiple: int) -> list[int]:
    """The shared ``docs`` layout, each length rounded to a multiple of `multiple`.

    TE splits every document into ``2 * cp`` chunks, so lengths must divide
    evenly; rounding moves fewer than `multiple` tokens per document.
    """
    docs, _, _ = docs_layout(seq_len)
    rounded = [max(multiple, round(n / multiple) * multiple) for n in docs[:-1]]
    return [*rounded, seq_len - sum(rounded)]


def kwargs_for(
    docs: list[int], device: torch.device
) -> dict[str, Int[torch.Tensor, " num_boundaries"] | int]:
    """Build global THD sequence metadata for TransformerEngine.

    Args:
        docs: Lengths of the packed documents before CP partitioning.
        device: Device for the cumulative sequence-length tensors.

    Returns:
        Keyword arguments containing global cumulative query/KV lengths and
        maximum sequence lengths; TE partitions documents internally.
    """
    cu = torch.tensor(
        [0, *torch.tensor(docs).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    return dict(
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max(docs),
        max_seqlen_kv=max(docs),
    )


def _check(args, attn, docs, thd_kwargs, qkv_format, device) -> None:
    """Print TE's output / gradient error vs fp32 on this rank's tokens.

    Every rank builds the same global Q/K/V and takes the tokens TE expects on
    its CP rank: chunks ``r`` and ``2 * cp - 1 - r`` of ``2 * cp`` (BSHD), or
    TE's own per-document partition (THD).
    """
    import transformer_engine_torch as tex

    rank, world = dist.get_rank(), dist.get_world_size()
    seq, kv_heads = args.seq_len, args.kv_heads or args.heads
    gen = torch.Generator(device=device).manual_seed(0)
    q, k, v, dout = (
        torch.randn(
            1,
            seq,
            h,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        for h in (args.heads, kv_heads, kv_heads, args.heads)
    )
    if qkv_format == "thd":
        idx = tex.thd_get_partitioned_indices(
            thd_kwargs["cu_seqlens_q"], seq, world, rank
        ).long()
        layout = (docs, docs, ["causal"] * len(docs))
    else:
        chunks = torch.arange(seq, device=device).chunk(2 * world)
        idx = torch.cat([chunks[rank], chunks[2 * world - 1 - rank]])
        layout = ([seq], [seq], [args.layout])
    lead = (lambda t: t[0]) if qkv_format == "thd" else (lambda t: t)
    shards = [
        lead(t.index_select(1, idx)).contiguous().requires_grad_(True)
        for t in (q, k, v)
    ]
    out = attn(*shards, **(thd_kwargs if qkv_format == "thd" else {}))
    out.backward(lead(dout.index_select(1, idx)).reshape(out.shape))
    reference = dense_reference(q, k, v, dout, dense_mask(*layout, device=device))
    ours = [out.reshape(shards[0].shape)] + [t.grad for t in shards]
    errors = torch.tensor(
        mean_rel_errors(ours, [lead(r.index_select(1, idx)) for r in reference]),
        device=device,
    )
    dist.all_reduce(errors)
    errors = (errors / world).tolist()
    if rank == 0:
        print(
            "CHECK "
            + result_json(
                {
                    "impl": f"te_{args.comm}",
                    "layout": args.layout,
                    "seq_len": seq,
                    "ulysses": args.ulysses,
                    "kv_heads": kv_heads,
                    "out": errors[0],
                    "dq": errors[1],
                    "dk": errors[2],
                    "dv": errors[3],
                }
            ),
            flush=True,
        )


def main() -> None:
    """Benchmark TransformerEngine context parallelism from sys.argv.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--comm", choices=["p2p", "a2a", "all_gather", "a2a+p2p"], required=True
    )
    p.add_argument(
        "--ulysses", type=int, default=None, help="a2a group size for a2a+p2p"
    )
    p.add_argument("--layout", choices=["causal", "full", "docs"], required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument(
        "--check",
        action="store_true",
        help="Check output and gradients against an fp32 reference instead of timing.",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--label", default="")
    args = p.parse_args()

    import transformer_engine
    from transformer_engine.pytorch import DotProductAttention

    sys.excepthook = sys.__excepthook__
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    ranks = list(range(world))

    if args.comm == "a2a+p2p":
        uly = args.ulysses
        ring = world // uly
        a2a_group = p2p_group = None
        for r in range(ring):
            g = dist.new_group([r * uly + u for u in range(uly)])
            if rank // uly == r:
                a2a_group = g
        for u in range(uly):
            g = dist.new_group([r * uly + u for r in range(ring)])
            if rank % uly == u:
                p2p_group = g
        cp_group = [a2a_group, p2p_group]
    else:
        uly = world if args.comm == "a2a" else 1
        ring = 1 if args.comm == "a2a" else world
        cp_group = dist.new_group(ranks)

    kv_heads = args.kv_heads or args.heads
    seq = args.seq_len
    if args.layout == "docs":
        qkv_format, mask_type = "thd", "padding_causal"
        docs = _docs(seq, 2 * world)
    else:
        qkv_format = "bshd"
        mask_type = "causal" if args.layout == "causal" else "no_mask"
        docs = [seq]

    attn = DotProductAttention(
        num_attention_heads=args.heads,
        kv_channels=args.head_dim,
        num_gqa_groups=kv_heads,
        qkv_format=qkv_format,
        attn_mask_type=mask_type,
    ).to(device)
    attn.set_context_parallel_group(
        cp_group, ranks, torch.cuda.Stream(), cp_comm_type=args.comm
    )

    if args.check:
        _check(args, attn, docs, kwargs_for(docs, device), qkv_format, device)
        dist.destroy_process_group()
        return

    local = seq // world
    gen = torch.Generator(device=device).manual_seed(rank)
    lead = () if qkv_format == "thd" else (1,)
    q = torch.randn(
        *lead,
        local,
        args.heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    k = torch.randn(
        *lead,
        local,
        kv_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    v = torch.randn_like(k)
    for t in (q, k, v):
        t.requires_grad_(True)
    kwargs = kwargs_for(docs, device) if qkv_format == "thd" else {}
    dout = None

    def step() -> None:
        nonlocal dout
        out = attn(q, k, v, **kwargs)
        if dout is None:
            dout = torch.randn_like(out)
        out.backward(dout)
        q.grad = k.grad = v.grad = None

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.trials * args.iters):
        dist.barrier()
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        step()
        e.record()
        e.synchronize()
        t = torch.tensor([s.elapsed_time(e)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        times.append(t.item())
    peak = torch.cuda.max_memory_allocated() / 2**30

    if rank == 0:
        times.sort()
        median = statistics.median(times)
        if args.layout == "full":
            pairs = seq * seq
        else:
            pairs = sum(n * (n + 1) // 2 for n in docs)
        flops = 3.5 * 4 * args.heads * args.head_dim * pairs
        print(
            "RESULT "
            + result_json(
                {
                    "bench": "attention",
                    "impl": f"te{transformer_engine.__version__}_{args.comm}",
                    "label": args.label,
                    "layout": args.layout,
                    "seq_len": seq,
                    "ring": ring,
                    "ulysses": uly,
                    "world": world,
                    "combine": "-",
                    "load_balance": args.layout != "full",
                    "transport": args.comm,
                    "heads": args.heads,
                    "kv_heads": kv_heads,
                    "head_dim": args.head_dim,
                    "median_ms": median,
                    "p10_ms": times[len(times) // 10],
                    "p90_ms": times[(len(times) * 9) // 10],
                    "tflops_per_gpu": flops / (median * 1e-3) / world / 1e12,
                    "mask_density": pairs / seq**2,
                    "ring_imbalance": float("nan"),
                    "ring_imbalance_contiguous": float("nan"),
                    "plan_ms": 0.0,
                    "peak_mem_gib": peak,
                    "docs": docs if args.layout == "docs" else None,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
