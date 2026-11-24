# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""MagiAttention baseline on the same packed layouts as ``bench_attention.py``.

MagiAttention (https://github.com/SandAI-org/MagiAttention) is context-parallel
attention for irregular masks: the mask is a list of (query range, key range,
type) slices, which it load-balances and computes with its own Hopper kernel.
This script expresses each synthetic layout exactly as such slices, runs
``calc_attn`` forward + backward on the dispatched local shards, and prints a
``RESULT {json}`` line with the same timing and FLOP accounting as
``bench_attention.py``. Runs in its own environment::

    torchrun --nproc-per-node 8 benchmarks/baselines/magi_attn.py --layout mixed --seq-len 131072

Across nodes it needs nothing beyond torchrun's own ``--nnodes`` /
``--node-rank`` / ``--master-addr`` rendezvous (``sweep.py`` supplies them):
the process group spans every rank and MagiAttention's context-parallel group
is the whole world. ``--layout file:<pack>.json#0`` also finds the gzipped
pack. ``--chunk-size`` and ``--overlap-degree`` take several values; every
combination runs in the same process, one ``RESULT`` line each, with the
MagiAttention settings recorded under ``magi_config``. The defaults (chunk 512,
static overlap of degree 1) are MagiAttention's own defaults at commit
de13b2a. ``sweep_magi.py`` drives a tuning sweep with these flags.
"""

from __future__ import annotations

import argparse
import itertools
import os
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layouts import (  # noqa: E402
    build_layout,
    dense_mask,
    dense_reference,
    mean_rel_errors,
    visible_key_counts,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import result_json  # noqa: E402

if TYPE_CHECKING:
    from magi_attention.config import DistAttnConfig


def _window_slices(q0, q1, k0, k1, left, right):
    """MagiAttention's own decomposition of one sliding-window block (bottom-right aligned)."""
    from magi_attention.api.functools import infer_attn_mask_from_sliding_window
    from magi_attention.common import AttnRange

    qs, ks, ts = infer_attn_mask_from_sliding_window(
        AttnRange.from_range([q0, q1]), AttnRange.from_range([k0, k1]), (left, right)
    )
    names = {
        "full": "full",
        "causal": "causal",
        "inv_causal": "inv_causal",
        "bi_causal": "bi_causal",
    }
    return (
        [[r.start, r.end] for r in qs],
        [[r.start, r.end] for r in ks],
        [names[t.value] for t in ts],
    )


def _span_window_slices(a, b, s, window):
    """Full span [a, b) of a document starting at s: its whole span, plus keys
    k < a with q - k < window (a top-left-aligned inverse-causal band)."""
    qs, ks, ts = [[a, b]], [[a, b]], ["full"]
    rows_end = min(b, a + window - 1)  # later rows see nothing before the span
    if a > s and rows_end > a:
        lo = a - window + 1
        if lo >= s:
            qs.append([a, rows_end])
            ks.append([lo, a])
            ts.append("inv_causal")
        else:
            # Rows a .. a+c see every key from s on; later rows a band from s + 1.
            c = s - lo
            full_end = min(rows_end, a + c + 1)
            qs.append([a, full_end])
            ks.append([s, a])
            ts.append("full")
            if full_end < rows_end and s + 1 < a:
                qs.append([full_end, rows_end])
                ks.append([s + 1, a])
                ts.append("inv_causal")
    return qs, ks, ts


def window_mask_slices(
    docs: list[int], splits: list[int], modes: list[str], window: int
) -> tuple[list, list, list[str]]:
    """The packed mask with a sliding window as MagiAttention slices.

    Within a document starting at ``s``, a causal run ``[a, b)`` sees keys
    ``[max(s, q - W + 1), q]``: a bottom-right-aligned window ``(W - 1, 0)``
    over ``[a, b) x [s, b)``. A full span ``[a, b)`` sees itself in full plus
    the keys before it with ``q - k < W`` (see :func:`_span_window_slices`).

    Args:
        docs: Lengths of packed documents.
        splits: Lengths of attention spans in document order.
        modes: Attention mode of each span.
        window: Number of visible causal keys, including the query token
            itself.

    Returns:
        A tuple (q_ranges, k_ranges, types) of aligned query intervals, key
        intervals and MagiAttention mask-type names.
    """
    q_ranges, k_ranges, types = [], [], []
    split_iter = iter(zip(splits, modes))
    offset = 0
    for doc in docs:
        cur = offset
        while cur < offset + doc:
            n, mode = next(split_iter)
            if mode not in ("causal", "full"):
                raise NotImplementedError(f"sliding window with {mode!r} splits")
            if mode == "causal":
                qs, ks, ts = _window_slices(
                    cur, cur + n, offset, cur + n, window - 1, 0
                )
            else:
                qs, ks, ts = _span_window_slices(cur, cur + n, offset, window)
            q_ranges += qs
            k_ranges += ks
            types += ts
            cur += n
        offset += doc
    return q_ranges, k_ranges, types


def mask_slices(
    docs: list[int], splits: list[int], modes: list[str]
) -> tuple[list, list, list[str]]:
    """Build MagiAttention slices for the packed mixed-mode mask.

    A split attends to the earlier non-noise tokens of its document in full,
    and to itself causally (causal splits) or in full (full, noise and group
    splits). Noise splits are invisible to every other split; a group split
    also attends in full to every later group split of its document.

    Args:
        docs: Lengths of packed documents.
        splits: Lengths of attention spans in document order.
        modes: Attention mode of each span.

    Returns:
        A tuple (q_ranges, k_ranges, types) of aligned query intervals, key
        intervals and MagiAttention mask-type names.
    """
    q_ranges, k_ranges, types = [], [], []
    split_iter = iter(zip(splits, modes))
    offset = 0
    for doc in docs:
        doc_splits = []
        cur = offset
        while cur < offset + doc:
            n, mode = next(split_iter)
            doc_splits.append((cur, cur + n, mode))
            cur += n
        visible: list[list[int]] = []
        for j, (a, b, mode) in enumerate(doc_splits):
            later = (
                [[s, e] for s, e, m in doc_splits[j + 1 :] if m == "group"]
                if mode == "group"
                else []
            )
            for s, e in [*visible, *later]:
                q_ranges.append([a, b])
                k_ranges.append([s, e])
                types.append("full")
            q_ranges.append([a, b])
            k_ranges.append([a, b])
            types.append("causal" if mode == "causal" else "full")
            if mode != "noise":
                if visible and visible[-1][1] == a:
                    visible[-1][1] = b
                else:
                    visible.append([a, b])
        offset += doc
    return q_ranges, k_ranges, types


def dist_attn_config(
    degree: str, overlap_alg: str, min_chunk_size: int
) -> DistAttnConfig:
    """MagiAttention's ``DistAttnConfig`` for one overlap setting.

    ``degree`` is ``"0"`` (no overlap), ``"1"`` (local + one remote stage, the
    default), ``"N"`` (static N-stage overlap) or ``"dynamic"`` (the solver
    picks the degree per step).

    Args:
        degree: Overlap degree as "0", "1", "N" or "dynamic".
        overlap_alg: Overlap algorithm name: "uniform" or "greedy".
        min_chunk_size: Minimum chunk size supplied to MagiAttention's overlap
            solver.

    Returns:
        A MagiAttention DistAttnConfig for the requested overlap policy.
    """
    from magi_attention.common.enum import AttnOverlapMode
    from magi_attention.config import (
        DistAttnConfig,
        GreedyOverlapAlg,
        OverlapConfig,
        UniformOverlapAlg,
    )

    alg = GreedyOverlapAlg() if overlap_alg == "greedy" else UniformOverlapAlg()
    if degree == "dynamic":
        overlap = OverlapConfig(
            mode=AttnOverlapMode.DYNAMIC,
            degree=None,
            min_chunk_size=min_chunk_size,
            alg=alg,
        )
    else:
        overlap = OverlapConfig(
            mode=AttnOverlapMode.STATIC,
            degree=int(degree),
            min_chunk_size=min_chunk_size,
            alg=alg,
        )
    return DistAttnConfig(overlap_config=overlap)


def _parse_args() -> argparse.Namespace:
    """Parse the driver's command line.

    Module-level and dependent on argparse alone, so ``benchmarks/preflight.py``
    can check sweep commands against it on a machine without MagiAttention.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--layout",
        default="mixed",
        help="a synthetic layout name (_layouts.LAYOUT_NAMES) or file:<path>[#index]",
    )
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--sliding-window", type=int, default=None)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--chunk-size", type=int, nargs="+", default=[512])
    p.add_argument(
        "--overlap-degree",
        nargs="+",
        default=["1"],
        help="0 (no overlap), 1 (default), N >= 2 (static), or 'dynamic'.",
    )
    p.add_argument("--overlap-alg", choices=["uniform", "greedy"], default="uniform")
    p.add_argument("--min-chunk-size", type=int, default=512)
    p.add_argument(
        "--check",
        action="store_true",
        help="Check output and gradients against an fp32 reference instead of timing.",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--label", default="")
    return p.parse_args()


def main() -> None:
    """Run the MagiAttention sweep from sys.argv and print measurement rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    args = _parse_args()

    from magi_attention.api import (
        calc_attn,
        compute_pad_size,
        dispatch,
        magi_attn_flex_key,
    )
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.common.ranges import AttnRanges

    sys.excepthook = sys.__excepthook__
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.new_group(list(range(world)))

    seq = args.seq_len
    kv_heads = args.kv_heads or args.heads
    docs, splits, modes = build_layout(args.layout, seq, ring_size=world)
    q_r, k_r, types = (
        window_mask_slices(docs, splits, modes, args.sliding_window)
        if args.sliding_window
        else mask_slices(docs, splits, modes)
    )
    mask_type = {
        "full": AttnMaskType.FULL,
        "causal": AttnMaskType.CAUSAL,
        "inv_causal": AttnMaskType.INVCAUSAL,
        "bi_causal": AttnMaskType.BICAUSAL,
    }
    env = {
        k: v for k, v in sorted(os.environ.items()) if k.startswith("MAGI_ATTENTION_")
    }

    for chunk_size, degree in itertools.product(args.chunk_size, args.overlap_degree):
        magi_config = {
            "chunk_size": chunk_size,
            "overlap_degree": degree,
            "overlap_alg": args.overlap_alg,
            "min_chunk_size": args.min_chunk_size,
            "env": env,
        }
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pad = compute_pad_size(seq, world, chunk_size)
        key = magi_attn_flex_key(
            q_ranges=AttnRanges.from_ranges(q_r),
            k_ranges=AttnRanges.from_ranges(k_r),
            attn_mask_type=[mask_type[t] for t in types],
            total_seqlen_q=seq,
            total_seqlen_k=seq,
            num_heads_q=args.heads,
            num_heads_kv=kv_heads,
            head_dim=args.head_dim,
            pad_size=pad,
            cp_group_or_mesh=group,
            dist_attn_config=dist_attn_config(
                degree, args.overlap_alg, args.min_chunk_size
            ),
            chunk_size=chunk_size,
        )
        torch.cuda.synchronize()
        plan_ms = (time.perf_counter() - t0) * 1e3

        # The same seed for every configuration, so each one sees identical inputs.
        gen = torch.Generator(device=device).manual_seed(0)
        if args.check:
            full_q = torch.randn(
                seq,
                args.heads,
                args.head_dim,
                device=device,
                dtype=torch.bfloat16,
                generator=gen,
            )
            full_k = torch.randn(
                seq,
                kv_heads,
                args.head_dim,
                device=device,
                dtype=torch.bfloat16,
                generator=gen,
            )
            # Every rank must hold the same global tensors: draw all of them from `gen`.
            full_v = torch.randn(
                full_k.shape, device=device, dtype=full_k.dtype, generator=gen
            )
            q, k, v = (dispatch(t, key).contiguous() for t in (full_q, full_k, full_v))
        else:
            # Timing only: draw and dispatch a few heads at a time, so no rank holds
            # the global tensors (BAGEL 8M: 56 GiB of queries, which dispatch would
            # copy again while padding). Every rank draws the same values from
            # `gen`, so each shard is a slice of one global tensor.
            def sharded(heads: int, key=key) -> torch.Tensor:
                step = max(1, (8 << 30) // (seq * args.head_dim * 2))
                parts = []
                for h0 in range(0, heads, step):
                    part = torch.randn(
                        seq,
                        min(step, heads - h0),
                        args.head_dim,
                        device=device,
                        dtype=torch.bfloat16,
                        generator=gen,
                    )
                    parts.append(dispatch(part, key).contiguous())
                    del part
                return torch.cat(parts, dim=1)

            q, k, v = sharded(args.heads), sharded(kv_heads), sharded(kv_heads)
            full_q = full_k = full_v = None
        for t in (q, k, v):
            t.requires_grad_(True)
        if args.check:
            full_dout = torch.randn(
                full_q.shape, device=device, dtype=full_q.dtype, generator=gen
            )
            out = calc_attn(q, k, v, key)[0]
            out.backward(dispatch(full_dout, key))
            # dense_reference takes (B, S, H, D); Magi works on (S, H, D).
            reference = dense_reference(
                *(t[None] for t in (full_q, full_k, full_v, full_dout)),
                dense_mask(
                    docs,
                    splits,
                    modes,
                    device=device,
                    sliding_window=args.sliding_window,
                ),
            )
            local_ref = [dispatch(r[0].contiguous(), key) for r in reference]
            errors = torch.tensor(
                mean_rel_errors([out, q.grad, k.grad, v.grad], local_ref), device=device
            )
            dist.all_reduce(errors)
            errors = (errors / world).tolist()
            if rank == 0:
                print(
                    "CHECK "
                    + result_json(
                        {
                            "impl": "magi_attention",
                            "label": args.label,
                            "sliding_window": args.sliding_window,
                            "layout": args.layout,
                            "seq_len": seq,
                            "heads": args.heads,
                            "kv_heads": kv_heads,
                            "out": errors[0],
                            "dq": errors[1],
                            "dk": errors[2],
                            "dv": errors[3],
                            "magi_config": magi_config,
                        }
                    ),
                    flush=True,
                )
            del out, reference, local_ref, full_dout
            del full_q, full_k, full_v, q, k, v, key
            torch.cuda.empty_cache()
            continue
        del full_q, full_k, full_v
        dout = torch.randn_like(q)

        def step(q, k, v, dout, key) -> None:
            out = calc_attn(q, k, v, key)[0]
            out.backward(dout)
            q.grad = k.grad = v.grad = None

        for _ in range(args.warmup):
            step(q, k, v, dout, key)
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
            step(q, k, v, dout, key)
            e.record()
            e.synchronize()
            t = torch.tensor([s.elapsed_time(e)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            times.append(t.item())
        peak = torch.cuda.max_memory_allocated() / 2**30

        if rank == 0:
            times.sort()
            median = statistics.median(times)
            pairs = int(visible_key_counts(docs, splits, modes).sum())
            flops = 3.5 * 4 * args.heads * args.head_dim * pairs
            print(
                "RESULT "
                + result_json(
                    {
                        "bench": "attention",
                        "impl": "magi_attention",
                        "label": args.label,
                        "sliding_window": args.sliding_window,
                        "layout": args.layout,
                        "seq_len": seq,
                        "ring": world,
                        "ulysses": 1,
                        "world": world,
                        "combine": "-",
                        "load_balance": True,
                        "transport": "magi",
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
                        "plan_ms": plan_ms,
                        "peak_mem_gib": peak,
                        "num_slices": len(types),
                        "magi_config": magi_config,
                    }
                ),
                flush=True,
            )
        del q, k, v, dout, key
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
