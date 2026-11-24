# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-FileCopyrightText: 2025 Tri Dao
# SPDX-FileCopyrightText: 2026 The FlashMask Authors
# SPDX-License-Identifier: MIT AND BSD-3-Clause
"""FlashMask baselines for the 32-128 GPU and B300 tables (attention only).

Every attention-only FlashMask row in ``results/mlsys/`` came from this
script, run in FlashMask's own venv (``flashmask_runs/setup_flashmask.sh``;
``flashmask_runs/README.md`` lists the runs). Masks come from
:mod:`flashmask_masks`, exact for every layout we use, vision groups
included. Three baselines:

- ``--mode balanced`` (FlashMask CP): FlashMask's public PyTorch distributed
  path (``tests/torch/distributed/balanced_flashmask.py``): its load balancer
  (``flashmask.cp_balance.balance_flashmask_input``, ``--ipo`` for the IPO
  solver) assigns chunks to ranks, each rank all-gathers K/V, runs one
  FlashMask-4 kernel pass and reduce-scatters dK/dV.
- ``--ulysses U`` (FM + Ulysses, our construction): Ulysses all-to-all over U
  consecutive ranks, FlashMask CP across the groups
  (``--replicate-kv`` repeats KV heads so U divides them).
- ``--mode overlap`` (B300 only): FlashMask's overlapped dual-chunk path
  (``overlap_flashmask.py``), which runs on a single Blackwell node.

    torchrun --nnodes 4 --nproc-per-node 8 ... flashmask_cp_bench.py \
        --layout benchmarks/layouts/packs/molmo2_video_1024k.json#0 \
        --seq-len 1048576 --heads 32 --kv-heads 8 --ulysses 8 --ipo
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
import time
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

FM = os.environ.get("FLASHMASK_DIR", "/path/to/flashmask")
sys.path.insert(0, FM)
sys.path.insert(0, os.path.join(FM, "tests/torch/distributed"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from _layouts import mean_rel_errors  # noqa: E402
from cp_utils import all_gather_contiguous, reduce_scatter_any_axis_order  # noqa: E402
from flashmask.cp_balance import balance_flashmask_input  # noqa: E402
from flashmask.cute.ops.bwd import _flash_attn_bwd  # noqa: E402
from flashmask.cute.ops.fwd import _flash_attn_fwd  # noqa: E402
from flashmask_masks import flashmask_ranges  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from provenance import result_json  # noqa: E402

if TYPE_CHECKING:
    from jaxtyping import Float, Int


class _BalancedFlashMask(torch.autograd.Function):
    """FlashMask's tests/torch/distributed/balanced_flashmask.py (BalancedFlashMaskFM4),
    inlined to import the kernels from flashmask.cute.ops (its own import path pulls
    in the PaddlePaddle frontend)."""

    @staticmethod
    def forward(ctx, query, key, value, local_mask, group):
        k = all_gather_contiguous(key, group=group, axis=1)
        v = all_gather_contiguous(value, group=group, axis=1)
        out, lse = _flash_attn_fwd(
            query,
            k,
            v,
            startend_row_indices=local_mask,
            causal=False,
            return_lse=True,
            pack_gqa=False,
        )[:2]
        ctx.save_for_backward(query, key, value, local_mask, out, lse)
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, local_mask, out, lse = ctx.saved_tensors
        group = ctx.group
        k = all_gather_contiguous(key, group=group, axis=1)
        v = all_gather_contiguous(value, group=group, axis=1)
        dq, dk, dv = _flash_attn_bwd(
            query,
            k,
            v,
            out,
            dout,
            lse,
            local_mask,
            causal=False,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )[:3]
        return (
            dq,
            reduce_scatter_any_axis_order(dk, axis=1, group=group),
            reduce_scatter_any_axis_order(dv, axis=1, group=group),
            None,
            None,
        )


def balanced_flashmask_attention_fm4(
    query: Float[torch.Tensor, " batch local_tokens num_heads head_dim"],
    key: Float[torch.Tensor, " batch local_tokens kv_heads head_dim"],
    value: Float[torch.Tensor, " batch local_tokens kv_heads head_dim"],
    local_mask: Int[torch.Tensor, " 1 1 local_tokens 2"],
    group: dist.ProcessGroup,
) -> Float[torch.Tensor, " batch local_tokens num_heads head_dim"]:
    """Run balanced FlashMask attention with autograd support.

    Args:
        query: Local query tensor shaped (batch, local_tokens, heads,
            head_dim).
        key: Local key tensor in the same token order as query.
        value: Local value tensor with the same shape as key.
        local_mask: Local packed FlashMask start/end mask.
        group: Context-parallel process group.

    Returns:
        The local attention output tensor with the same shape as query.
    """
    return _BalancedFlashMask.apply(query, key, value, local_mask, group)


def rearrange_blocks(
    input_tensor: Int[torch.Tensor, " batch 1 tokens 2"], cp_size: int
) -> Int[torch.Tensor, " batch 1 tokens 2"]:
    """Interleave FlashMask mask blocks into dual-chunk order.

    Args:
        input_tensor: Mask tensor shaped (batch, 1, tokens, 2).
        cp_size: Context-parallel group size; tokens must divide into 2 *
            cp_size equal blocks.

    Returns:
        A reordered mask tensor with the same shape as input_tensor.
    """
    b, _, n, _ = input_tensor.shape
    n_blocks = cp_size * 2
    order = []
    for i in range(cp_size):
        order += [i, n_blocks - 1 - i]
    idx = torch.tensor(order, dtype=torch.int64, device=input_tensor.device)
    blocks = input_tensor.reshape([b, -1, n_blocks, n // n_blocks, 2])
    return blocks.index_select(2, idx).reshape([b, -1, n, 2])


class _OverlappedFlashMask(torch.autograd.Function):
    """FlashMask's tests/torch/distributed/overlap_flashmask.py (OverlappedFlashMaskFM4,
    single-node dual-chunk, SM100 only), inlined with the flashmask.cute.ops imports:
    the kernel gathers remote K/V while computing and reduce-scatters dK/dV."""

    @staticmethod
    def forward(ctx, query, key, value, mask, group):
        from cp_utils import preprocess_index_dual_chunks

        rank, cp = dist.get_rank(group=group), dist.get_world_size(group=group)
        half = query.shape[1] // 2
        sri = preprocess_index_dual_chunks(
            mask,
            chunk_id_first=rank,
            chunk_id_second=2 * cp - rank - 1,
            seq_blocksize=half,
            max_seqlen_q=half,
        )
        sri = rearrange_blocks(sri, cp)
        fwd_mask = torch.roll(sri, shifts=-query.shape[1] * (rank + 1), dims=2)
        out, lse = _flash_attn_fwd(
            query,
            key,
            value,
            startend_row_indices=fwd_mask,
            causal=False,
            return_lse=True,
            pack_gqa=False,
            group=group,
        )[:2]
        ctx.save_for_backward(query, key, value, out, lse, sri)
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, out, lse, sri = ctx.saved_tensors
        group = ctx.group
        rank = dist.get_rank(group=group)
        bwd_mask = torch.roll(sri, shifts=-query.shape[1] * rank, dims=2)
        dq, dk, dv = _flash_attn_bwd(
            query,
            key,
            value,
            out,
            dout,
            lse,
            bwd_mask,
            causal=False,
            group=group,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )[:3]
        return dq, dk, dv, None, None


def _seq_to_heads(
    x: Float[torch.Tensor, " 1 local_tokens num_heads head_dim"], group, u
) -> Float[torch.Tensor, " 1 gathered_tokens local_heads head_dim"]:
    """Ulysses all-to-all: (1, l, H, D) -> (1, u*l, H/u, D) within the Ulysses group."""
    from torch.distributed.nn.functional import all_to_all_single

    b, l, h, d = x.shape
    x = x.reshape(l, u, h // u, d).transpose(0, 1).contiguous()
    out = all_to_all_single(torch.empty_like(x), x, group=group)
    return out.reshape(1, u * l, h // u, d)


def _heads_to_seq(
    x: Float[torch.Tensor, " 1 gathered_tokens local_heads head_dim"], group, u
) -> Float[torch.Tensor, " 1 local_tokens num_heads head_dim"]:
    """Inverse all-to-all: (1, u*l, H/u, D) -> (1, l, H, D)."""
    from torch.distributed.nn.functional import all_to_all_single

    b, L, hu, d = x.shape
    x = x.reshape(u, L // u, hu, d).contiguous()
    out = all_to_all_single(torch.empty_like(x), x, group=group)
    return out.transpose(0, 1).reshape(1, L // u, u * hu, d)


def load_layout(spec: str, seq_len: int) -> tuple[list[int], list[int], list[str]]:
    """Load one packed layout, resolving a missing JSON path to its gzip copy.

    Args:
        spec: JSON path with optional #index; index defaults to zero.
        seq_len: Expected total number of tokens.

    Returns:
        A tuple (document_lens, split_lens, attn_modes) describing the pack.
    """
    path, _, index = spec.partition("#")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"  # benchmarks/layouts/packs/ holds the gzipped packs
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        entry = json.load(f)["layouts"][int(index or 0)]
    docs, splits, modes = (
        entry["document_lens"],
        entry["split_lens"],
        entry["attn_modes"],
    )
    assert sum(docs) == seq_len == sum(splits), (sum(docs), seq_len)
    return docs, splits, modes


def packed_mask(
    docs: list[int],
    splits: list[int],
    modes: list[str],
    window: int | None,
    device: torch.device | str,
) -> tuple[Int[torch.Tensor, " 1 1 tokens 2"], int, bool]:
    """Build a packed FlashMask and count its visible token pairs.

    Exact for every mode (see flashmask_masks.py); documents with vision groups
    are reordered (text first, then image spans in reverse), which only
    permutes tokens once per model. Returns the mask and the exact visible-pair
    count.

    Args:
        docs: Packed document lengths in tokens.
        splits: Attention-span lengths in document order.
        modes: Attention mode for each span.
        window: Optional sliding-window size; None selects global attention.
        device: Device on which to place the mask.

    Returns:
        A tuple (mask, visible_pairs, reordered). The mask is int32 with shape
        (1, 1, tokens, 2) and stores [hi, lo] bounds. The boolean reordered
        indicates that vision-group tokens need the canonical permutation
        before attention.
    """
    perm, lo, hi = flashmask_ranges(docs, splits, modes, window)
    mask = torch.from_numpy(np.stack([hi, lo], axis=-1).astype(np.int32))[None, None]
    return mask.to(device), int((hi - lo).sum()), perm is not None


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    mask: torch.Tensor,
) -> list[torch.Tensor]:
    """fp32 out / dq / dk / dv for full ``(1, S, H, D)`` inputs under a FlashMask.

    The mask is FlashMask's column-wise ``(1, 1, S, 2)`` ``[hi, lo]`` bounds: query
    row ``i`` sees key ``j`` iff ``lo[j] <= i < hi[j]``. One head at a time keeps the
    fp32 score matrix to ``S**2`` floats; GQA repeats the key/value heads.
    """
    S = q.shape[1]
    hi, lo = mask[0, 0, :, 0], mask[0, 0, :, 1]
    rows = torch.arange(S, device=q.device)[:, None]
    visible = (rows >= lo[None, :]) & (rows < hi[None, :])
    rep = q.shape[2] // k.shape[2]
    outs, dqs = [], []
    dk = torch.zeros(k.shape, dtype=torch.float32, device=q.device)
    dv = torch.zeros_like(dk)
    for h in range(q.shape[2]):
        qh, kh, vh = (
            t[0, :, i].float().requires_grad_(True)
            for t, i in ((q, h), (k, h // rep), (v, h // rep))
        )
        out = torch.nn.functional.scaled_dot_product_attention(
            qh[None], kh[None], vh[None], attn_mask=visible
        )[0]
        out.backward(dout[0, :, h].float())
        outs.append(out.detach())
        dqs.append(qh.grad)
        dk[0, :, h // rep] += kh.grad
        dv[0, :, h // rep] += vh.grad
    return [torch.stack(outs, 1)[None], torch.stack(dqs, 1)[None], dk, dv]


def _check(args, mask, bucket, attn, kv_heads, world, rank, device) -> None:
    """Print CHECK rows comparing this CP run and one-GPU FlashMask to fp32.

    Every rank draws the same global tensors, runs FlashMask CP on its balanced
    chunks, and compares them with the same chunks of the fp32 reference. Rank 0
    also runs FlashMask on the whole sequence (one GPU) for the ratio's denominator.
    """
    gen = torch.Generator(device=device).manual_seed(0)
    shapes = [
        (1, args.seq_len, h, args.head_dim)
        for h in (args.heads, kv_heads, kv_heads, args.heads)
    ]
    full = [
        torch.randn(s, device=device, dtype=torch.bfloat16, generator=gen)
        for s in shapes
    ]
    reference = _reference(*full, mask)
    chunk = args.balance_chunk

    def local(t: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [t[:, i * chunk : (i + 1) * chunk] for _, i in bucket], 1
        ).contiguous()

    q, k, v = (local(t).requires_grad_(True) for t in full[:3])
    out = attn(q, k, v)
    out.backward(local(full[3]))
    errors = torch.tensor(
        mean_rel_errors([out, q.grad, k.grad, v.grad], [local(r) for r in reference]),
        device=device,
    )
    dist.all_reduce(errors)
    rows = [("cp", (errors / world).tolist())]
    if rank == 0:
        q1, k1, v1 = (t.clone().requires_grad_(True) for t in full[:3])
        out1, lse1 = _flash_attn_fwd(
            q1,
            k1,
            v1,
            startend_row_indices=mask,
            causal=False,
            return_lse=True,
            pack_gqa=False,
        )[:2]
        dq1, dk1, dv1 = _flash_attn_bwd(
            q1, k1, v1, out1, full[3], lse1, mask, causal=False
        )[:3]
        rows.append(("single_gpu", mean_rel_errors([out1, dq1, dk1, dv1], reference)))
        for kind, e in rows:
            print(
                "CHECK "
                + result_json(
                    {
                        "impl": "flashmask_cp_balanced_allgather"
                        if kind == "cp"
                        else "flashmask_single_gpu",
                        "label": args.label,
                        "layout": args.layout,
                        "seq_len": args.seq_len,
                        "world": world if kind == "cp" else 1,
                        "heads": args.heads,
                        "kv_heads": kv_heads,
                        "head_dim": args.head_dim,
                        "reference": "fp32",
                        "out": e[0],
                        "dq": e[1],
                        "dk": e[2],
                        "dv": e[3],
                    }
                ),
                flush=True,
            )


def main() -> None:
    """Benchmark FlashMask attention from sys.argv and print result rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--balance-chunk", type=int, default=2048)
    p.add_argument("--sliding-window", type=int, default=None)
    p.add_argument(
        "--mode",
        choices=["balanced", "overlap"],
        default="balanced",
        help="balanced: cp_balance + all-gather (H100/B300); overlap: in-kernel overlapped "
        "all-gather / reduce-scatter, single node, dual-chunk (SM100 only)",
    )
    p.add_argument(
        "--ipo",
        action="store_true",
        help="IPO solver (FlashMask falls back to LPT above 512 chunks)",
    )
    p.add_argument(
        "--ulysses",
        type=int,
        default=1,
        help="Hybrid baseline: Ulysses all-to-all over this many consecutive ranks (intra-node), "
        "FlashMask's balanced all-gather CP across the remaining world/ulysses ranks",
    )
    p.add_argument(
        "--replicate-kv",
        type=int,
        default=1,
        help="repeat KV heads so --ulysses divides them",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--label", default="flashmask")
    p.add_argument("--forward-only", action="store_true")
    p.add_argument(
        "--check",
        action="store_true",
        help="print CHECK rows: mean relative error of out/dq/dk/dv vs an fp32 "
        "reference, for this run and for FlashMask on one GPU (balanced, no Ulysses)",
    )
    p.add_argument(
        "--profile-dir",
        default=None,
        help="rank 0 writes a torch profiler trace of one extra step here",
    )
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    U = args.ulysses
    R = world // U
    assert R * U == world
    # Ulysses innermost (consecutive ranks share a node); ring groups stride by U.
    group = u_group = None
    for r_ in range(R):
        g = dist.new_group(ranks=list(range(r_ * U, (r_ + 1) * U)))
        if rank // U == r_:
            u_group = g
    for u_ in range(U):
        g = dist.new_group(ranks=list(range(u_, world, U)))
        if rank % U == u_:
            group = g
    ring_rank = rank // U

    docs, splits, modes = load_layout(args.layout, args.seq_len)
    kv_heads = args.kv_heads or args.heads
    t0 = time.perf_counter()
    mask, pairs, reordered = packed_mask(
        docs, splits, modes, args.sliding_window, device
    )
    if args.mode == "balanced":
        local_mask, buckets = balance_flashmask_input(
            mask, R, ring_rank, balance_chunk_size=args.balance_chunk, use_ipo=args.ipo
        )
        if U == 1:
            attn = lambda q_, k_, v_: balanced_flashmask_attention_fm4(
                q_, k_, v_, local_mask, group
            )  # noqa: E731
        else:

            def attn(q_, k_, v_):
                rep = args.replicate_kv
                if rep > 1:
                    k_, v_ = (t.repeat_interleave(rep, dim=2) for t in (k_, v_))
                q_, k_, v_ = (_seq_to_heads(t, u_group, U) for t in (q_, k_, v_))
                out = balanced_flashmask_attention_fm4(q_, k_, v_, local_mask, group)
                return _heads_to_seq(out, u_group, U)
    else:
        assert U == 1
        local_mask = mask
        attn = lambda q_, k_, v_: _OverlappedFlashMask.apply(q_, k_, v_, mask, group)  # noqa: E731
    torch.cuda.synchronize()
    plan_ms = (time.perf_counter() - t0) * 1e3
    local_len = args.seq_len // world
    # Column-wise mask: one column per (gathered) key, query rows rebased to this rank.
    assert local_mask.shape[2] == args.seq_len, (local_mask.shape, args.seq_len)

    if args.check:
        if args.mode != "balanced" or U != 1:
            raise SystemExit("--check supports --mode balanced without --ulysses")
        _check(args, mask, buckets[ring_rank], attn, kv_heads, world, rank, device)
        dist.destroy_process_group()
        return

    gen = torch.Generator(device=device).manual_seed(rank)
    q = torch.randn(
        1,
        local_len,
        args.heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    k = torch.randn(
        1,
        local_len,
        kv_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    for t in (q, k, v):
        t.requires_grad_(True)

    def step():
        out = attn(q, k, v)
        if args.forward_only:
            return
        out.backward(dout)
        q.grad = k.grad = v.grad = None

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iters):
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
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
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
            os.makedirs(args.profile_dir, exist_ok=True)
            prof.export_chrome_trace(
                os.path.join(args.profile_dir, f"{args.label}.json")
            )

    if rank == 0:
        median = statistics.median(times)
        flops = 3.5 * 4 * args.heads * args.head_dim * pairs
        row = {
            "bench": "attention",
            "label": args.label,
            "impl": "flashmask_cp_"
            + ("balanced_allgather" if args.mode == "balanced" else "overlap_dualchunk")
            + (f"_ulysses{U}" if U > 1 else ""),
            "layout": os.path.basename(args.layout.partition("#")[0])
            .removesuffix(".gz")
            .removesuffix(".json")
            + "#"
            + (args.layout.partition("#")[2] or "0"),
            "seq_len": args.seq_len,
            "world": world,
            "ring": R,
            "ulysses": U,
            "dp_replicas": 1,
            "replicate_kv": args.replicate_kv,
            "heads": args.heads,
            "kv_heads": kv_heads,
            "head_dim": args.head_dim,
            "balance_chunk": args.balance_chunk,
            "sliding_window": args.sliding_window,
            "reordered_for_groups": reordered,
            "mask_density": pairs / args.seq_len**2,
            "ipo": args.ipo,
            "forward_only": args.forward_only,
            "median_ms": median,
            "step_ms_all": times,
            "tflops_per_gpu": flops / (median * 1e-3) / world / 1e12,
            "peak_mem_gib": peak.item(),
            "plan_ms": plan_ms,
        }
        print("RESULT " + result_json(row), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
