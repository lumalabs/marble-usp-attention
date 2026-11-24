# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""FlashMask-3 on Hopper through PaddlePaddle: the public context parallelism.

FlashMask-3's C++ Hopper kernel ships in PaddlePaddle
(``paddle.nn.functional.flash_attention.flashmask_attention`` with
``FLAGS_flash_attn_version=3``, i.e. ``_C_ops.flashmask_attention_v2``). Its
public context parallelism on Hopper is PaddleFleet's
``cp_flashmask_allgatherkv_balance_forward`` / ``_backward``
(``paddlefleet/context_parallel_utils.py``): dual-chunk balanced token split,
all-gather K/V, one FlashMask-3 pass, reduce-scatter dK/dV (``--mode allgather``).
Paddle 3.4 also exposes FlashMask-3's mask-aware NVSHMEM overlap on Hopper
(``flashmask_attention(..., group=)``, local lengths 4k-128k, at most 64 ranks):
``--mode overlap_fwd`` uses its forward with the all-gather backward, and
``--mode overlap`` its backward too, whose gradients we could not validate.
This script times these paths, attention only, on our packed layouts with the
same arguments as ``flashmask_cp_bench.py``; ``--ulysses U`` adds a Ulysses
all-to-all over U consecutive ranks around them.

``CP_UTILS`` must point at PaddleFleet's ``context_parallel_utils.py`` (Apache
2.0); masks come from :mod:`flashmask_masks`::

    python -m paddle.distributed.launch --nnodes 4 --master IP:PORT --rank R \\
        --devices 0,1,2,3,4,5,6,7 benchmarks/baselines/flashmask3_paddle_bench.py \\
        --layout benchmarks/layouts/packs/molmo2_video_1024k.json#0 --seq-len 1048576 \\
        --heads 32 --kv-heads 8 --head-dim 128
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import statistics
import sys
import time

os.environ.setdefault("FLAGS_flash_attn_version", "3")

import numpy as np  # noqa: E402
import paddle  # noqa: E402
import paddle.distributed as dist  # noqa: E402
from paddle.autograd.py_layer import PyLayer  # noqa: E402
from paddle.nn.functional.flash_attention import flashmask_attention  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flashmask_masks import flashmask_ranges  # noqa: E402

_spec = importlib.util.spec_from_file_location("cp_utils", os.environ["CP_UTILS"])
cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpu)


class _FlashMask3CP(PyLayer):
    """PaddleFleet's FlashMaskContextParallel with the group passed explicitly."""

    @staticmethod
    def forward(ctx, query, key, value, mask, group):
        out, lse, local_mask, fa_version = cpu.cp_flashmask_allgatherkv_balance_forward(
            query, key, value, mask, group, False, True
        )
        ctx.save_for_backward(query, key, value, out, lse, local_mask)
        ctx.group, ctx.fa_version = group, fa_version
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, out, lse, local_mask = ctx.saved_tensor()
        return cpu.cp_flashmask_allgatherkv_balance_backward(
            query,
            key,
            value,
            local_mask,
            out,
            lse,
            dout,
            ctx.group,
            False,
            ctx.fa_version,
        )


def _rearrange_blocks(mask, cp):
    """FlashMask's overlap reorder: interleave the 2*cp mask blocks into dual-chunk order."""
    b, _, n, _ = mask.shape
    order = []
    for i in range(cp):
        order += [i, 2 * cp - 1 - i]
    idx = paddle.to_tensor(order, dtype="int64")
    blocks = mask.reshape([b, -1, 2 * cp, n // (2 * cp), 2])
    return blocks.index_select(index=idx, axis=2).reshape([b, -1, n, 2])


def _overlap_forward(query, key, value, mask, group):
    """FlashMask-3's overlapped CP forward on Hopper: ``flashmask_attention(..., group=)``.

    Local dual-chunk Q/K/V; the kernel fetches remote K/V over NVSHMEM while it
    computes, skipping blocks the mask does not need. Mask handling follows
    FlashMask's overlap test (``tests/paddle/distributed/overlap_flashmask.py``):
    dual-chunk preprocessing, block interleave, then a roll by ``-(rank+1)*L``.
    Returns the output, the LSE and the un-rolled dual-chunk mask.
    """
    rank, cp, length = group.rank, group.nranks, query.shape[1]
    sri = cpu.preprocess_index_dual_chunks(
        mask,
        chunk_id_first=rank,
        chunk_id_second=2 * cp - rank - 1,
        seq_blocksize=length // 2,
        max_seqlen_q=length // 2,
    )
    fwd_mask = paddle.roll(
        _rearrange_blocks(sri, cp), shifts=-length * (rank + 1), axis=2
    )
    out, lse = cpu.flashmask_attention(
        query,
        key,
        value,
        startend_row_indices=fwd_mask,
        causal=False,
        return_softmax_lse=True,
        training=True,
        group=group,
    )
    return out, lse, sri


class _FlashMask3Overlap(PyLayer):
    """Overlapped forward and overlapped backward (``flashmask_attention_v2_grad``).

    The backward rolls the mask by ``-rank*L`` as FlashMask's overlap test does.
    In Paddle 3.4 its dQ/dK/dV disagree with the all-gather path by 2-260% under
    every mask alignment we tried (Paddle's own autograd through
    ``flashmask_attention(group=)`` too), so its time is an unverified bound.
    """

    @staticmethod
    def forward(ctx, query, key, value, mask, group):
        out, lse, sri = _overlap_forward(query, key, value, mask, group)
        ctx.save_for_backward(
            query, key, value, out, lse, _rearrange_blocks(sri, group.nranks)
        )
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, out, lse, sri = ctx.saved_tensor()
        group = ctx.group
        bwd_mask = paddle.roll(sri, shifts=-query.shape[1] * group.rank, axis=2)
        dq, dk, dv = paddle._C_ops.flashmask_attention_v2_grad(
            query,
            key,
            value,
            out,
            lse,
            bwd_mask,
            None,
            dout,
            query.shape[-1] ** (-0.5),
            False,
            group.rank,
            group.nranks,
        )
        assert dk.shape[1] == key.shape[1], (
            "overlap backward returned gathered dK",
            dk.shape,
        )
        return dq, dk, dv


class _FlashMask3OverlapFwd(PyLayer):
    """FlashMask-3's overlapped forward with PaddleFleet's all-gather backward.

    The overlap forward matches an fp32 reference; the overlap backward does not
    (see :class:`_FlashMask3Overlap`). This is the fastest correct use of the
    public API.
    """

    @staticmethod
    def forward(ctx, query, key, value, mask, group):
        out, lse, sri = _overlap_forward(query, key, value, mask, group)
        ctx.save_for_backward(query, key, value, out, lse, sri)
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, dout):
        query, key, value, out, lse, sri = ctx.saved_tensor()
        return cpu.cp_flashmask_allgatherkv_balance_backward(
            query, key, value, sri, out, lse, dout, ctx.group, False, 3
        )


def _seq_to_heads(x, group, u):
    """(1, l, H, D) -> (1, u*l, H/u, D) within the Ulysses group."""
    _, l, h, d = x.shape
    parts = paddle.split(x.reshape([l, u, h // u, d]), u, axis=1)
    recv = [paddle.empty_like(p) for p in parts]
    dist.alltoall(recv, [p.contiguous() for p in parts], group=group)
    return paddle.concat(recv, axis=0).reshape([1, u * l, h // u, d])


class _SeqToHeads(PyLayer):
    @staticmethod
    def forward(ctx, x, group, u):
        ctx.group, ctx.u = group, u
        return _seq_to_heads(x, group, u)

    @staticmethod
    def backward(ctx, g):
        return _heads_to_seq(g, ctx.group, ctx.u)


def _heads_to_seq(x, group, u):
    """(1, u*l, H/u, D) -> (1, l, H, D)."""
    _, length, hu, d = x.shape
    parts = paddle.split(x.reshape([u, length // u, hu, d]), u, axis=0)
    recv = [paddle.empty_like(p) for p in parts]
    dist.alltoall(recv, [p.contiguous() for p in parts], group=group)
    out = paddle.stack([r.reshape([length // u, hu, d]) for r in recv], axis=1)
    return out.reshape([1, length // u, u * hu, d])


class _HeadsToSeq(PyLayer):
    @staticmethod
    def forward(ctx, x, group, u):
        ctx.group, ctx.u = group, u
        return _heads_to_seq(x, group, u)

    @staticmethod
    def backward(ctx, g):
        return _seq_to_heads(g, ctx.group, ctx.u)


def load_layout(spec, seq_len):
    path, _, index = spec.partition("#")
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--sliding-window", type=int, default=None)
    p.add_argument("--ulysses", type=int, default=1)
    p.add_argument(
        "--mode",
        choices=["allgather", "overlap", "overlap_fwd"],
        default="allgather",
        help="allgather: PaddleFleet's dual-chunk all-gather CP; overlap: FlashMask-3's "
        "NVSHMEM overlap (flashmask_attention(..., group=)), which skips unneeded K/V, "
        "forward and backward (backward unverified); overlap_fwd: overlap forward, "
        "all-gather backward",
    )
    p.add_argument("--replicate-kv", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--label", default="flashmask3")
    p.add_argument(
        "--check",
        action="store_true",
        help="compare the output on sampled rows with an fp32 reference (no timing)",
    )
    p.add_argument("--check-rows", type=int, default=256)
    # Accepted for list compatibility with flashmask_cp_bench.py; unused here.
    p.add_argument("--ipo", action="store_true")
    p.add_argument("--balance-chunk", type=int, default=None)
    args = p.parse_args()

    dist.init_parallel_env()
    world, rank = dist.get_world_size(), dist.get_rank()
    u = args.ulysses
    ring = world // u
    u_group = cp_group = None
    for r_ in range(ring):
        g = dist.new_group(list(range(r_ * u, (r_ + 1) * u)))
        if rank // u == r_:
            u_group = g
    for u_ in range(u):
        g = dist.new_group(list(range(u_, world, u)))
        if rank % u == u_:
            cp_group = g

    docs, splits, modes = load_layout(args.layout, args.seq_len)
    kv_heads = args.kv_heads or args.heads
    t0 = time.perf_counter()
    perm, lo, hi = flashmask_ranges(docs, splits, modes, args.sliding_window)
    mask = paddle.to_tensor(np.stack([hi, lo], axis=-1).astype(np.int32)[None, None])
    pairs = int((hi - lo).sum())
    plan_ms = (time.perf_counter() - t0) * 1e3

    local_len = args.seq_len // world
    paddle.seed(rank)
    q = paddle.randn([1, local_len, args.heads, args.head_dim]).astype("bfloat16")
    k = paddle.randn([1, local_len, kv_heads, args.head_dim]).astype("bfloat16")
    v = paddle.randn([1, local_len, kv_heads, args.head_dim]).astype("bfloat16")
    dout = paddle.randn([1, local_len, args.heads, args.head_dim]).astype("bfloat16")
    for t in (q, k, v):
        t.stop_gradient = False

    cp_op = {
        "overlap": _FlashMask3Overlap,
        "overlap_fwd": _FlashMask3OverlapFwd,
    }.get(args.mode, _FlashMask3CP)

    def cp_attn(q_, k_, v_):
        if ring == 1:
            # One rank per CP group (Ulysses over every rank): Paddle's CP helpers
            # do not run real attention on a single-rank group, so call the
            # kernel on the whole sequence directly.
            return flashmask_attention(
                q_, k_, v_, startend_row_indices=mask, causal=False
            )
        return cp_op.apply(q_, k_, v_, mask, cp_group)

    def attn(q_, k_, v_):
        if u == 1:
            return cp_attn(q_, k_, v_)
        if args.replicate_kv > 1:
            k_, v_ = (
                paddle.repeat_interleave(t, args.replicate_kv, axis=2) for t in (k_, v_)
            )
        q_, k_, v_ = (_SeqToHeads.apply(t, u_group, u) for t in (q_, k_, v_))
        out = cp_attn(q_, k_, v_)
        return _HeadsToSeq.apply(out, u_group, u)

    if args.check:
        gen = np.random.default_rng(0)
        full = [
            gen.standard_normal((1, args.seq_len, h, args.head_dim), dtype=np.float32)
            for h in (args.heads, kv_heads, kv_heads)
        ]
        qf, kf, vf = (paddle.to_tensor(x).astype("bfloat16") for x in full)
        if ring == 1:
            shards = [qf, kf, vf]
        else:
            shards = [
                cpu.scatter_balance(t, group=cp_group, axis=1) for t in (qf, kf, vf)
            ]
        if u == 1:
            for t in shards:
                t.stop_gradient = False
            out = cp_op.apply(*shards, mask, cp_group)
            if (
                args.mode != "allgather"
            ):  # gradients against the validated all-gather path
                g = paddle.randn(out.shape).astype("bfloat16")
                dist.broadcast(g, src=0)
                out.backward(g)
                ref_in = [t.detach().clone() for t in shards]
                for t in ref_in:
                    t.stop_gradient = False
                _FlashMask3CP.apply(*ref_in, mask, cp_group).backward(g)
                for name, a_, b_ in zip(("dq", "dk", "dv"), shards, ref_in):
                    x, y = a_.grad.astype("float32"), b_.grad.astype("float32")
                    rel = float((x - y).abs().mean() / y.abs().mean())
                    print(
                        f"CHECK rank {rank} {name} overlap vs all-gather mean rel diff: {rel:.3e}",
                        flush=True,
                    )
                out = out.detach()
        else:  # this rank's 1/u of its CP shard, through the hybrid, then regathered
            part = shards[0].shape[1] // u
            u_rank = rank % u
            out = attn(*(t[:, u_rank * part : (u_rank + 1) * part] for t in shards))
            parts = []
            dist.all_gather(parts, out, group=u_group)
            out = paddle.concat(parts, axis=1)
        if ring > 1:
            out = cpu.all_gather_balance(out, group=cp_group, axis=1)
        if rank == 0:
            rows = gen.choice(args.seq_len, args.check_rows, replace=False)
            q32, k32, v32 = (t.astype("float32").numpy()[0] for t in (qf, kf, vf))
            rep = args.heads // kv_heads
            k32, v32 = (np.repeat(t, rep, axis=1) for t in (k32, v32))
            keys = np.arange(args.seq_len)
            err, ref_mag = 0.0, 0.0
            got = out.astype("float32").numpy()[0]
            for r in rows:
                visible = ~(
                    (r >= hi) | (r < lo)
                )  # FlashMask: rows [hi, S) and [0, lo) masked
                s_ = np.einsum("hd,shd->hs", q32[r], k32[visible]) / np.sqrt(
                    args.head_dim
                )
                w = np.exp(s_ - s_.max(-1, keepdims=True))
                ref = np.einsum(
                    "hs,shd->hd", w / w.sum(-1, keepdims=True), v32[visible]
                )
                err += np.abs(got[r] - ref).mean()
                ref_mag += np.abs(ref).mean()
            del keys
            print(
                f"CHECK mean rel err vs fp32 on 256 rows: {err / ref_mag:.3e}",
                flush=True,
            )
        return

    def step():
        out = attn(q, k, v)
        out.backward(dout)
        q.clear_gradient(), k.clear_gradient(), v.clear_gradient()

    for _ in range(args.warmup):
        step()
    paddle.device.synchronize()
    times = []
    for _ in range(args.iters):
        dist.barrier()
        paddle.device.synchronize()
        s = time.perf_counter()
        step()
        paddle.device.synchronize()
        t = paddle.to_tensor([(time.perf_counter() - s) * 1e3], dtype="float32")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        times.append(float(t.item()))
    peak = paddle.to_tensor(
        [paddle.device.cuda.max_memory_allocated() / 2**30], dtype="float32"
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)

    if rank == 0:
        median = statistics.median(times)
        flops = 3.5 * 4 * args.heads * args.head_dim * pairs
        row = {
            "bench": "attention",
            "label": args.label,
            "impl": ("flashmask3_paddle_" if ring == 1 else "flashmask3_paddle_cp_")
            + {
                "allgather": "balanced_allgather",
                "overlap": "overlap_unverified_bwd",
                "overlap_fwd": "overlap_fwd_allgather_bwd",
            }[args.mode]
            + (f"_ulysses{u}" if u > 1 else ""),
            "mode": args.mode,
            "layout": os.path.basename(args.layout)
            .replace(".gz", "")
            .replace(".json", ""),
            "seq_len": args.seq_len,
            "world": world,
            "ring": ring,
            "ulysses": u,
            "replicate_kv": args.replicate_kv,
            "heads": args.heads,
            "kv_heads": kv_heads,
            "head_dim": args.head_dim,
            "sliding_window": args.sliding_window,
            "reordered_for_groups": perm is not None,
            "flash_attn_version": os.environ.get("FLAGS_flash_attn_version"),
            "paddle": paddle.__version__,
            "median_ms": median,
            "step_ms_all": times,
            "plan_ms": plan_ms,
            "peak_mem_gib": float(peak.item()),
            "mask_density": pairs / args.seq_len**2,
            "tflops_per_gpu": flops / (median * 1e-3) / world / 1e12,
        }
        print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
