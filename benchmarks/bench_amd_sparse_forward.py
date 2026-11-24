# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Forward-only sparse AMD appendix comparisons; no NVIDIA or dense-varlen path.

Run with PYTHONPATH=src .venv/bin/python -m torch.distributed.run --standalone
--nproc-per-node=8 benchmarks/bench_amd_sparse_forward.py --seq-len 262144.
Every configuration passes a distributed small-shape fp32-reference check first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from usp_attention.distributed import USPGroups, init_distributed
from usp_attention.layouts import build_layout
from usp_attention.online_flash_amd.distributed import (
    build_forward_plan,
    sparse_usp_forward,
)
from usp_attention.ring.packed_layout import build_packed_layout


def inputs(n, heads, kv_heads, layout, device):
    spec = build_layout(layout, n)
    packed = build_packed_layout(*spec, device=device)
    aux = torch.stack(
        [
            packed.full_and_noise_seq_id,
            packed.noise_seq_id,
            packed.document_id,
            torch.arange(n, device=device),
        ]
    ).int()
    torch.manual_seed(123)
    q = torch.randn(1, n, heads, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, n, kv_heads, 128, device=device, dtype=torch.bfloat16)
    return q, k, torch.randn_like(k), aux


def correctness(groups, args, device):
    n = 6144
    q, k, v, aux = inputs(
        n,
        args.heads,
        args.kv_heads,
        "mixed" if args.layout.startswith("file:") else args.layout,
        device,
    )
    factor = (
        args.heads // groups.ulysses_size
        if args.pack_mqa and args.kv_heads == groups.ulysses_size
        else 1
    )
    plan = build_forward_plan(
        aux,
        groups,
        balance=args.balance,
        query_head_factor=factor,
        kv_exchange=args.kv_exchange if groups.ulysses_size > 1 else "ring",
    )
    local = [plan.shard(x) for x in (q, k, v)]
    return check_reference(local, k, v, aux, plan, args, device)


def check_reference(local, k, v, aux, plan, args, device, sample_rows=None):
    groups = plan.groups
    actual = sparse_usp_forward(*local, plan, round_mode=args.round_mode)
    ids = plan.permutation.chunk(groups.ring_size * groups.ulysses_size)[groups.cp_rank]
    if sample_rows is not None:
        selected = (
            torch.linspace(0, ids.numel() - 1, sample_rows, device=device)
            .long()
            .unique()
        )
        ids = ids[selected]
        actual = actual[:, selected]
        local = [local[0][:, selected]]
    qa = aux[:, ids]
    mask = (qa[3, :, None] >= aux[3, None, :]) | (
        (qa[0, :, None] >= 0) & (qa[0, :, None] == aux[0, None, :])
    )
    mask &= (aux[1, None, :] < 0) | (qa[1, :, None] == aux[1, None, :])
    mask &= qa[2, :, None] == aux[2, None, :]
    # Check head by head to bound reference memory independently of head count.
    error, norm = 0.0, 0.0
    for h in range(args.heads):
        kh = h // (args.heads // args.kv_heads)
        scores = local[0][0, :, h].float() @ k[0, :, kh].float().T / 128**0.5
        expected = (
            scores.masked_fill(~mask, -torch.inf).softmax(-1).nan_to_num()
            @ v[0, :, kh].float()
        )
        torch.testing.assert_close(
            actual[0, :, h].float(), expected, atol=0.004, rtol=0.01
        )
        error += (actual[0, :, h].float() - expected).square().sum().item()
        norm += expected.square().sum().item()
    ratio = torch.tensor((error / norm) ** 0.5, device=device)
    dist.all_reduce(ratio, op=dist.ReduceOp.MAX)
    return ratio.item()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=262144)
    p.add_argument("--heads", type=int, default=28)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--layout", default="mixed")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--output", type=Path)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--pack-mqa", action="store_true")
    p.add_argument("--round-mode", choices=("rtne", "rtna", "rtz"), default="rtne")
    p.add_argument("--kv-exchange", choices=("ring", "gather"), default="ring")
    p.add_argument(
        "--configs",
        nargs="+",
        choices=("ulysses_dp", "ring_contiguous", "ring_ptrr", "usp", "usp_u2"),
        help="Run selected configurations (default: all four).",
    )
    args = p.parse_args()
    device = init_distributed()
    world = dist.get_world_size()
    if world not in (4, 8, 16) or args.heads % 4 or args.kv_heads % 4:
        raise ValueError("comparison requires 4/8/16 GPUs and heads divisible by four")
    if min(args.seq_len, args.trials, args.iterations) < 1 or args.warmup < 0:
        raise ValueError(
            "length, trials and iterations must be positive; warmup nonnegative"
        )
    configs = [
        ("ulysses_dp", 1, 4, False),
        ("ring_contiguous", world, 1, False),
        ("ring_ptrr", world, 1, True),
        ("usp", world // 4, 4, True),
    ]
    if args.configs and "usp_u2" in args.configs:
        configs.append(("usp_u2", world // 2, 2, True))
    for name, r, u, balance in configs:
        if args.configs and name not in args.configs:
            continue
        args.balance = balance
        groups = USPGroups.create(ring_size=r, ulysses_size=u)
        error = correctness(groups, args, device)
        if args.check_only:
            if dist.get_rank() == 0:
                print(
                    json.dumps({"config": name, "relative_l2_error": error}), flush=True
                )
            continue
        # Same physical length for every configuration. Padding is a separate
        # isolated document and removed from the useful-token throughput count.
        n = ((args.seq_len + 384 * world - 1) // (384 * world)) * (384 * world)
        q, k, v, aux = inputs(
            args.seq_len, args.heads, args.kv_heads, args.layout, device
        )
        if n > args.seq_len:
            tail = aux.new_full((4, n - args.seq_len), -1)
            tail[2] = int(aux[2].max()) + 1
            tail[3] = torch.arange(args.seq_len, n, device=device)
            aux = torch.cat([aux, tail], 1)
            q, k, v = [
                torch.cat([x, x.new_zeros((1, n - args.seq_len, x.shape[2], 128))], 1)
                for x in (q, k, v)
            ]
        torch.cuda.synchronize()
        start = time.perf_counter()
        factor = args.heads // u if args.pack_mqa and args.kv_heads == u else 1
        plan = build_forward_plan(
            aux,
            groups,
            balance=balance,
            query_head_factor=factor,
            kv_exchange=args.kv_exchange if u > 1 else "ring",
        )
        local = [plan.shard(x) for x in (q, k, v)]
        torch.cuda.synchronize()
        plan_s = time.perf_counter() - start
        bitmap_bytes = torch.tensor(
            sum(s.mask_bits.numel() * s.mask_bits.element_size() for s in plan.steps),
            device=device,
            dtype=torch.int64,
        )
        dist.all_reduce(bitmap_bytes, op=dist.ReduceOp.MAX)
        storages = {
            (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            for t in [plan.permutation, *(t for s in plan.steps for t in s.tensors)]
        }
        plan_bytes = torch.tensor(
            sum(size for _, size in storages), device=device, dtype=torch.int64
        )
        dist.all_reduce(plan_bytes, op=dist.ReduceOp.MAX)
        sampled_error = check_reference(
            local, k, v, aux, plan, args, device, sample_rows=17
        )
        del q, k, v
        for _ in range(args.warmup):
            sparse_usp_forward(*local, plan, round_mode=args.round_mode)
        torch.cuda.synchronize()
        trials = []
        for _ in range(args.trials):
            times = []
            for _ in range(args.iterations):
                dist.barrier()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                sparse_usp_forward(*local, plan, round_mode=args.round_mode)
                end.record()
                end.synchronize()
                elapsed = torch.tensor(begin.elapsed_time(end), device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                times.append(elapsed.item())
            trials.append(times)
        if dist.get_rank() == 0:
            ext = (
                Path(__file__).resolve().parents[1]
                / "src/usp_attention/online_flash_amd"
            )
            source_hash = hashlib.sha256(
                b"".join(
                    p.read_bytes()
                    for p in sorted(ext.rglob("*"))
                    if p.suffix in (".py", ".hip", ".h", ".hpp", ".cpp")
                    and "dist" not in p.parts
                )
            ).hexdigest()
            median = statistics.median([x for trial in trials for x in trial])
            row = dict(
                config=name,
                ring=r,
                ulysses=u,
                dp=world // (r * u),
                world=world,
                forward_only=True,
                seq_len=args.seq_len,
                padded_seq_len=n,
                heads=args.heads,
                kv_heads=args.kv_heads,
                layout=args.layout,
                query_head_factor=factor,
                kv_exchange=plan.kv_exchange,
                round_mode=args.round_mode,
                median_ms=median,
                trials_ms=trials,
                trial_medians_ms=[statistics.median(t) for t in trials],
                useful_tokens_per_second=args.seq_len
                * (world // (r * u))
                / (median / 1000),
                plan_and_shard_s=plan_s,
                max_rank_partial_bitmap_bytes=bitmap_bytes.item(),
                max_rank_plan_bytes=plan_bytes.item(),
                correctness_relative_l2=error,
                long_shape_sampled_relative_l2=sampled_error,
                long_shape_sampled_rows_per_rank=17,
                metadata_sha256=hashlib.sha256(aux.cpu().numpy().tobytes()).hexdigest(),
                extension_sha256=hashlib.sha256(
                    next(ext.glob("_C*.so")).read_bytes()
                ).hexdigest(),
                git_sha=git("rev-parse", "HEAD"),
                dirty=bool(git("status", "--porcelain")),
                amd_source_sha256=source_hash,
                timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                kernel=platform.release(),
                communication_env={
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith(("NCCL_", "RCCL_"))
                },
                torch=torch.__version__,
                rocm=torch.version.hip,
                rccl=torch.cuda.nccl.version(),
                gpu=str(torch.cuda.get_device_properties(device)),
            )
            print("RESULT " + json.dumps(row), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("a") as f:
                    f.write(json.dumps(row) + "\n")
        del local, plan, aux
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
