# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Forward-only AMD benchmark of the article's random-weight 28-layer MoT decoder.

Run from projects/usp_attention with PYTHONPATH=src .venv/bin/python -m
 torch.distributed.run --standalone --nproc-per-node=8
 benchmarks/bench_amd_model_forward.py --seq-len 131072.

Times the decoder stack, including FSDP weight gathers, projections, RoPE,
attention communication, norms, residuals and MLPs. No embeddings, modality
encoders, output heads, backward or optimizer are included. Plans and input
preparation are cached outside timing. This is packed full-sequence forward,
not autoregressive generation. Weights and hidden inputs are synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from bench_amd_sparse_forward import correctness, git
from bench_model import MoTConfig, MoTDecoder, _rope
from torch.distributed.fsdp import fully_shard

from usp_attention.distributed import USPGroups, init_distributed
from usp_attention.layouts import build_layout
from usp_attention.online_flash_amd.distributed import (
    build_forward_plan,
    sparse_usp_forward,
)
from usp_attention.ring.packed_layout import build_packed_layout


class ForwardDecoder(MoTDecoder):
    """Reuse the article's layers without activation checkpointing in inference."""

    def forward(self, x, und, gen, cos, sin, attn):
        for layer in self.layers:
            x = layer(x, und, gen, cos, sin, attn)
        return x


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=131072)
    p.add_argument("--layers", type=int, default=28)
    p.add_argument("--layout", default="mixed")
    p.add_argument(
        "--config", choices=("ring_contiguous", "ring_ptrr", "usp"), default="usp"
    )
    p.add_argument("--kv-exchange", choices=("ring", "gather"), default="gather")
    p.add_argument("--no-pack-mqa", action="store_true")
    p.add_argument("--round-mode", choices=("rtne", "rtna", "rtz"), default="rtna")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if min(args.seq_len, args.layers, args.trials, args.iterations) < 1:
        p.error("sequence length, layers, trials and iterations must be positive")
    if args.warmup < 0:
        p.error("warmup must be nonnegative")
    if not torch.version.hip:
        p.error("this benchmark requires ROCm")
    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    u = 4 if args.config == "usp" else 1
    if world not in (1, 2, 4, 8, 16):
        raise ValueError("supported world sizes: 1, 2, 4, 8, 16")
    if world % u:
        raise ValueError("USP requires a world size divisible by four")
    groups = USPGroups.create(ring_size=world // u, ulysses_size=u)
    cfg = MoTConfig(layers=args.layers)
    # Reuse the attention benchmark's all-row, distributed FP32-reference gate.
    args.heads, args.kv_heads = cfg.heads, cfg.kv_heads
    args.balance = args.config != "ring_contiguous"
    args.pack_mqa = not args.no_pack_mqa
    if u == 1:
        args.kv_exchange = "ring"
    reference_error = correctness(groups, args, device)

    docs, splits, modes = build_layout(
        args.layout, args.seq_len, ring_size=max(groups.ring_size, 2), seed=args.seed
    )
    packed = build_packed_layout(docs, splits, modes, device=device)
    aux = torch.stack(
        [
            packed.full_and_noise_seq_id,
            packed.noise_seq_id,
            packed.document_id,
            torch.arange(args.seq_len, device=device),
        ]
    ).int()
    multiple = 384 * world
    n = ((args.seq_len + multiple - 1) // multiple) * multiple
    if n > args.seq_len:
        tail = aux.new_full((4, n - args.seq_len), -1)
        tail[2] = int(aux[2].max()) + 1
        tail[3] = torch.arange(args.seq_len, n, device=device)
        aux = torch.cat((aux, tail), dim=1)
    torch.cuda.synchronize()
    start = time.perf_counter()
    factor = cfg.heads // u if args.pack_mqa and u == cfg.kv_heads else 1
    plan = build_forward_plan(
        aux,
        groups,
        balance=args.balance,
        query_head_factor=factor,
        kv_exchange=args.kv_exchange,
    )
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - start) * 1000
    ids = plan.permutation.chunk(world)[groups.cp_rank]
    local_gen = aux[1, ids] >= 0
    und, gen = (~local_gen).nonzero().flatten(), local_gen.nonzero().flatten()
    cos, sin = _rope(aux[3, ids], cfg)

    torch.manual_seed(args.seed)
    with torch.device("meta"):
        model = ForwardDecoder(cfg, args.layers).to(torch.bfloat16)
    for layer in model.layers:
        fully_shard(layer, reshard_after_forward=True)
    fully_shard(model, reshard_after_forward=True)
    model.to_empty(device=device)
    model.eval()
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.zero_()
        elif param.ndim == 1:
            param.fill_(1)
        else:
            param.normal_(0, 0.02)
    rng = torch.Generator(device=device).manual_seed(args.seed + rank)
    x = torch.randn(
        n // world, cfg.hidden, device=device, dtype=torch.bfloat16, generator=rng
    )

    def attention(q, k, v):
        # The model splits packed QKV projections into strided views. RCCL
        # P2P requires dense buffers; include input packing in every timed path.
        return sparse_usp_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            plan,
            round_mode=args.round_mode,
        )

    def step():
        return model(x, und, gen, cos, sin, attention)

    # Verify a complete decoder pass before recording any speed number.
    out = step()
    valid = torch.isfinite(out).all().int()
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise AssertionError("decoder output contains NaN or infinity")
    del out
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_times, gpu_times = [], []
    for _ in range(args.trials):
        for _ in range(args.iterations):
            dist.barrier()
            torch.cuda.synchronize()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start = time.perf_counter()
            begin.record()
            out = step()
            end.record()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            measurements = torch.tensor(
                [elapsed, begin.elapsed_time(end)], device=device
            )
            dist.all_reduce(measurements, op=dist.ReduceOp.MAX)
            wall_times.append(measurements[0].item())
            gpu_times.append(measurements[1].item())
            del out
    peak = torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        median = statistics.median(wall_times)
        root = Path(__file__).resolve().parents[1]
        ext = root / "src/usp_attention/online_flash_amd"
        sources = [Path(__file__), Path(__file__).with_name("bench_model.py")]
        sources += sorted(
            path
            for path in ext.rglob("*")
            if path.suffix in (".py", ".hip", ".h", ".hpp", ".cpp")
            and "dist" not in path.parts
        )
        source_hash = hashlib.sha256(
            b"".join(path.read_bytes() for path in sources)
        ).hexdigest()
        from usp_attention.online_flash_amd import _C

        extension_hash = hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest()
        row = {
            "bench": "amd_model_forward",
            "model": "mot7b",
            "phase": "forward",
            "random_weights": True,
            "config": args.config,
            "layers": args.layers,
            "seq_len": args.seq_len,
            "padded_seq_len": n,
            "layout": args.layout,
            "world": world,
            "ring": groups.ring_size,
            "ulysses": u,
            "kv_exchange": args.kv_exchange,
            "query_head_factor": factor,
            "round_mode": args.round_mode,
            "seed": args.seed,
            "fsdp": True,
            "reshard_after_forward": True,
            "cached_plan": True,
            "warmup": args.warmup,
            "trials": args.trials,
            "iterations": args.iterations,
            "forward_ms_median": median,
            "forward_ms_all": wall_times,
            "gpu_ms_median": statistics.median(gpu_times),
            "gpu_ms_all": gpu_times,
            "tokens_per_s": args.seq_len / (median / 1000),
            "peak_mem_gib_max_rank": peak.item(),
            "plan_setup_ms_rank0": setup_ms,
            "attention_reference_relative_l2": reference_error,
            "decoder_output_finite": True,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(device),
            "source_sha256": source_hash,
            "extension_sha256": extension_hash,
            "metadata_sha256": hashlib.sha256(aux.cpu().numpy().tobytes()).hexdigest(),
            "git_sha": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
        }
        print("RESULT " + json.dumps(row), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
