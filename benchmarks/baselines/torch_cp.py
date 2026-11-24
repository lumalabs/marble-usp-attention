# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-FileCopyrightText: 2025 Bytedance Ltd. and/or its affiliates (packed-mask encoding adapted from BAGEL create_sparse_mask)
# SPDX-License-Identifier: MIT AND Apache-2.0
"""PyTorch-native context-parallel attention baselines (torch 2.11).

Same timing and FLOP accounting as ``bench_attention.py``, for:

- ``flex_allgather``: PyTorch's context parallelism for ``flex_attention``
  (the TorchTitan recipe). Q/K/V are reordered by PyTorch's PTRR load balancer
  over the global ``BlockMask``, each rank all-gathers the full K/V and runs
  compiled ``flex_attention`` for its query shard. Works for any layout.
- ``sdpa_ring``: ``context_parallel()`` with SDPA ring attention and its
  built-in head-tail load balancing. Causal and full layouts only.

Run with torchrun, e.g.::

    torchrun --nproc-per-node 8 benchmarks/baselines/torch_cp.py \
        --mode flex_allgather --layout mixed --seq-len 131072
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import _context_parallel as cp
from torch.distributed.tensor.experimental._context_parallel._attention import (
    _create_cp_block_mask,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

from usp_attention import init_distributed
from usp_attention.layouts import LAYOUT_NAMES, build_layout
from usp_attention.reference import visible_key_counts
from usp_attention.ring.packed_layout import build_packed_layout

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _layouts import dense_mask, dense_reference, mean_rel_errors  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import result_json  # noqa: E402


def _mask_mod(docs, splits, modes, device):
    """The packed mixed-mode predicate as a flex_attention mask_mod."""
    layout = build_packed_layout(docs, splits, modes, device=device)
    fn, noise, doc = (
        layout.full_and_noise_seq_id,
        layout.noise_seq_id,
        layout.document_id,
    )

    def causal(b, h, q, kv):
        return q >= kv

    def full_and_noise(b, h, q, kv):
        return (fn[q] == fn[kv]) & (fn[q] >= 0)

    def remove_noise(b, h, q, kv):
        return ~((noise[kv] >= 0) & (noise[q] != noise[kv]))

    def sample(b, h, q, kv):
        return doc[q] == doc[kv]

    return and_masks(or_masks(causal, full_and_noise), remove_noise, sample)


def main() -> None:
    """Benchmark the selected PyTorch context-parallel baseline from sys.argv.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--mode", choices=["flex_allgather", "sdpa_ring"], required=True)
    p.add_argument(
        "--layout",
        default="mixed",
        help=f"one of {LAYOUT_NAMES} or file:<path>[#index]",
    )
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--rotate", choices=["allgather", "alltoall"], default="allgather")
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

    device = init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))
    docs, splits, modes = build_layout(args.layout, args.seq_len, ring_size=world)
    seq, heads, hd = args.seq_len, args.heads, args.head_dim

    gen = torch.Generator(device=device).manual_seed(0)
    shape = (1, heads, seq, hd)  # BHSD, as SDPA and flex_attention expect
    q, k, v, dout = (
        torch.randn(shape, device=device, dtype=torch.bfloat16, generator=gen)
        for _ in range(4)
    )
    # The fp32 reference [out, dq, dk, dv] (BHSD), sharded alongside Q/K/V.
    refs: list[torch.Tensor] = []
    if args.check:
        refs = [
            r.transpose(1, 2).contiguous()
            for r in dense_reference(
                *(t.transpose(1, 2) for t in (q, k, v, dout)),
                dense_mask(docs, splits, modes, device=device),
            )
        ]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if args.mode == "flex_allgather":
        mask_mod = _mask_mod(docs, splits, modes, device)
        global_mask = create_block_mask(
            mask_mod, 1, 1, seq, seq, device=device, _compile=True
        )
        balancer = cp._PTRRLoadBalancer(global_mask, world)
        q, k, v, dout, *refs = cp._context_parallel_shard(
            mesh, [q, k, v, dout, *refs], [2] * (4 + len(refs)), balancer
        )
        local_mask = _create_cp_block_mask(mask_mod, 1, 1, seq, seq, mesh, balancer)
        compiled = torch.compile(flex_attention)
        pg = dist.distributed_c10d._get_process_group_name(mesh.get_group())
        for t in (q, k, v):
            t.requires_grad_(True)

        def forward() -> torch.Tensor:
            k_full, v_full = cp.flex_cp_allgather(k, v, 2, pg)
            return compiled(q, k_full, v_full, block_mask=local_mask)
    else:
        if args.layout not in ("causal", "full"):
            raise ValueError("sdpa_ring supports only the causal and full layouts")
        cp.set_rotate_method(args.rotate)
        is_causal = args.layout == "causal"
        # Head-tail load balancing only supports causal masks; a full mask is
        # balanced already.
        cp._cp_options.enable_load_balance = is_causal
        buffers = [q, k, v, dout, *refs]
        ctx = cp.context_parallel(
            mesh,
            buffers=buffers,
            buffer_seq_dims=[2] * len(buffers),
            no_restore_buffers=set(buffers),
        )
        # context_parallel shards the buffers in place on entry, which is not
        # allowed on leaves that already require grad.
        ctx.__enter__()
        for t in (q, k, v):
            t.requires_grad_(True)

        def forward() -> torch.Tensor:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    def step() -> None:
        forward().backward(dout)
        q.grad = k.grad = v.grad = None

    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - t0) * 1e3

    if args.check:
        out = forward()
        out.backward(dout)
        errors = torch.tensor(
            mean_rel_errors([out, q.grad, k.grad, v.grad], refs), device=device
        )
        dist.all_reduce(errors)
        errors = (errors / world).tolist()
        if rank == 0:
            print(
                "CHECK "
                + result_json(
                    {
                        "impl": f"torch_{args.mode}",
                        "layout": args.layout,
                        "seq_len": seq,
                        "out": errors[0],
                        "dq": errors[1],
                        "dk": errors[2],
                        "dv": errors[3],
                    }
                ),
                flush=True,
            )
        dist.destroy_process_group()
        return

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
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
        pairs = int(visible_key_counts(docs, splits, modes).sum())
        flops = 3.5 * 4 * heads * hd * pairs
        print(
            "RESULT "
            + result_json(
                {
                    "bench": "attention",
                    "impl": f"torch_{args.mode}",
                    "label": args.label,
                    "layout": args.layout,
                    "seq_len": seq,
                    "ring": world,
                    "ulysses": 1,
                    "world": world,
                    "combine": "-",
                    "load_balance": True,
                    "transport": args.rotate
                    if args.mode == "sdpa_ring"
                    else "allgather",
                    "heads": heads,
                    "kv_heads": heads,
                    "head_dim": hd,
                    "median_ms": median,
                    "p10_ms": times[len(times) // 10],
                    "p90_ms": times[(len(times) * 9) // 10],
                    "tflops_per_gpu": flops / (median * 1e-3) / world / 1e12,
                    "mask_density": pairs / seq**2,
                    "ring_imbalance": float("nan"),
                    "ring_imbalance_contiguous": float("nan"),
                    "plan_ms": setup_ms,
                    "peak_mem_gib": peak,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
