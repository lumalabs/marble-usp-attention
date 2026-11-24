# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-FileCopyrightText: Jiarui Fang and the YunChang contributors (_patch_fa3_causal)
# SPDX-License-Identifier: MIT AND Apache-2.0
"""yunchang (the reference USP implementation used by xDiT) attention baseline.

Runs ``yunchang.LongContextAttention`` (Ulysses x ring) forward + backward on
local shards and prints a ``RESULT {json}`` line with the same timing and FLOP
accounting as ``bench_attention.py``. yunchang supports plain causal or full
masks only. yunchang 0.6.4's FA3 wrappers pass ``causal=False`` to the kernel
regardless of the requested mask, so causal runs compute full attention;
:func:`_patch_fa3_causal` makes them honor the flag. Its two wrappers are
adapted from yunchang's ``flash_attn3_func_{forward,backward}`` (Apache-2.0,
see THIRD_PARTY_NOTICES.md), changed only to pass ``causal`` through. Self-contained so it runs
in a separate environment::

    torchrun --nproc-per-node 8 benchmarks/baselines/yunchang_usp.py \
        --ring 2 --ulysses 4 --layout causal --seq-len 131072
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import result_json  # noqa: E402


def _patch_fa3_causal() -> None:
    """Make yunchang's FA3 forward / backward wrappers pass ``causal`` through.

    Same calls as ``yunchang.kernels.attention.flash_attn3_func_{forward,backward}``
    except for the mask flag. yunchang resolves these through
    ``yunchang.kernels`` at call time, so rebinding them there is enough.
    """
    import yunchang.kernels as yk
    from yunchang.kernels import attention as ya

    def forward(q, k, v, dropout_p, softmax_scale, causal, *args, **kwargs):
        out, lse, *_ = ya.flash_attn_forward_hopper(
            q=q,
            k=k,
            v=v,
            k_new=None,
            v_new=None,
            qv=None,
            out=None,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            page_table=None,
            kv_batch_idx=None,
            leftpad_k=None,
            rotary_cos=None,
            rotary_sin=None,
            seqlens_rotary=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            rotary_interleaved=True,
            scheduler_metadata=None,
            num_splits=0,
            pack_gqa=None,
            sm_margin=0,
        )
        return out, lse

    def backward(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dq,
        dk,
        dv,
        dropout_p,
        softmax_scale,
        causal,
        *args,
        **kwargs,
    ):
        ya.flash_attn_func_hopper_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            sequed_q=None,
            sequed_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            dq=dq,
            dk=dk,
            dv=dv,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            softcap=0.0,
            deterministic=False,
            sm_margin=0,
        )

    yk.flash_attn3_func_forward = forward
    yk.flash_attn3_func_backward = backward


def _check(args, attn, causal, ring_impl, extract_funcs, device) -> None:
    """Print the output error of yunchang vs single-GPU SDPA on the full sequence."""
    rank, world = dist.get_rank(), dist.get_world_size()
    kv_heads = args.kv_heads or args.heads
    gen = torch.Generator(device=device).manual_seed(0)
    q, k, v = (
        torch.randn(
            1,
            args.seq_len,
            h,
            args.head_dim,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        for h in (args.heads, kv_heads, kv_heads)
    )
    extract = extract_funcs[ring_impl]

    def local(t):
        return extract(t, rank, world, rd=args.ring, ud=args.ulysses)

    full = [t.float().requires_grad_(True) for t in (q, k, v)]
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in full),
        is_causal=causal,
        enable_gqa=kv_heads != args.heads,
    ).transpose(1, 2)
    dout = torch.randn(ref.shape, device=device, generator=gen)
    ref.backward(dout)
    shards = [local(t).requires_grad_(True) for t in (q, k, v)]
    out = attn(*shards, causal=causal)
    out.backward(local(dout).to(out.dtype))
    pairs = [(out, ref)] + [(s.grad, f.grad) for s, f in zip(shards, full)]
    errs = torch.stack(
        [(a.float() - local(b)).abs().mean() / local(b).abs().mean() for a, b in pairs]
    )
    dist.all_reduce(errs)
    errs = (errs / world).tolist()
    if rank == 0:
        print(
            "CHECK "
            + result_json(
                {
                    "impl": f"yunchang_{args.attn_type}_{ring_impl}"
                    + ("" if args.no_causal_fix else "_patched"),
                    "layout": args.layout,
                    "seq_len": args.seq_len,
                    "ring": args.ring,
                    "ulysses": args.ulysses,
                    "out": errs[0],
                    "dq": errs[1],
                    "dk": errs[2],
                    "dv": errs[3],
                }
            ),
            flush=True,
        )


def main() -> None:
    """Benchmark yunchang sequence parallelism and emit measurement rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--ring", type=int, required=True)
    p.add_argument("--ulysses", type=int, required=True)
    p.add_argument("--layout", choices=["causal", "full"], required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument(
        "--attn-type",
        default="fa3",
        help="yunchang AttnType value (its PyTorch SDPA types have no backward)",
    )
    p.add_argument(
        "--ring-impl",
        default=None,
        help="basic | zigzag | strip (default: zigzag for causal, basic for full)",
    )
    p.add_argument(
        "--no-causal-fix",
        action="store_true",
        help="Run yunchang's FA3 wrappers unpatched (they ignore causal).",
    )
    p.add_argument("--check", action="store_true", help="Check output vs SDPA.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--label", default="")
    args = p.parse_args()

    from yunchang import EXTRACT_FUNC_DICT, LongContextAttention, set_seq_parallel_pg
    from yunchang.kernels import AttnType

    sys.excepthook = sys.__excepthook__
    if not args.no_causal_fix:
        _patch_fa3_causal()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    set_seq_parallel_pg(args.ulysses, args.ring, rank, world, use_ulysses_low=True)

    causal = args.layout == "causal"
    ring_impl = args.ring_impl or ("zigzag" if causal else "basic")
    attn = LongContextAttention(
        ring_impl_type=ring_impl, attn_type=AttnType(args.attn_type)
    )

    kv_heads = args.kv_heads or args.heads
    if args.check:
        _check(args, attn, causal, ring_impl, EXTRACT_FUNC_DICT, device)
        dist.destroy_process_group()
        return
    local = args.seq_len // world
    gen = torch.Generator(device=device).manual_seed(rank)
    q = torch.randn(
        1,
        local,
        args.heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    k = torch.randn(
        1,
        local,
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

    def step() -> None:
        out = attn(q, k, v, causal=causal)
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
        seq = args.seq_len
        pairs = seq * (seq + 1) // 2 if causal else seq * seq
        flops = 3.5 * 4 * args.heads * args.head_dim * pairs
        print(
            "RESULT "
            + result_json(
                {
                    "bench": "attention",
                    "impl": f"yunchang_{args.attn_type}_{ring_impl}",
                    "label": args.label,
                    "layout": args.layout,
                    "seq_len": seq,
                    "ring": args.ring,
                    "ulysses": args.ulysses,
                    "world": world,
                    "combine": "-",
                    "load_balance": ring_impl == "zigzag",
                    "transport": "nccl",
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
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
