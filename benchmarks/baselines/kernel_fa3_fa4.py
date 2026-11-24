# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Single-GPU FA3 / FA4 / cuDNN kernel time on the per-GPU shape of a CP benchmark.

Ulysses over ``U`` GPUs leaves each GPU one attention call over the whole
sequence with ``heads / U`` heads, so comparing the two kernels on that shape
isolates the kernel's share of an end-to-end difference. Run once per
environment (FA3 and FA4 are usually installed separately)::

    python benchmarks/baselines/kernel_fa3_fa4.py --impl fa3 --seq-len 131072 --heads 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import result_json  # noqa: E402


def _attn_fn(impl: str):
    if impl == "cudnn":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def cudnn(q, k, v, causal):
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=causal,
                ).transpose(1, 2)

        return cudnn
    if impl == "fa3":
        from flash_attn_interface import flash_attn_func

        return lambda q, k, v, causal: flash_attn_func(q, k, v, causal=causal)
    from flash_attn.cute.interface import flash_attn_func

    return lambda q, k, v, causal: flash_attn_func(q, k, v, causal=causal)[0]


def main() -> None:
    """Benchmark the selected attention kernel and print stamped timing rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--impl", choices=["fa3", "fa4", "cudnn"], required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--layout", choices=["causal", "full"], default="causal")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    attn = _attn_fn(args.impl)
    causal = args.layout == "causal"
    shape = (1, args.seq_len, args.heads, args.head_dim)
    q, k, v = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    dout = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def run(backward: bool) -> float:
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        out = attn(q, k, v, causal)
        if backward:
            out.backward(dout)
        e.record()
        e.synchronize()
        q.grad = k.grad = v.grad = None
        return s.elapsed_time(e)

    result = {"impl": args.impl, "layout": args.layout, "seq_len": args.seq_len}
    for name, backward in (("fwd_ms", False), ("fwd_bwd_ms", True)):
        for _ in range(args.warmup):
            run(backward)
        result[name] = statistics.median(run(backward) for _ in range(args.iters))
    pairs = args.seq_len * (args.seq_len + 1) // 2 if causal else args.seq_len**2
    flops = 4 * args.heads * args.head_dim * pairs
    result["fwd_tflops"] = flops / (result["fwd_ms"] * 1e-3) / 1e12
    result["fwd_bwd_tflops"] = 3.5 * flops / (result["fwd_bwd_ms"] * 1e-3) / 1e12
    print("RESULT " + result_json(result), flush=True)


if __name__ == "__main__":
    main()
