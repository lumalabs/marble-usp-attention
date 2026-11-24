# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Distributed correctness check for usp_attention against single-GPU FA4 and fp32.

Run with torchrun on one node, e.g.::

    torchrun --nproc-per-node 8 tests/distributed/check_usp.py \
        --ring 2 --ulysses 4 --layout mixed --seq-len 32768 --combine online flash

Every rank builds the same global Q/K/V and upstream gradient. For each combine
mode the script shards them, runs ``usp_attention`` forward and backward,
gathers output and gradients, and compares them on rank 0 against

- ``fa4``: one FA4 call on the full sequence (bf16, the no-parallelism result),
- ``fp32``: the dense fp32 reference (skipped above ``--max-reference-len``).

Each comparison prints one JSON line prefixed with ``RESULT``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from usp_attention import (
    USPConfig,
    USPGroups,
    build_usp_plan,
    init_distributed,
    pad_packed_layout,
    usp_attention,
)
from usp_attention.fa4.flash4_attn import (
    build_fa4_sparse_mask,
    flash4_attn,
    flash4_attn_dense,
)
from usp_attention.layouts import LAYOUT_NAMES, build_layout
from usp_attention.reference import reference_attention

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from provenance import result_json  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--ring", type=int, required=True)
    p.add_argument("--ulysses", type=int, required=True)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument(
        "--pad",
        action="store_true",
        help="pad the layout with an isolated causal document (pad_packed_layout) "
        "when --seq-len is not a multiple of ring * ulysses * 128",
    )
    p.add_argument(
        "--layout",
        default="mixed",
        help=f"one of {LAYOUT_NAMES} or file:<path>[#index] (a packed-layout JSON)",
    )
    p.add_argument("--dense-min-span", type=int, default=512)
    p.add_argument("--dense-ring-chunks", type=int, default=None)
    p.add_argument("--no-dense-ring-refine", action="store_true")
    p.add_argument("--dense-ring-inplace", action="store_true")
    p.add_argument("--dense-ring-dkv-overlap", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument(
        "--combine",
        nargs="+",
        default=["online", "flash", "python"],
        choices=["online", "flash", "python", "fp32"],
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
        help="Without a ring, keep the position-aware predicate for vision-group layouts.",
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
        help="gather: hide the halo exchanges behind two online-kernel calls (H100)",
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
    p.add_argument("--max-reference-len", type=int, default=32768)
    p.add_argument(
        "--no-fa4-reference",
        action="store_true",
        help="Skip the single-GPU block-sparse FA4 reference (compare against fp32 "
        "only), e.g. on GPUs whose FA4 block sparsity needs other block sizes.",
    )
    p.add_argument(
        "--sliding-window",
        type=int,
        default=None,
        help="Check a sliding-window plan (causal keys limited to the last N).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _errors(name: str, got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    if not torch.isfinite(got).all() or not torch.isfinite(want).all():
        raise ValueError(f"{name}: nonfinite output or reference")
    diff = (got.float() - want.float()).abs()
    finite = torch.isfinite(want.float())
    diff = torch.where(finite, diff, torch.zeros_like(diff))
    scale = (
        want.float().abs().where(finite, torch.zeros_like(diff)).mean().clamp_min(1e-12)
    )
    return {
        f"{name}_max_abs": diff.max().item(),
        f"{name}_mean_abs": diff.mean().item(),
        f"{name}_mean_rel": (diff.mean() / scale).item(),
    }


def main() -> None:
    # CuTe DSL installs its own sys.excepthook on import, which hides
    # uncaught exceptions from torchrun workers; restore the default.
    """Compare distributed attention against reference outputs and gradients.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__
    args = _parse_args()
    device = init_distributed()
    rank = dist.get_rank()
    groups = USPGroups.create(
        ring_size=args.ring,
        ulysses_size=args.ulysses,
        symm_mem_copy_engine=args.transport.startswith("symm_mem")
        and args.symm_mem_backend == "nccl",
    )
    docs, splits, modes = build_layout(
        args.layout, args.seq_len, ring_size=args.ring, seed=args.seed
    )
    if args.pad:
        docs, splits, modes = pad_packed_layout(
            docs, splits, modes, multiple=args.ring * args.ulysses * 128
        )
        args.seq_len = sum(docs)
    kv_heads = args.kv_heads or args.heads

    gen = torch.Generator(device=device).manual_seed(args.seed)
    shape_q = (args.batch, args.seq_len, args.heads, args.head_dim)
    shape_kv = (args.batch, args.seq_len, kv_heads, args.head_dim)
    q = torch.randn(shape_q, device=device, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(shape_kv, device=device, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(shape_kv, device=device, dtype=torch.bfloat16, generator=gen)
    dout = torch.randn(shape_q, device=device, dtype=torch.bfloat16, generator=gen)

    references: dict[str, tuple[torch.Tensor, ...]] = {}
    if rank == 0:
        if not args.no_fa4_reference:
            qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
            native_single = len(splits) == 1 and modes[0] in ("causal", "full")
            native_packed = (
                args.head_dim == 256
                and splits == docs
                and len(set(modes)) == 1
                and modes[0] in ("causal", "full")
            )
            if native_single or native_packed:
                # Native kernels also cover head_dim=256, where Hopper's
                # block-sparse reference does not support 128-token KV tiles.
                out = flash4_attn_dense(
                    qr,
                    kr,
                    vr,
                    causal=modes[0] == "causal",
                    sliding_window=args.sliding_window,
                    cu_seqlens=(
                        torch.tensor([0, *docs], device=device, dtype=torch.int32)
                        .cumsum(0)
                        .to(torch.int32)
                        if len(docs) > 1
                        else None
                    ),
                    max_seqlen=max(docs) if len(docs) > 1 else None,
                )
            else:
                mask = build_fa4_sparse_mask(
                    docs,
                    splits,
                    modes,
                    device=device,
                    sliding_window=args.sliding_window,
                )
                out = flash4_attn(qr, kr, vr, mask)
            out.backward(dout)
            references["fa4"] = (out.detach(), qr.grad, kr.grad, vr.grad)
        if args.seq_len <= args.max_reference_len:
            qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
            out, _ = reference_attention(
                qr,
                kr,
                vr,
                document_lens=docs,
                split_lens=splits,
                attn_modes=modes,
                dtype=torch.float32,
                sliding_window=args.sliding_window,
            )
            out.backward(dout.float())
            references["fp32"] = (out.detach(), qr.grad, kr.grad, vr.grad)

    for combine in args.combine:
        config = USPConfig(
            combine_mode=combine,  # type: ignore[arg-type]
            load_balance=not args.no_load_balance,
            balance=args.balance,
            kv_exchange=args.kv_exchange,
            fa3_gather=args.fa3_gather,
            gather_regather=args.gather_regather,
            halo_overlap=args.halo_overlap,
            exact_dense_ring=args.exact_dense_ring,
            dense_min_span=args.dense_min_span,
            dense_ring_chunks=args.dense_ring_chunks,
            dense_ring_refine=not args.no_dense_ring_refine,
            dense_ring_inplace=args.dense_ring_inplace,
            dense_ring_dkv_overlap=args.dense_ring_dkv_overlap,
            reorder_vision_groups=not args.no_group_reorder,
            column_range_masks=not args.no_column_ranges,
            native_plain_masks=not args.no_native_plain_masks,
            fa3_on_hopper=not args.no_fa3,
            interval_masks={"auto": None, "on": True, "off": False}[
                args.interval_masks
            ],
            dense_ring={"auto": None, "on": True, "off": False}[args.dense_ring],
            fp32_grads=not args.bf16_grads,
            transport=args.transport,  # type: ignore[arg-type]
        )
        plan = build_usp_plan(
            document_lens=docs,
            split_lens=splits,
            attn_modes=modes,
            groups=groups,
            config=config,
            device=device,
            sliding_window=args.sliding_window,
        )
        ql, kl, vl = (plan.shard(t).clone().requires_grad_(True) for t in (q, k, v))
        out_local = usp_attention(ql, kl, vl, plan)
        out_local.backward(plan.shard(dout))
        gathered = [
            plan.gather(t) for t in (out_local.detach(), ql.grad, kl.grad, vl.grad)
        ]

        if rank == 0:
            for ref_name in ("fa4", "fp32"):
                if ref_name not in references:
                    continue
                row: dict[str, object] = {
                    "check": "usp",
                    "reference": ref_name,
                    "combine": combine,
                    "ring": args.ring,
                    "ulysses": args.ulysses,
                    "layout": args.layout,
                    "seq_len": args.seq_len,
                    "heads": args.heads,
                    "kv_heads": kv_heads,
                    "head_dim": args.head_dim,
                    "load_balance": config.load_balance,
                    "fp32_grads": config.fp32_grads,
                    "transport": config.transport,
                    "native_mask": plan.native_mask,
                    "fa3_on_hopper": not args.no_fa3,
                    "interval_masks": args.interval_masks,
                    "interval": plan.interval_plan is not None,
                    "dense_ring": plan.dense_ring is not None,
                    "halo": plan.halo is not None,
                    "dense_gather": plan.dense_gather is not None,
                    "generation_ring": plan.generation_ring is not None,
                    "balance": args.balance,
                    "kv_exchange": args.kv_exchange,
                    "fa3_gather": args.fa3_gather,
                    "gather_regather": args.gather_regather,
                    "halo_overlap": args.halo_overlap,
                    "exact_dense_ring": args.exact_dense_ring,
                    "batch": args.batch,
                    "sliding_window": args.sliding_window,
                    "block_sparse": plan.fa4_mask is not None
                    or (plan.step_meta is not None and plan.step_meta.native is None),
                }
                for name, got, want in zip(
                    ("out", "dq", "dk", "dv"), gathered, references[ref_name]
                ):
                    row.update(_errors(name, got, want))
                print("RESULT " + result_json(row), flush=True)
        dist.barrier()

    if rank == 0 and "fp32" in references and "fa4" in references:
        row = {
            "check": "fa4_single_gpu",
            "reference": "fp32",
            "layout": args.layout,
            "seq_len": args.seq_len,
            "sliding_window": args.sliding_window,
        }
        for name, got, want in zip(
            ("out", "dq", "dk", "dv"), references["fa4"], references["fp32"]
        ):
            row.update(_errors(name, got, want))
        print("RESULT " + result_json(row), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
