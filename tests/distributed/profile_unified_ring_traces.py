# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor
from torch.profiler import ProfilerActivity, profile

from usp_attention.fa4.flash4_attn import build_fa4_sparse_mask
from usp_attention.ring.flash4_ring_attn import (
    CROSS_QKV_SPARSE_MASK_MOD,
    Flash4RingStepMeta,
    build_precomputed_cross_qkv_aux_plan,
    build_ring_flash4_plan,
    get_rank_plan,
    ring_flash4_attn,
)
from usp_attention.ring.packed_layout import (
    AttentionMode,
    build_packed_layout,
)
from usp_attention.ring.ring_planning import RankRingPlan

TRACE_CASES = (
    "causal_imbalance",
    "mixed_sparse_ptrr_off",
    "mixed_sparse_ptrr_on",
    "transport_overlap",
)
TRANSPORTS = ("nccl_p2p", "symm", "symm_rotator", "symm_direct", "nccl4py")
TraceCase = Literal[
    "causal_imbalance",
    "mixed_sparse_ptrr_off",
    "mixed_sparse_ptrr_on",
    "transport_overlap",
]
Transport = Literal["nccl_p2p", "symm", "symm_rotator", "symm_direct", "nccl4py"]


@dataclass(kw_only=True, slots=True)
class CaseSpec:
    """Synthetic packed attention layout for one profiler trace."""

    document_lens: list[int]
    split_lens: list[int]
    attn_modes: list[AttentionMode]
    enable_ptrr: bool


@dataclass(kw_only=True, slots=True)
class TraceContext:
    """All metadata needed to run the ring attention trace."""

    step_meta: Flash4RingStepMeta
    rank_plan: RankRingPlan | None
    token_permutation: Tensor
    local_start: int
    local_end: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export named Perfetto traces for unified Flash4 ring attention "
            "presentation screenshots."
        )
    )
    parser.add_argument(
        "--case",
        choices=TRACE_CASES,
        default="transport_overlap",
        help="Synthetic packed attention case to trace.",
    )
    parser.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default="nccl_p2p",
        help="Ring KV transport backend.",
    )
    parser.add_argument(
        "--trace-name",
        default=None,
        help="Output trace basename. Defaults to '<case>_<transport>'.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("profiler_traces"),
        help="Directory for the merged .json.gz trace.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=1)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Global packed sequence length. Must divide by world_size * 128.",
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--backward",
        dest="backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include backward pass in the trace.",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Record tensor shapes in the torch profiler trace.",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Record Python stacks. Useful but expensive.",
    )
    parser.add_argument(
        "--keep-rank-traces",
        action="store_true",
        help="Keep temporary per-rank JSON traces after merging.",
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _sanitize_trace_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not sanitized:
        raise ValueError("trace name must contain at least one safe character.")
    return sanitized


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    block_multiple = world_size * 128
    if args.seq_len % block_multiple != 0:
        raise ValueError(
            "--seq-len must be divisible by world_size * 128 for FA4 block "
            f"planning, got seq_len={args.seq_len}, world_size={world_size}."
        )
    if args.warmup < 0 or args.active <= 0:
        raise ValueError(
            f"Expected --warmup >= 0 and --active > 0, got {args.warmup=} "
            f"{args.active=}."
        )
    if args.heads <= 0 or args.head_dim <= 0:
        raise ValueError(
            f"Expected positive head dimensions, got {args.heads=} {args.head_dim=}."
        )
    if args.transport != "nccl_p2p" and world_size == 1:
        raise ValueError(f"Transport {args.transport!r} requires at least two ranks.")


def _case_spec(case: TraceCase, seq_len: int) -> CaseSpec:
    if case == "causal_imbalance":
        return CaseSpec(
            document_lens=[seq_len],
            split_lens=[seq_len],
            attn_modes=["causal"],
            enable_ptrr=False,
        )
    if case == "transport_overlap":
        return CaseSpec(
            document_lens=[seq_len],
            split_lens=[seq_len],
            attn_modes=["full"],
            enable_ptrr=False,
        )

    quarter_blocks = seq_len // 128 // 4
    if quarter_blocks <= 0:
        raise ValueError("mixed sparse cases require at least four 128-token blocks.")
    first = quarter_blocks * 128
    second = quarter_blocks * 128
    third = quarter_blocks * 128
    fourth = seq_len - first - second - third
    enable_ptrr = case == "mixed_sparse_ptrr_on"
    return CaseSpec(
        document_lens=[seq_len // 2, seq_len - seq_len // 2],
        split_lens=[first, second, third, fourth],
        attn_modes=["full", "causal", "noise", "causal"],
        enable_ptrr=enable_ptrr,
    )


def _init_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("This trace harness requires CUDA GPUs.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Run this script with torchrun, e.g. "
            "`torchrun --nproc_per_node=4 .../profile_unified_ring_traces.py`."
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, torch.device("cuda", local_rank)


def _build_trace_context(
    *,
    spec: CaseSpec,
    seq_len: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> TraceContext:
    fa4_mask = build_fa4_sparse_mask(
        spec.document_lens,
        spec.split_lens,
        spec.attn_modes,
        device=device,
        return_sparsity=True,
        retain_block_mask=True,
    )
    layout = build_packed_layout(
        spec.document_lens,
        spec.split_lens,
        spec.attn_modes,
        device=device,
        aux_tensors=fa4_mask.aux_tensors,
    )

    rank_plan = None
    if spec.enable_ptrr:
        ring_plan = build_ring_flash4_plan(
            layout,
            world_size=world_size,
            device=device,
            block_mask=fa4_mask.block_mask,
        )
        token_permutation = ring_plan.token_permutation.to(device=device)
        rank_plan = get_rank_plan(ring_plan, rank=rank)
    else:
        token_permutation = torch.arange(seq_len, dtype=torch.long, device=device)

    ring_aux_tensors = [
        aux[token_permutation].contiguous() for aux in fa4_mask.aux_tensors
    ]
    ring_position_ids = token_permutation.to(dtype=torch.int64).contiguous()
    local_len = seq_len // world_size
    precomputed_aux_plan = build_precomputed_cross_qkv_aux_plan(
        global_aux_tensors=ring_aux_tensors,
        rank=rank,
        world_size=world_size,
        local_len=local_len,
        global_position_ids=ring_position_ids,
    )
    step_meta = Flash4RingStepMeta(
        mask_mod=CROSS_QKV_SPARSE_MASK_MOD,
        precomputed_aux_plan=precomputed_aux_plan,
    )
    local_start = rank * local_len
    return TraceContext(
        step_meta=step_meta,
        rank_plan=rank_plan,
        token_permutation=token_permutation,
        local_start=local_start,
        local_end=local_start + local_len,
    )


def _make_local_qkv(
    *,
    args: argparse.Namespace,
    context: TraceContext,
    dtype: torch.dtype,
    device: torch.device,
    seed_offset: int,
) -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(args.seed + seed_offset)
    shape = (1, args.seq_len, args.heads, args.head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    q = q[:, context.token_permutation].contiguous()
    k = k[:, context.token_permutation].contiguous()
    v = v[:, context.token_permutation].contiguous()
    return (
        q[:, context.local_start : context.local_end].contiguous(),
        k[:, context.local_start : context.local_end].contiguous(),
        v[:, context.local_start : context.local_end].contiguous(),
    )


def _run_attention_once(
    *,
    args: argparse.Namespace,
    context: TraceContext,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    iteration: int,
) -> None:
    with torch.autograd.profiler.record_function(
        f"trace_iter_prepare_inputs/iter={iteration}"
    ):
        q_local, k_local, v_local = _make_local_qkv(
            args=args,
            context=context,
            dtype=dtype,
            device=device,
            seed_offset=iteration,
        )
        if args.backward:
            q_local.requires_grad_(True)
            k_local.requires_grad_(True)
            v_local.requires_grad_(True)

    with torch.autograd.profiler.record_function(f"trace_iter/iter={iteration}"):
        out = ring_flash4_attn(
            q_local,
            k_local,
            v_local,
            process_group=process_group,
            step_meta=context.step_meta,
            rank_plan=context.rank_plan,
            input_is_sharded=True,
            gather_output=False,
            use_cuda_comm_stream=True,
            combine_mode="auto",
            use_symm_mem=args.transport in {"symm", "symm_rotator", "symm_direct"},
            use_symm_mem_rotator=args.transport == "symm_rotator",
            use_symm_mem_direct=args.transport == "symm_direct",
            use_nccl4py_ring=args.transport == "nccl4py",
        )
        if isinstance(out, tuple):
            out = out[0]
        if args.backward:
            loss = out.float().square().mean()
            loss.backward()
    torch.cuda.synchronize(device)


def _merge_rank_traces(
    *,
    trace_dir: Path,
    output_path: Path,
    world_size: int,
) -> None:
    merged_events: list[dict] = []
    metadata: list[dict] = []
    for rank in range(world_size):
        rank_path = trace_dir / f"rank{rank}.json"
        with open(rank_path, encoding="utf-8") as f:
            data = json.load(f)
        for event in data.get("traceEvents", []):
            event["pid"] = rank
            merged_events.append(event)
        metadata.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": rank,
                "args": {"name": f"rank {rank}"},
            }
        )
        metadata.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": rank,
                "args": {"sort_index": rank},
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": metadata + merged_events}, f)


def main() -> None:
    args = _parse_args()
    rank, world_size, device = _init_distributed()
    _validate_args(args, world_size)

    trace_name = _sanitize_trace_name(
        args.trace_name or f"{args.case}_{args.transport}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out_dir / f".{trace_name}_rank_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    dtype = _dtype_from_name(args.dtype)
    spec = _case_spec(args.case, args.seq_len)
    context = _build_trace_context(
        spec=spec,
        seq_len=args.seq_len,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    for iteration in range(args.warmup):
        _run_attention_once(
            args=args,
            context=context,
            dtype=dtype,
            device=device,
            process_group=dist.group.WORLD,
            iteration=iteration,
        )
    dist.barrier()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    rank_trace_path = trace_dir / f"rank{rank}.json"
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        with_stack=args.with_stack,
        acc_events=True,
    ) as prof:
        for active_iter in range(args.active):
            _run_attention_once(
                args=args,
                context=context,
                dtype=dtype,
                device=device,
                process_group=dist.group.WORLD,
                iteration=args.warmup + active_iter,
            )
            prof.step()
    prof.export_chrome_trace(str(rank_trace_path))
    dist.barrier()

    if rank == 0:
        output_path = args.out_dir / f"{trace_name}.json.gz"
        _merge_rank_traces(
            trace_dir=trace_dir,
            output_path=output_path,
            world_size=world_size,
        )
        if not args.keep_rank_traces:
            shutil.rmtree(trace_dir)
        print(f"Wrote merged Perfetto trace: {output_path}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
