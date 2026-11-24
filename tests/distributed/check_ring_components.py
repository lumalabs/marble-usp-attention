# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Manual diagnostics for Flash4 ring-attention component correctness.

This module is intentionally kept under ``flash4.diagnostics`` because it is a
torchrun-only validation tool, not runtime model code.  It compares distributed
ring output/LSE/dq/dk/dv against full local Flash4 attention across combine
modes, block-sparse builders, NCCL4Py/P2P transports, and synthetic packed
long-context layouts.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention.distributed import init_distributed
from usp_attention.fa4.flash4_attn import (
    HAS_FLASH4,
    build_fa4_sparse_mask,
    flash4_attn,
)
from usp_attention.ring.flash4_ring_attn import (
    CROSS_QKV_SPARSE_MASK_MOD,
    Flash4RingStepMeta,
    RankRingPlan,
    build_precomputed_cross_qkv_aux_plan,
    build_step_block_sparse_direct,
    build_step_block_sparse_tensors_from_aux,
    get_rank_plan,
    ring_flash4_attn,
)
from usp_attention.ring.packed_layout import build_packed_layout
from usp_attention.ring.ring_planning import build_ring_plan

CombineMode = Literal["python", "flash", "online"]
BlockSparseMode = Literal["direct", "create_block_mask"]
TransportMode = Literal["p2p", "nccl4py", "symm_mem"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed Flash4 ring-attention component checker. Run from "
            "`projects/usp_attention` with torchrun."
        )
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layout",
        choices=("single_doc_mixed", "multi_doc_mixed", "many_docs_interleaved"),
        default="single_doc_mixed",
        help="Synthetic packed attention layout to validate.",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=32,
        help="Number of documents for layout=many_docs_interleaved.",
    )
    parser.add_argument(
        "--splits-per-doc",
        type=int,
        default=9,
        help="Number of attention-mode segments per document for many_docs_interleaved.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument(
        "--combine-mode",
        action="append",
        choices=("python", "flash", "online"),
        dest="combine_modes",
        help="May be passed multiple times. Defaults to all modes.",
    )
    parser.add_argument(
        "--block-sparse-mode",
        action="append",
        choices=("direct", "create_block_mask"),
        dest="block_sparse_modes",
        help="May be passed multiple times. Defaults to both modes.",
    )
    parser.add_argument(
        "--transport",
        action="append",
        choices=("p2p", "nccl4py", "symm_mem"),
        dest="transports",
        help="May be passed multiple times. Defaults to p2p.",
    )
    parser.add_argument("--output-atol", type=float, default=3e-2)
    parser.add_argument("--output-rtol", type=float, default=3e-2)
    parser.add_argument("--grad-atol", type=float, default=6e-2)
    parser.add_argument("--grad-rtol", type=float, default=6e-2)
    parser.add_argument(
        "--skip-backward",
        action="store_true",
        help="Only validate block sparse metadata and forward output/LSE.",
    )
    parser.add_argument(
        "--allow-online-non-sm90-skip",
        action="store_true",
        help="Skip combine_mode=online when the local CUDA device is not Hopper.",
    )
    parser.add_argument(
        "--enable-ring-load-balance",
        action="store_true",
        help=(
            "Apply the PTRR token permutation before sharding, matching the "
            "load-balanced Flash4 training path."
        ),
    )
    return parser.parse_args()


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _print_rank0(message: str) -> None:
    if _rank() == 0:
        print(message, flush=True)


def _set_local_cuda_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name!r}.")


def _default_modes(values: list[str] | None, default: Iterable[str]) -> list[str]:
    return list(values) if values is not None else list(default)


def _make_layout(
    seq_len: int,
    *,
    layout: str,
    num_docs: int,
    splits_per_doc: int,
) -> tuple[list[int], list[int], list[str]]:
    if layout == "many_docs_interleaved":
        if num_docs <= 0:
            raise ValueError(f"num_docs must be positive, got {num_docs=}.")
        if splits_per_doc <= 0:
            raise ValueError(f"splits_per_doc must be positive, got {splits_per_doc=}.")
        if seq_len < num_docs * splits_per_doc:
            raise ValueError(
                "seq_len must be large enough to give each split at least one token, "
                f"got {seq_len=} {num_docs=} {splits_per_doc=}."
            )
        base_doc_len = seq_len // num_docs
        document_lens = [
            base_doc_len + (1 if doc < seq_len % num_docs else 0)
            for doc in range(num_docs)
        ]
        split_lens: list[int] = []
        attn_modes: list[str] = []
        mode_cycle = ["causal", "full", "noise", "causal", "noise", "full"]
        for doc, doc_len in enumerate(document_lens):
            base_split_len = doc_len // splits_per_doc
            extra = doc_len % splits_per_doc
            for split in range(splits_per_doc):
                split_len = base_split_len + (1 if split < extra else 0)
                split_lens.append(split_len)
                attn_modes.append(mode_cycle[(doc + split) % len(mode_cycle)])
        return document_lens, split_lens, attn_modes
    if layout == "multi_doc_mixed":
        base = seq_len // 4
        document_lens = [base, base, base, seq_len - 3 * base]
        split_lens: list[int] = []
        attn_modes: list[str] = []
        for doc_len in document_lens:
            first = max(1, doc_len // 2)
            second = max(1, doc_len // 4)
            third = doc_len - first - second
            if third <= 0:
                split_lens.append(doc_len)
                attn_modes.append("causal")
                continue
            split_lens.extend([first, second, third])
            attn_modes.extend(["causal", "full", "noise"])
        return document_lens, split_lens, attn_modes
    if layout != "single_doc_mixed":
        raise ValueError(f"Unsupported layout: {layout!r}.")
    first = seq_len // 4
    second = seq_len // 4
    third = seq_len - first - second
    return [seq_len], [first, second, third], ["causal", "full", "noise"]


def _make_qkv(
    *,
    batch: int,
    seq_len: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch,
        seq_len,
        heads,
        head_dim,
        dtype=dtype,
        device=device,
        generator=gen,
    )
    k = torch.randn(
        batch,
        seq_len,
        kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
        generator=gen,
    )
    v = torch.randn(
        batch,
        seq_len,
        kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
        generator=gen,
    )
    weights = torch.randn(
        batch,
        seq_len,
        heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        generator=gen,
    )
    return q, k, v, weights


def _local_slice(x: Tensor) -> Tensor:
    chunks = x.chunk(_world_size(), dim=1)
    return chunks[_rank()].contiguous()


def _permute_seq(x: Tensor, token_permutation: Tensor | None) -> Tensor:
    if token_permutation is None:
        return x
    return torch.index_select(x, dim=1, index=token_permutation.to(x.device))


def _lse_to_bsh(lse: Tensor, *, batch: int, seq_len: int, heads: int) -> Tensor:
    if lse.shape == (batch, seq_len, heads):
        return lse
    if lse.shape == (batch, heads, seq_len):
        return lse.transpose(-1, -2).contiguous()
    raise ValueError(
        f"Unexpected LSE shape {tuple(lse.shape)}, expected "
        f"{(batch, seq_len, heads)} or {(batch, heads, seq_len)}."
    )


def _max_abs(x: Tensor) -> Tensor:
    return x.detach().float().abs().max()


def _global_max(value: Tensor) -> Tensor:
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def _assert_close_metric(
    *,
    label: str,
    abs_diff: Tensor,
    ref_abs: Tensor,
    atol: float,
    rtol: float,
) -> None:
    global_abs = _global_max(abs_diff)
    global_ref = _global_max(ref_abs)
    allowed = atol + rtol * float(global_ref.item())
    if _rank() == 0:
        print(
            f"{label}: max_abs={float(global_abs.item()):.6e}, "
            f"ref_abs={float(global_ref.item()):.6e}, allowed={allowed:.6e}",
            flush=True,
        )
    if float(global_abs.item()) > allowed:
        raise AssertionError(
            f"{label} exceeded tolerance: max_abs={float(global_abs.item()):.6e}, "
            f"ref_abs={float(global_ref.item()):.6e}, allowed={allowed:.6e}."
        )


def _ordered_to_set(
    counts: Tensor | None, indices: Tensor | None, row: int
) -> set[int]:
    if counts is None or indices is None:
        return set()
    flat_counts = counts.flatten()
    flat_indices = indices.reshape(flat_counts.shape[0], -1)
    count = int(flat_counts[row].item())
    return set(int(v) for v in flat_indices[row, :count].tolist())


def _check_sparse_pair(
    *,
    label: str,
    direct_counts: Tensor | None,
    direct_indices: Tensor | None,
    direct_full_counts: Tensor | None,
    direct_full_indices: Tensor | None,
    reference_counts: Tensor | None,
    reference_indices: Tensor | None,
    reference_full_counts: Tensor | None,
    reference_full_indices: Tensor | None,
) -> None:
    if reference_counts is None or direct_counts is None:
        return
    rows = int(reference_counts.numel())
    for row in range(rows):
        direct_partial = _ordered_to_set(direct_counts, direct_indices, row)
        direct_full = _ordered_to_set(direct_full_counts, direct_full_indices, row)
        reference_partial = _ordered_to_set(reference_counts, reference_indices, row)
        reference_full = _ordered_to_set(
            reference_full_counts, reference_full_indices, row
        )
        direct_union = direct_partial | direct_full
        reference_union = reference_partial | reference_full
        if direct_union != reference_union:
            raise AssertionError(
                f"{label} row {row}: direct sparse union differs from "
                f"create_block_mask. missing={reference_union - direct_union}, "
                f"extra={direct_union - reference_union}."
            )
        if not direct_full <= reference_full:
            raise AssertionError(
                f"{label} row {row}: direct marks non-reference-full blocks as full: "
                f"{direct_full - reference_full}."
            )


def _check_block_sparse_components(step_meta: Flash4RingStepMeta) -> None:
    ok = torch.ones((), dtype=torch.int32, device=torch.cuda.current_device())
    error_message = ""
    try:
        for step in range(_world_size()):
            aux = step_meta.precomputed_aux_plan.aux_for_step(step)
            direct = build_step_block_sparse_direct(aux)
            reference = build_step_block_sparse_tensors_from_aux(aux)
            if direct.block_size != reference.block_size:
                raise AssertionError(
                    f"step={step}: block size mismatch {direct.block_size} "
                    f"!= {reference.block_size}."
                )
            _check_sparse_pair(
                label=f"step={step}/fwd",
                direct_counts=direct.mask_block_cnt,
                direct_indices=direct.mask_block_idx,
                direct_full_counts=direct.full_block_cnt,
                direct_full_indices=direct.full_block_idx,
                reference_counts=reference.mask_block_cnt,
                reference_indices=reference.mask_block_idx,
                reference_full_counts=reference.full_block_cnt,
                reference_full_indices=reference.full_block_idx,
            )
            _check_sparse_pair(
                label=f"step={step}/bwd",
                direct_counts=direct.bwd_mask_block_cnt,
                direct_indices=direct.bwd_mask_block_idx,
                direct_full_counts=direct.bwd_full_block_cnt,
                direct_full_indices=direct.bwd_full_block_idx,
                reference_counts=reference.bwd_mask_block_cnt,
                reference_indices=reference.bwd_mask_block_idx,
                reference_full_counts=reference.bwd_full_block_cnt,
                reference_full_indices=reference.bwd_full_block_idx,
            )
    except Exception as exc:
        ok.zero_()
        error_message = f"rank={_rank()}: {exc}"
    gathered_errors: list[str | None] = [None for _ in range(_world_size())]
    dist.all_gather_object(gathered_errors, error_message or None)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)  # ty: ignore[unresolved-attribute]
    if int(ok.item()) == 0:
        messages = [msg for msg in gathered_errors if msg]
        if _rank() == 0:
            for message in messages:
                print(f"block_sparse_components: FAIL {message}", flush=True)
        raise AssertionError("block_sparse_components failed.")
    _print_rank0("block_sparse_components: ok")


def _build_step_meta(
    *,
    global_aux_tensors: list[Tensor],
    global_position_ids: Tensor | None,
    seq_len: int,
    softmax_scale: float | None,
    block_sparse_mode: BlockSparseMode,
) -> Flash4RingStepMeta:
    local_len = seq_len // _world_size()
    precomputed_aux_plan = build_precomputed_cross_qkv_aux_plan(
        global_aux_tensors=global_aux_tensors,
        rank=_rank(),
        world_size=_world_size(),
        local_len=local_len,
        precompute_block_sparse=True,
        use_direct_block_sparse=block_sparse_mode == "direct",
        global_position_ids=global_position_ids,
    )
    return Flash4RingStepMeta(
        mask_mod=CROSS_QKV_SPARSE_MASK_MOD,
        precomputed_aux_plan=precomputed_aux_plan,
        softmax_scale=softmax_scale,
    )


def _run_reference(
    *,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    weights: Tensor,
    mask,
    softmax_scale: float | None,
    run_backward: bool,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    q_ref = q.detach().clone().requires_grad_(run_backward)
    k_ref = k.detach().clone().requires_grad_(run_backward)
    v_ref = v.detach().clone().requires_grad_(run_backward)
    out_ref, lse_ref = flash4_attn(
        q_ref,
        k_ref,
        v_ref,
        mask,
        softmax_scale=softmax_scale,
        return_lse=True,
    )
    if run_backward:
        (out_ref.float() * weights).sum().backward()
    return out_ref.detach(), lse_ref.detach(), q_ref.grad, k_ref.grad, v_ref.grad


def _run_ring_case(
    *,
    combine_mode: CombineMode,
    block_sparse_mode: BlockSparseMode,
    transport: TransportMode,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    weights: Tensor,
    ref_out: Tensor,
    ref_lse: Tensor,
    ref_dq: Tensor | None,
    ref_dk: Tensor | None,
    ref_dv: Tensor | None,
    step_meta: Flash4RingStepMeta,
    rank_plan: RankRingPlan | None,
    output_atol: float,
    output_rtol: float,
    grad_atol: float,
    grad_rtol: float,
    run_backward: bool,
) -> None:
    q_local = _local_slice(q).detach().clone().requires_grad_(run_backward)
    k_local = _local_slice(k).detach().clone().requires_grad_(run_backward)
    v_local = _local_slice(v).detach().clone().requires_grad_(run_backward)
    local_weights = _local_slice(weights)

    assert isinstance(dist.group.WORLD, dist.ProcessGroup), (
        "dist.group.WORLD is not a ProcessGroup"
    )

    pg: dist.ProcessGroup = dist.group.WORLD

    out_ring, lse_ring = ring_flash4_attn(
        q_local,
        k_local,
        v_local,
        process_group=pg,
        step_meta=step_meta,
        rank_plan=rank_plan,
        input_is_sharded=True,
        gather_output=False,
        combine_mode=combine_mode,
        return_lse=True,
        use_nccl4py_ring=transport == "nccl4py",
        use_symm_mem=transport == "symm_mem",
    )
    local_ref_out = _local_slice(ref_out)
    local_ref_lse = _local_slice(
        _lse_to_bsh(ref_lse, batch=q.shape[0], seq_len=q.shape[1], heads=q.shape[2])
    )
    local_lse_ring = _lse_to_bsh(
        lse_ring, batch=q.shape[0], seq_len=q_local.shape[1], heads=q.shape[2]
    )

    prefix = (
        f"case[combine={combine_mode},block_sparse={block_sparse_mode},"
        f"transport={transport}]"
    )
    _assert_close_metric(
        label=f"{prefix}/out",
        abs_diff=_max_abs(out_ring.float() - local_ref_out.float()),
        ref_abs=_max_abs(local_ref_out),
        atol=output_atol,
        rtol=output_rtol,
    )
    _assert_close_metric(
        label=f"{prefix}/lse",
        abs_diff=_max_abs(local_lse_ring.float() - local_ref_lse.float()),
        ref_abs=_max_abs(local_ref_lse),
        atol=output_atol,
        rtol=output_rtol,
    )

    if not run_backward:
        return

    (out_ring.float() * local_weights).sum().backward()
    assert ref_dq is not None and ref_dk is not None and ref_dv is not None
    local_ref_dq = _local_slice(ref_dq)
    local_ref_dk = _local_slice(ref_dk)
    local_ref_dv = _local_slice(ref_dv)
    assert (
        q_local.grad is not None
        and k_local.grad is not None
        and v_local.grad is not None
    )
    for name, actual, expected in (
        ("dq", q_local.grad, local_ref_dq),
        ("dk", k_local.grad, local_ref_dk),
        ("dv", v_local.grad, local_ref_dv),
    ):
        _assert_close_metric(
            label=f"{prefix}/{name}",
            abs_diff=_max_abs(actual.float() - expected.float()),
            ref_abs=_max_abs(expected),
            atol=grad_atol,
            rtol=grad_rtol,
        )
    _print_rank0(f"{prefix}: ok")


def main() -> None:
    args = _parse_args()
    if not HAS_FLASH4:
        raise RuntimeError("flash_attn.cute is not installed; cannot run this checker.")
    init_distributed()
    try:
        device = _set_local_cuda_device()
        world_size = _world_size()
        rank = _rank()
        if args.seq_len % world_size != 0:
            raise ValueError(
                f"--seq-len must be divisible by world size, got "
                f"{args.seq_len=} and {world_size=}."
            )
        if args.seq_len // world_size < 128:
            raise ValueError(
                "Each ring rank should have at least one 128-token FA4 block; got "
                f"local_len={args.seq_len // world_size}."
            )
        if args.heads % (args.kv_heads or args.heads) != 0:
            raise ValueError("--heads must be divisible by --kv-heads.")
        combine_modes = _default_modes(
            args.combine_modes, ("python", "flash", "online")
        )
        block_sparse_modes = _default_modes(
            args.block_sparse_modes, ("direct", "create_block_mask")
        )
        transports = _default_modes(args.transports, ("p2p",))

        if "online" in combine_modes and args.allow_online_non_sm90_skip:
            major, _minor = torch.cuda.get_device_capability(device)
            if major != 9:
                combine_modes = [m for m in combine_modes if m != "online"]
                _print_rank0("Skipping combine_mode=online because device is not SM90.")

        dtype = _dtype(args.dtype)
        kv_heads = args.kv_heads or args.heads
        q, k, v, weights = _make_qkv(
            batch=args.batch,
            seq_len=args.seq_len,
            heads=args.heads,
            kv_heads=kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
            seed=args.seed,
        )
        document_lens, split_lens, attn_modes = _make_layout(
            args.seq_len,
            layout=args.layout,
            num_docs=args.num_docs,
            splits_per_doc=args.splits_per_doc,
        )
        mask = build_fa4_sparse_mask(
            document_lens,
            split_lens,
            attn_modes,
            device=device,
            retain_block_mask=True,
            build_block_sparse=True,
        )
        softmax_scale = args.head_dim**-0.5
        _print_rank0(
            "Running Flash4 ring component checker with "
            f"world_size={world_size}, seq_len={args.seq_len}, "
            f"local_len={args.seq_len // world_size}, dtype={args.dtype}, "
            f"layout={args.layout}, num_docs={len(document_lens)}, "
            f"num_splits={len(split_lens)}, "
            f"combine_modes={combine_modes}, block_sparse_modes={block_sparse_modes}, "
            f"transports={transports}."
        )

        ref_out, ref_lse, ref_dq, ref_dk, ref_dv = _run_reference(
            q=q,
            k=k,
            v=v,
            weights=weights,
            mask=mask,
            softmax_scale=softmax_scale,
            run_backward=not args.skip_backward,
        )

        token_permutation: Tensor | None = None
        rank_plan: RankRingPlan | None = None
        global_position_ids: Tensor | None = None
        if args.enable_ring_load_balance:
            layout = build_packed_layout(
                document_lens,
                split_lens,
                attn_modes,
                device=device,
                aux_tensors=mask.aux_tensors,
            )
            ring_plan = build_ring_plan(
                layout,
                world_size=world_size,
                device=device,
                block_mask=mask.block_mask,
                use_direct_planner_block_sparse=True,
            )
            token_permutation = ring_plan.token_permutation.to(
                device=device, dtype=torch.long
            )
            global_position_ids = token_permutation.contiguous()
            rank_plan = get_rank_plan(ring_plan, rank=rank)
            q_under_test = _permute_seq(q, token_permutation)
            k_under_test = _permute_seq(k, token_permutation)
            v_under_test = _permute_seq(v, token_permutation)
            weights_under_test = _permute_seq(weights, token_permutation)
            ref_out_under_test = _permute_seq(ref_out, token_permutation)
            ref_lse_under_test = _permute_seq(
                _lse_to_bsh(
                    ref_lse,
                    batch=args.batch,
                    seq_len=args.seq_len,
                    heads=args.heads,
                ),
                token_permutation,
            )
            ref_dq_under_test = (
                _permute_seq(ref_dq, token_permutation) if ref_dq is not None else None
            )
            ref_dk_under_test = (
                _permute_seq(ref_dk, token_permutation) if ref_dk is not None else None
            )
            ref_dv_under_test = (
                _permute_seq(ref_dv, token_permutation) if ref_dv is not None else None
            )
            _print_rank0("Ring load balancing enabled for checker.")
        else:
            q_under_test = q
            k_under_test = k
            v_under_test = v
            weights_under_test = weights
            ref_out_under_test = ref_out
            ref_lse_under_test = ref_lse
            ref_dq_under_test = ref_dq
            ref_dk_under_test = ref_dk
            ref_dv_under_test = ref_dv

        for block_sparse_mode in block_sparse_modes:
            global_aux_tensors = mask.aux_tensors
            if token_permutation is not None:
                global_aux_tensors = [
                    torch.index_select(aux, 0, token_permutation)
                    for aux in mask.aux_tensors
                ]
            step_meta = _build_step_meta(
                global_aux_tensors=global_aux_tensors,
                global_position_ids=global_position_ids,
                seq_len=args.seq_len,
                softmax_scale=softmax_scale,
                block_sparse_mode=block_sparse_mode,
            )
            if block_sparse_mode == "direct":
                _check_block_sparse_components(step_meta)
            for transport in transports:
                for combine_mode in combine_modes:
                    if transport == "symm_mem" and not torch.cuda.is_available():
                        continue
                    _run_ring_case(
                        combine_mode=combine_mode,
                        block_sparse_mode=block_sparse_mode,  # ty: ignore[arg-type]
                        transport=transport,  # ty: ignore[arg-type]
                        q=q_under_test,
                        k=k_under_test,
                        v=v_under_test,
                        weights=weights_under_test,
                        ref_out=ref_out_under_test,
                        ref_lse=ref_lse_under_test,
                        ref_dq=ref_dq_under_test,
                        ref_dk=ref_dk_under_test,
                        ref_dv=ref_dv_under_test,
                        step_meta=step_meta,
                        rank_plan=rank_plan,
                        output_atol=args.output_atol,
                        output_rtol=args.output_rtol,
                        grad_atol=args.grad_atol,
                        grad_rtol=args.grad_rtol,
                        run_backward=not args.skip_backward,
                    )
                dist.barrier()
        if rank == 0:
            print("All Flash4 ring component checks passed.", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
