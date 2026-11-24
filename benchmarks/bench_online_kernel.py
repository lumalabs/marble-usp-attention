# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Single-GPU microbenchmark of the online FA4 kernel against chunked alternatives.

For one query block and a KV sequence split into ``W`` equal chunks (what a
rank sees over ``W`` ring steps, without communication), compares:

- ``fa4_single``: one FA4 forward over the whole KV (the lower bound),
- ``online``: ``W`` online FA4 calls threading fp32 softmax state,
- ``fa4_flash_combine``: ``W`` FA4 forwards, merging each partial with
  ``flash_attn_combine`` after every chunk (the ring "flash" path),
- ``fa4_python_combine``: the same with a PyTorch log-sum-exp merge.

Reports the median time of each method and its mean relative output error
against an fp32 reference computed from the same inputs::

    python benchmarks/bench_online_kernel.py --seq-q 16384 --chunks 1 2 4 8 16

Two masks mirror ring steps:

- ``--causal``: the query block is the last KV chunk, as on a causal ring's
  last rank. Earlier chunks are unmasked and the last one is causal, so one
  causal FA4 call over the whole KV gives the same mask.
- ``--block-sparse DENSITY``: whole-block sparsity with ``tile_n``-wide KV
  blocks. In each chunk a query block row attends a random contiguous run of
  KV blocks; after the first chunk a row is non-empty only with probability
  DENSITY, as on document-packed ring steps where most rows skip most steps.
"""

from __future__ import annotations

import argparse
import statistics
import sys

import torch
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_fwd
from provenance import result_json

from usp_attention.online_flash.interface import (
    _tile_size_online_sm90,
    online_flash_attn_func,
)
from usp_attention.ring.ring_combine import (
    _LocalRingResultMerger,
    _LocalRingResultMergerFlash,
    _normalize_step_lse,
)


def _time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def _reference(q, k, v, keep=None) -> torch.Tensor:
    """fp32 attention; ``keep(start, stop)`` gives the (rows, seqlen_k) mask."""
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    scale = q.shape[-1] ** -0.5
    kf, vf = k.float(), v.float()
    rows = max(16, (1 << 29) // (k.shape[1] * q.shape[2]))  # ~2 GiB of fp32 scores
    for start in range(0, q.shape[1], rows):
        qc = q[:, start : start + rows].float()
        s = torch.einsum("bqhd,bkhd->bhqk", qc, kf) * scale
        if keep is not None:
            s = s.masked_fill(~keep(start, start + qc.shape[1]), float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, start : start + rows] = torch.einsum("bhqk,bkhd->bqhd", p, vf)
    return out


def _block_sparse_chunks(
    seq_q: int,
    chunk: int,
    chunks: int,
    heads: int,
    tile: tuple[int, int],
    density: float,
) -> tuple[list[BlockSparseTensorsTorch], BlockSparseTensorsTorch, torch.Tensor]:
    """Full-block sparsity per chunk, for the whole KV, and as a block mask.

    Every query block row attends a random contiguous run of KV blocks in the
    first chunk; in each later chunk it does so with probability ``density``
    and is otherwise empty there. Returns the per-chunk tensors, the same
    pattern over the concatenated KV, and the (q blocks, KV blocks) bool mask.
    """
    gen = torch.Generator().manual_seed(0)
    n_m, n_n = -(-seq_q // tile[0]), chunk // tile[1]
    mask = torch.zeros(n_m, chunks * n_n, dtype=torch.bool)
    for c in range(chunks):
        for r in range(n_m):
            if c > 0 and torch.rand((), generator=gen) >= density:
                continue
            length = int(torch.randint(1, n_n + 1, (), generator=gen))
            first = int(torch.randint(0, n_n - length + 1, (), generator=gen))
            mask[r, c * n_n + first : c * n_n + first + length] = True

    def tensors(m: torch.Tensor) -> BlockSparseTensorsTorch:
        cnt = m.sum(-1, dtype=torch.int32)
        # Block indices ascending, packed left; FA4 walks them from the end.
        idx = torch.sort(torch.where(m, torch.arange(m.shape[1]), m.shape[1]), -1)
        idx = torch.where(idx.values < m.shape[1], idx.values, 0).to(torch.int32)

        def expand(t: torch.Tensor) -> torch.Tensor:
            return t.expand(1, heads, *t.shape).contiguous().cuda()

        return BlockSparseTensorsTorch(
            mask_block_cnt=expand(torch.zeros_like(cnt)),
            mask_block_idx=expand(torch.zeros_like(idx)),
            full_block_cnt=expand(cnt),
            full_block_idx=expand(idx),
            block_size=tile,
        )

    per_chunk = [tensors(mask[:, c * n_n : (c + 1) * n_n]) for c in range(chunks)]
    return per_chunk, tensors(mask), mask


def main() -> None:
    """Compare online and merged attention kernels using CLI options.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sys.excepthook = sys.__excepthook__  # CuTe DSL replaces it on import
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--seq-q", type=int, default=16384)
    p.add_argument(
        "--chunk-k", type=int, default=None, help="KV tokens per chunk (default: seq-q)"
    )
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-accuracy", action="store_true")
    mask_args = p.add_mutually_exclusive_group()
    mask_args.add_argument(
        "--causal",
        action="store_true",
        help="causal last chunk (needs chunk-k = seq-q)",
    )
    mask_args.add_argument(
        "--block-sparse",
        type=float,
        default=None,
        metavar="DENSITY",
        help="whole-block sparsity; later chunks keep a row with this probability",
    )
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    chunk = args.chunk_k or args.seq_q
    if args.causal and chunk != args.seq_q:
        p.error("--causal needs --chunk-k equal to --seq-q")
    tile, mask_name = None, "causal" if args.causal else "none"
    if args.block_sparse is not None:
        mask_name = f"block_sparse_{args.block_sparse}"
        # KV blocks are tile_n wide (FA4 requires it); chunks hold whole blocks.
        cfg = _tile_size_online_sm90(args.head_dim, args.head_dim, False)
        tile = (128, cfg.n_block_size)
        chunk -= chunk % tile[1]
    q = torch.randn(
        1, args.seq_q, args.heads, args.head_dim, device=device, dtype=torch.bfloat16
    )
    for w in args.chunks:
        k = torch.randn(
            1, chunk * w, args.heads, args.head_dim, device=device, dtype=torch.bfloat16
        )
        v = torch.randn_like(k)
        k_chunks, v_chunks = list(k.chunk(w, dim=1)), list(v.chunk(w, dim=1))
        chunk_kw = [{} for _ in range(w)]
        single_kw, keep = {}, None
        if args.causal:
            chunk_kw[-1] = single_kw = {"causal": True}

            def keep(a: int, b: int, w: int = w) -> torch.Tensor:
                rows = torch.arange(a, b, device=device)[:, None] + chunk * (w - 1)
                return torch.arange(chunk * w, device=device)[None] <= rows

        elif tile is not None:
            per_chunk, whole, block_mask = _block_sparse_chunks(
                args.seq_q, chunk, w, args.heads, tile, args.block_sparse
            )
            chunk_kw = [{"block_sparse_tensors": t} for t in per_chunk]
            single_kw = {"block_sparse_tensors": whole}
            block_mask = block_mask.to(device)

            def keep(a: int, b: int, block_mask=block_mask) -> torch.Tensor:
                rows = block_mask[torch.arange(a, b, device=device) // tile[0]]
                return rows.repeat_interleave(tile[1], dim=1)

        def fa4_single():
            return _flash_attn_fwd(q, k, v, return_lse=False, **single_kw)[0]

        def online():
            out = max_ = sum_ = None
            for i, (kc, vc, kw) in enumerate(zip(k_chunks, v_chunks, chunk_kw)):
                out, _, max_, sum_ = online_flash_attn_func(
                    q,
                    kc,
                    vc,
                    out=out,
                    max_=max_,
                    sum_=sum_,
                    is_online_first=i == 0,
                    is_online_last=i == w - 1,
                    **kw,
                )
            return out

        def chunked(merger_cls):
            def run():
                merger = merger_cls()
                for kc, vc, kw in zip(k_chunks, v_chunks, chunk_kw):
                    out, lse = _flash_attn_fwd(q, kc, vc, return_lse=True, **kw)[:2]
                    merger.step(out, _normalize_step_lse(out, lse))
                return merger.result()[0]

            return run

        methods = {
            "fa4_single": fa4_single,
            "online": online,
            "fa4_flash_combine": chunked(_LocalRingResultMergerFlash),
            "fa4_python_combine": chunked(_LocalRingResultMerger),
        }
        ref = None if args.no_accuracy else _reference(q, k, v, keep)
        row = {
            "bench": "online_kernel",
            "seq_q": args.seq_q,
            "chunk_k": chunk,
            "chunks": w,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "mask": mask_name,
        }
        for name, fn in methods.items():
            row[f"{name}_ms"] = _time(fn, args.iters, args.warmup)
            if ref is not None:
                out = fn().float()
                row[f"{name}_err"] = (
                    (out - ref).abs().mean() / ref.abs().mean()
                ).item()
        base = row["fa4_single_ms"]
        for name in methods:
            row[f"{name}_vs_single"] = row[f"{name}_ms"] / base
        print("RESULT " + result_json(row), flush=True)


if __name__ == "__main__":
    main()
