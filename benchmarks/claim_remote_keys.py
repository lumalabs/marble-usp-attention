# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Share of remote keys the sparse all-to-all (KV gather) does not fetch.

Reproduces the paper's claim (seqdim.tex, Sec. "Sequence dimension"):

    "Because each rank fetches only the blocks it needs, on our tested
    sequences we avoid sending 55--100% of the remote keys that ring
    attention would send."

A ring rotates all ``(R - 1) * seq / R`` remote keys past every rank; the KV
gather (``USPConfig(kv_exchange="gather", balance="chunks")``) fetches only
the remote keys its queries can see. For every real 1-4M-token understanding
pack and ring size this rebuilds each ring rank's fetch exactly as the planner
does (no GPU, no FA4 needed) and reports ``1 - fetched / (R - 1) * local``:

- ``block``: the block-sparse gather (H100 default, head_dim 128): the token
  order of ``usp._build_usp_plan``'s gather base (contiguous chunks from
  ``usp._balanced_chunks``, after ``vision_group_order`` for vision-group
  layouts), the visible keys from ``halo.needed_key_masks`` and the
  128-token KV blocks from ``halo._needed_blocks``, as in ``usp._halo_plan``;
- ``dense``: the dense gather (``dense_ring=True``, used for head_dim 256 on
  H100, e.g. Qwen3.8), where the layout allows it: the dense ring base's
  chunks (``chunk_assignment`` with 1-16 chunks per rank, accepted within
  1.15x balance, else the ``block`` order) and the token-exact keys of
  ``dense_gather._needed``, as in ``dense_gather.build_dense_gather_plan``.

Ring 4-16 is ``USP(R, U)`` with Ulysses 4-8 inside a node on 32-128 GPUs.
Each row's ``fetched`` list matches what ``bench_attention.py`` logs as
``halo_len`` for ring rank 0 (``--check`` compares against measured rows).

    PYTHONPATH=src:benchmarks python benchmarks/claim_remote_keys.py \\
        --out benchmarks/results/claims/remote_keys.jsonl \\
        --check benchmarks/results/rerun_oct26/mlsys
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from provenance import result_json

from usp_attention.layouts import build_layout
from usp_attention.reference import visible_key_counts
from usp_attention.ring.dense_gather import (
    _needed,
    dense_gather_supported,
)
from usp_attention.ring.dense_ring import (
    _segments,
    chunk_assignment,
    dense_ring_supported,
    rank_work,
)
from usp_attention.ring.group_order import vision_group_order
from usp_attention.ring.halo import _needed_blocks, needed_key_masks
from usp_attention.usp import (
    FA4_BLOCK_SIZE,
    USPConfig,
    _balanced_chunks,
    _chunk_ranges,
    _dense_gather_worthwhile,
)

PACKS = Path(__file__).resolve().parent / "layouts" / "packs"
# Real understanding packs at 1-4M tokens (BAGEL generation packs run the
# chunked ring; synth_* are synthetic).
FAMILIES = ("gemma3", "gemma3ps", "molmo2", "qwen38", "qwen3vl")
RINGS = (4, 8, 16)


def _block_order(docs, splits, modes, ring: int) -> torch.Tensor:
    """Stored token order of the block-sparse gather's base plan."""
    group_perm = vision_group_order(docs, splits, modes)
    if group_perm is None:
        return _balanced_chunks(docs, splits, modes, ring, torch.device("cpu"))
    counts = visible_key_counts(docs, splits, modes)
    return group_perm[
        _balanced_chunks(
            docs, splits, modes, ring, torch.device("cpu"), counts=counts[group_perm]
        )
    ]


def block_fetch(docs, splits, modes, ring: int) -> list[int]:
    """Remote keys each ring rank receives in the block-sparse gather."""
    local = sum(docs) // ring
    perm = _block_order(docs, splits, modes, ring)
    masks = needed_key_masks(
        docs,
        splits,
        modes,
        ring_size=ring,
        token_permutation=perm,
        sliding_window=None,
    )
    return [
        sum(
            int(
                _needed_blocks(
                    masks,
                    q_rank=r,
                    kv_rank=o,
                    local_len=local,
                    block_size=FA4_BLOCK_SIZE,
                ).numel()
            )
            * FA4_BLOCK_SIZE
            for o in range(ring)
            if o != r
        )
        for r in range(ring)
    ]


def _dense_chunks(docs, splits, modes, ring: int):
    """Chunks of the dense gather (``None``: falls back to the block gather)."""
    seq = sum(docs)
    if not dense_ring_supported(docs, splits, modes):
        return _chunk_ranges(docs, splits, modes, ring)
    # usp._build_usp_plan's dense ring branch (the gather's base plan).
    best = None
    for per_rank in (1, 2, 4, 8, 16):
        if seq % (ring * per_rank * FA4_BLOCK_SIZE):
            break
        chunks = chunk_assignment(
            docs, splits, modes, ring_size=ring, chunks_per_rank=per_rank
        )
        work = rank_work(docs, splits, modes, ring, chunks)
        imbalance = max(work) / (sum(work) / len(work))
        if best is None or imbalance < best[0] - 1e-9:
            best = (imbalance, chunks, work)
        if imbalance <= 1.025:
            break
    _, chunks, work = best
    return chunks if max(work) <= 1.15 * (sum(work) / len(work)) else None


def dense_fetch(docs, splits, modes, ring: int) -> list[int] | None:
    """Remote keys each ring rank receives in the dense gather, or ``None``
    when the planner would not use it on this layout."""
    if (
        len(splits) < 2
        or not dense_gather_supported(docs, splits, modes)
        or not _dense_gather_worthwhile(splits, modes, USPConfig())
    ):
        return None
    chunks = _dense_chunks(docs, splits, modes, ring)
    if chunks is None:
        return block_fetch(docs, splits, modes, ring)
    seq = sum(docs)
    segs = list(_segments(docs, splits, modes))
    doc_ranges = sorted({(d0, d1) for d0, d1, _, _, _ in segs})
    spans = [(s0, s1) for _, _, s0, s1, m in segs if m == "full"]
    noise = [(s0, s1) for _, _, s0, s1, m in segs if m == "noise"]
    owner = np.empty(seq, dtype=np.int64)
    for rank, ranges in enumerate(chunks):
        for lo, hi in ranges:
            owner[lo:hi] = rank
    return [
        int((_needed(chunks[r], doc_ranges, spans, noise, seq) & (owner != r)).sum())
        for r in range(ring)
    ]


def _packs(min_k: int, max_k: int) -> list[tuple[str, int]]:
    out = []
    for path in sorted(PACKS.glob("*.json.gz")):
        m = re.fullmatch(r"([a-z0-9]+)_[a-z]+_(\d+)k\.json\.gz", path.name)
        if m and m.group(1) in FAMILIES and min_k <= int(m.group(2)) <= max_k:
            out.append((path.name.removesuffix(".json.gz"), int(m.group(2)) * 1024))
    return out


def _measured_halos(result_dir: str) -> dict:
    """Ring rank 0's ``halo_len`` per (layout, ring, path) from measured rows."""
    out = {}
    for f in glob.glob(f"{result_dir}/*.jsonl"):
        for line in open(f):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cfg = row.get("usp_config") or {}
            if (
                row.get("halo_len") is None
                or cfg.get("kv_exchange") != "gather"
                or cfg.get("balance") != "chunks"
            ):
                continue
            path = "dense" if cfg.get("dense_ring") else "block"
            out[(row["layout"].split("#")[0], row["ring"], path)] = row["halo_len"]
    return out


def main() -> None:
    """Compute the avoided remote-key share for every pack and ring size."""
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", required=True)
    p.add_argument("--rings", type=int, nargs="+", default=list(RINGS))
    p.add_argument("--min-k", type=int, default=1024)
    p.add_argument("--max-k", type=int, default=4096)
    p.add_argument("--check", default=None, help="dir of measured rows (halo_len)")
    args = p.parse_args()
    measured = _measured_halos(args.check) if args.check else {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, list[float]] = {"block": [], "dense": []}
    with open(args.out, "w") as out:
        for name, seq in _packs(args.min_k, args.max_k):
            docs, splits, modes = build_layout(f"file:{PACKS / name}.json#0", seq)
            for ring in args.rings:
                local = seq // ring
                remote = (ring - 1) * local
                for path, fn in (("block", block_fetch), ("dense", dense_fetch)):
                    fetched = fn(docs, splits, modes, ring)
                    if fetched is None:
                        continue
                    avoided = [1 - f / remote for f in fetched]
                    summary[path] += avoided
                    row = {
                        "bench": "claim_remote_keys",
                        "layout": f"{name}#0",
                        "seq_len": seq,
                        "ring": ring,
                        "path": path,
                        "remote_keys_per_rank": remote,
                        "fetched": fetched,
                        "avoided_min": min(avoided),
                        "avoided_median": float(np.median(avoided)),
                        "avoided_max": max(avoided),
                        "measured_rank0_halo_len": measured.get((name, ring, path)),
                    }
                    out.write(result_json(row) + "\n")
                    out.flush()
                    check = row["measured_rank0_halo_len"]
                    print(
                        f"{name:28s} R{ring:<3d}{path:6s} avoided "
                        f"min {min(avoided):6.1%} med {np.median(avoided):6.1%} "
                        f"max {max(avoided):6.1%}  rank0 {fetched[0]}"
                        + (f" (measured {check})" if check is not None else ""),
                        flush=True,
                    )
    for path, vals in summary.items():
        if vals:
            print(
                f"{path}: avoided share over all ranks {min(vals):.1%}-{max(vals):.1%}"
            )


if __name__ == "__main__":
    main()
