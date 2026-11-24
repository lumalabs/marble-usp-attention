# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Planning cost per batch layout, without a cluster.

A plan depends only on the packed layout and the rank's coordinates, so this
builds every ring rank's plan in one process (no process groups) and reports
the slowest rank: in training each rank builds its own plan in parallel, once
per batch, and every layer of the step reuses it. ``--device cpu`` measures
planning in a data-loader worker (CPU only), ``--device cuda`` on the GPU.

    python benchmarks/bench_planning.py --layout file:<json>#0 --seq-len 4194304 \\
        --ring 8 --ulysses 4 --mode dense_ring --device cpu
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from provenance import result_json

from usp_attention import USPConfig, build_usp_plan
from usp_attention.distributed import USPGroups
from usp_attention.layouts import build_layout

MODES = {
    "ring_ptrr": dict(),
    "ring_chunks": dict(balance="chunks"),
    "gather": dict(balance="chunks", kv_exchange="gather", dense_ring=False),
    "dense_gather": dict(balance="chunks", kv_exchange="gather", dense_ring=True),
    "dense_ring": dict(balance="chunks", dense_ring=True),
}


def _groups(ring: int, ulysses: int, ring_rank: int) -> USPGroups:
    """Coordinates of one rank; planning never touches the process groups."""
    return USPGroups(
        ring_size=ring,
        ulysses_size=ulysses,
        ring_rank=ring_rank,
        ulysses_rank=0,
        cp_rank=ring_rank * ulysses,
        ring_group=None,
        ulysses_group=None,
        cp_group=None,
    )


def main() -> None:
    """Measure planning cost from sys.argv and print stamped result rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--layout", required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--ring", type=int, required=True)
    p.add_argument("--ulysses", type=int, default=1)
    p.add_argument("--mode", choices=sorted(MODES), action="append")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    p.add_argument(
        "--ranks", type=int, default=None, help="time only the first N ring ranks"
    )
    p.add_argument("--label", default="")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    docs, splits, modes = build_layout(args.layout, args.seq_len, ring_size=args.ring)
    for mode in args.mode or sorted(MODES):
        config = USPConfig(**MODES[mode])
        times, kind = [], None
        for r in range(args.ranks or args.ring):
            t = time.perf_counter()
            plan = build_usp_plan(
                document_lens=docs,
                split_lens=splits,
                attn_modes=modes,
                groups=_groups(args.ring, args.ulysses, r),
                config=config,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            kind = next(
                (k for k in ("generation_ring", "dense_gather", "halo", "dense_ring", "step_meta")
                 if getattr(plan, k, None) is not None),
                "other",
            )  # fmt: skip
        print(
            "RESULT "
            + result_json(
                {
                    "bench": "planning",
                    "label": args.label,
                    "layout": os.path.basename(args.layout),
                    "seq_len": args.seq_len,
                    "ring": args.ring,
                    "ulysses": args.ulysses,
                    "mode": mode,
                    "plan": kind,
                    "device": args.device,
                    "threads": torch.get_num_threads(),
                    "max_s": max(times),
                    "median_s": sorted(times)[len(times) // 2],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
