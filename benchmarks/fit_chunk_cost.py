# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Fit the chunked ring's per-range cost (tau) to measured BAGEL runs.

Reproduces the paper's claim (ring.tex, Sec. "Chunked ring attention"):

    "We fit $\\tau$ on nine BAGEL runs at 1--8M tokens, and $C$ predicts the
    runtime eight out of nine times within 3%."

with the layout cost (Eq. stepcost, :func:`~usp_attention.ring.generation_ring.layout_cost`)

    C = sum_s max_r W[r, (r - s) mod R] + tau * R * k,

``W`` the visible pairs between rank ``r``'s queries and rank ``j``'s keys and
``k`` the most contiguous ranges any rank holds. For every measured chunked
ring run (``generation_ring`` true, BAGEL long video) this rebuilds the run's
exact plan with :func:`~usp_attention.ring.generation_ring.ring_chunk_assignment`
(same ring size, forced chunk count and refine flag), counts ``W`` exactly
(every query, not the planner's 64 samples per chunk), checks the plan against
the row's logged ``ring_imbalance``, and fits

    runtime_ms = a * (step_pairs + tau * R * k)

by least squares on relative error (all runs share heads per GPU and head
dim, so ``a`` is one time per pair). It reports each run's relative error,
how many are within 3%, the same with the planner's ``tau`` = 6.4e6 pairs
(:data:`usp_attention.strategy._RANGE_PAIRS`) and leave-one-out errors.

The nine runs are the paper's (``reproduction_source`` bagel32k:1-4,
bagel32g:5, bagel64d:1, bagel64e:1-3), re-measured on 2026-10-26; repeats of
a source are averaged. ``--all-rows`` fits every row instead.

    PYTHONPATH=src:benchmarks python benchmarks/fit_chunk_cost.py \\
        benchmarks/results/rerun_oct26/mlsys/*.jsonl \\
        benchmarks/results/rerun_oct26/extra/QL.jsonl \\
        --out benchmarks/results/claims/chunk_cost_fit.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from provenance import result_json

from usp_attention.layouts import load_layout_file
from usp_attention.reference import visible_key_counts
from usp_attention.ring.generation_ring import (
    _range_counts,
    chunk_pair_counts,
    layout_cost,
    ring_chunk_assignment,
)
from usp_attention.strategy import _RANGE_PAIRS

PACKS = Path(__file__).resolve().parent / "layouts" / "packs"
PAPER_SOURCES = (
    "bagel32k:1",
    "bagel32k:2",
    "bagel32k:3",
    "bagel32k:4",
    "bagel32g:5",
    "bagel64d:1",
    "bagel64e:1",
    "bagel64e:2",
    "bagel64e:3",
)


@dataclass(kw_only=True, slots=True)
class Run:
    """One measured chunked ring configuration and its plan's cost terms.

    Attributes:
        key: Source label (``reproduction_source`` or ``file:line``).
        layout: Pack name (``bagel_video_4096k``).
        ring: Ring size.
        chunks: Forced chunks per rank (``None``: the planner's choice).
        refine: Whether chunk swaps refined the deal.
        ms: Measured forward + backward median (mean over repeats).
        repeats: Rows averaged into ``ms``.
        step_pairs: ``sum_s max_r W`` of the plan (exact counts).
        ranges: Most contiguous ranges any rank holds (``k``).
        imbalance: Plan's max / mean rank pairs (the row's ``ring_imbalance``).
        logged_imbalance: ``ring_imbalance`` logged by the run.
    """

    key: str
    layout: str
    ring: int
    chunks: int | None
    refine: bool
    ms: float
    repeats: int
    step_pairs: float = 0.0
    ranges: int = 0
    imbalance: float = 0.0
    logged_imbalance: float = 0.0


def _rows(paths: list[str]) -> list[tuple[str, dict]]:
    out = []
    for path in paths:
        for i, line in enumerate(open(path), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("generation_ring") and "bagel_video" in row.get("layout", ""):
                out.append((f"{Path(path).stem}:{i}", row))
    return out


def collect(paths: list[str], *, all_rows: bool) -> list[Run]:
    """Measured runs, one per paper source (repeats averaged) or per row."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for where, row in _rows(paths):
        key = where if all_rows else row.get("reproduction_source", where)
        if all_rows or key in PAPER_SOURCES:
            groups[key].append(row)
    runs = []
    for key, rows in groups.items():
        cfg = rows[0]["usp_config"]
        assert all(
            r["layout"] == rows[0]["layout"] and r["ring"] == rows[0]["ring"]
            for r in rows
        )
        runs.append(
            Run(
                key=key,
                layout=rows[0]["layout"].split("#")[0],
                ring=rows[0]["ring"],
                chunks=cfg.get("dense_ring_chunks"),
                refine=bool(cfg.get("dense_ring_refine", True)),
                ms=float(np.mean([r["median_ms"] for r in rows])),
                repeats=len(rows),
                logged_imbalance=float(rows[0]["ring_imbalance"]),
            )
        )
    return sorted(runs, key=lambda r: (r.layout, r.ring, r.chunks or 0, r.refine))


def plan_terms(run: Run) -> None:
    """Rebuild the run's plan and fill its cost terms in place."""
    seq = int(run.layout.split("_")[-1].removesuffix("k")) * 1024
    docs, splits, modes = load_layout_file(f"{PACKS / run.layout}.json#0", seq)
    bounds = (
        {"min_chunks_per_rank": run.chunks, "max_chunks_per_rank": run.chunks}
        if run.chunks
        else {}
    )
    chunks = ring_chunk_assignment(
        docs, splits, modes, ring_size=run.ring, refine=run.refine, **bounds
    )
    # Owners at the finest chunk size the planner considers (64 per rank).
    n = run.ring * 64
    while seq % (n * 128):
        n //= 2
    size = seq // n
    owner = np.empty(n, dtype=np.int64)
    for rank, ranges in enumerate(chunks):
        for lo, hi in ranges:
            assert lo % size == 0 and hi % size == 0
            owner[lo // size : hi // size] = rank
    pairs = chunk_pair_counts(docs, splits, modes, n_chunks=n, samples_per_chunk=size)
    run.step_pairs = float(layout_cost(pairs, owner, run.ring))
    run.ranges = int(_range_counts(owner, run.ring).max())
    counts = visible_key_counts(docs, splits, modes).numpy()
    work = [sum(int(counts[lo:hi].sum()) for lo, hi in rank) for rank in chunks]
    run.imbalance = max(work) / (sum(work) / len(work))


def fit(runs: list[Run], tau: float | None = None) -> tuple[float, float]:
    """``(a, tau)`` minimizing relative squared error of ``a * (S + tau * R * k)``;
    with ``tau`` given, only ``a``."""
    t = np.array([r.ms for r in runs])
    s = np.array([r.step_pairs for r in runs])
    rk = np.array([r.ring * r.ranges for r in runs], dtype=np.float64)
    if tau is not None:
        x = s + tau * rk
        a = float(np.sum(x / t) / np.sum((x / t) ** 2))
        return a, tau
    design = np.stack([s / t, rk / t], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, np.ones_like(t), rcond=None)
    return float(a), float(b / a)


def predict(run: Run, a: float, tau: float) -> float:
    return a * (run.step_pairs + tau * run.ring * run.ranges)


def main() -> None:
    """Fit tau and report the cost model's per-run error."""
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--all-rows", action="store_true")
    p.add_argument("--within", type=float, default=0.03)
    args = p.parse_args()
    runs = collect(args.results, all_rows=args.all_rows)
    for run in runs:
        plan_terms(run)
    a, tau = fit(runs)
    a_fixed, _ = fit(runs, tau=_RANGE_PAIRS)
    loo = []
    for i, run in enumerate(runs):
        a_i, tau_i = fit(runs[:i] + runs[i + 1 :])
        loo.append((predict(run, a_i, tau_i), tau_i))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    errs = {"fit": [], "fixed": [], "loo": []}
    with open(args.out, "w") as out:
        print(
            f"{'run':12s} {'layout':18s} R  chunks refine   k  imb(plan/log)  "
            f"ms      fit     tau=6.4e6  leave-one-out"
        )
        for run, (loo_ms, loo_tau) in zip(runs, loo, strict=True):
            e = {
                "fit": predict(run, a, tau) / run.ms - 1,
                "fixed": predict(run, a_fixed, _RANGE_PAIRS) / run.ms - 1,
                "loo": loo_ms / run.ms - 1,
            }
            for name, v in e.items():
                errs[name].append(v)
            row = {
                "bench": "fit_chunk_cost",
                "run": run.key,
                "layout": f"{run.layout}#0",
                "ring": run.ring,
                "dense_ring_chunks": run.chunks,
                "dense_ring_refine": run.refine,
                "measured_ms": run.ms,
                "repeats": run.repeats,
                "step_pairs": run.step_pairs,
                "ranges_per_rank": run.ranges,
                "plan_imbalance": run.imbalance,
                "logged_imbalance": run.logged_imbalance,
                "rel_err_fit": e["fit"],
                "rel_err_planner_tau": e["fixed"],
                "rel_err_leave_one_out": e["loo"],
                "leave_one_out_tau": loo_tau,
            }
            out.write(result_json(row) + "\n")
            print(
                f"{run.key:12s} {run.layout:18s} {run.ring:<3d}{run.chunks or '-':>5} "
                f"{run.refine!s:6s} {run.ranges:4d}  {run.imbalance:.4f}/"
                f"{run.logged_imbalance:.4f} {run.ms:7.1f} {e['fit']:+7.2%} "
                f"{e['fixed']:+8.2%}  {e['loo']:+8.2%}"
            )
        summary = {
            "bench": "fit_chunk_cost_summary",
            "runs": len(runs),
            "a_ms_per_gpair": a * 1e9,
            "tau_pairs": tau,
            "planner_tau_pairs": _RANGE_PAIRS,
            "a_ms_per_gpair_planner_tau": a_fixed * 1e9,
            **{
                f"within_{args.within:g}_{name}": int(
                    sum(abs(v) <= args.within for v in vals)
                )
                for name, vals in errs.items()
            },
            **{
                f"max_abs_err_{name}": max(map(abs, vals))
                for name, vals in errs.items()
            },
        }
        out.write(result_json(summary) + "\n")
    print(
        f"fit: tau = {tau:.3g} pairs, a = {a * 1e9:.4g} ms per 1e9 pairs; "
        f"planner tau {_RANGE_PAIRS:.3g}"
    )
    for name, vals in errs.items():
        n_ok = sum(abs(v) <= args.within for v in vals)
        print(
            f"{name:6s}: {n_ok}/{len(vals)} within {args.within:.0%}, "
            f"max |err| {max(map(abs, vals)):.2%}"
        )


if __name__ == "__main__":
    main()
