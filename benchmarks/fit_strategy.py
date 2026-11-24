# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Fit :class:`usp_attention.strategy.Hardware` to measured runs and score the selector.

Reads ``bench_attention.py`` result rows (JSONL), recomputes each row's
strategy features from its layout file, fits the hardware constants by least
squares on log step time, and reports, per measured case (model, layout,
GPUs), how the selector's pick among the measured strategies compares with the
fastest measured one (regret = picked / best). ``--holdout model`` refits
without each model and scores it, so the selector is judged on models it has
not seen.

    python benchmarks/fit_strategy.py --layouts <dir> results/*.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from collections import defaultdict

import numpy as np

from usp_attention.layouts import load_layout_file
from usp_attention.strategy import Hardware, Strategy, estimate, features, mask_stats

PARAMS = (
    "sparse_tflops",
    "dense_tflops",
    "fa3_tflops",
    "intra_gbps",
    "inter_gbps",
    "ring_step_ms",
    "ring_step_intra_ms",
    "gather_ms",
    "noise_tflops",
)
# Physical ranges for the fit (log-space clamps): throughputs below H100 peak,
# fixed costs positive and small.
BOUNDS = {
    "sparse_tflops": (100.0, 900.0),
    "dense_tflops": (100.0, 900.0),
    "fa3_tflops": (100.0, 900.0),
    "intra_gbps": (20.0, 400.0),
    "inter_gbps": (5.0, 100.0),
    "ring_step_ms": (0.05, 30.0),
    "ring_step_intra_ms": (0.01, 10.0),
    "gather_ms": (0.05, 10.0),
    "noise_tflops": (100.0, 900.0),
}
# Starting point of every fit (in-sample and held-out alike): round,
# physically plausible H100 numbers, so held-out fits never start from
# constants fitted on the cases they are scored on.
PRIOR = Hardware(
    sparse_tflops=400.0,
    dense_tflops=400.0,
    fa3_tflops=470.0,
    intra_gbps=150.0,
    inter_gbps=40.0,
    ring_step_ms=10.0,
    ring_step_intra_ms=1.0,
    gather_ms=1.0,
    noise_tflops=400.0,
)
_LO = np.log([BOUNDS[p][0] for p in PARAMS])
_HI = np.log([BOUNDS[p][1] for p in PARAMS])


def _chunked_ring(row: dict) -> bool:
    """Rows that ran the exact chunked ring (generation_ring): dense ring runs
    on layouts with noise spans inside documents, or with exact_dense_ring."""
    if (
        (row.get("kv_exchange") or "ring") != "ring"
        or row["ring"] == 1
        or row.get("halo_len") is not None
    ):
        return False
    dense = "dring" in (row.get("label") or "") or bool(row.get("dense_ring"))
    return dense and (
        bool(row.get("exact_dense_ring")) or "bagel_video" in row.get("layout", "")
    )


def _case_label(row: dict) -> str:
    """The workload a row measures: ablation, FA3 and chunked-ring runs of one
    workload carry suffixed labels but are alternatives within one case."""
    label = (row.get("label") or "").removeprefix("abl_").removesuffix("fa3")
    return label.split("_dring")[0]


def _strategy(row: dict) -> Strategy:
    exchange = row.get("kv_exchange") or "ring"
    if row.get("halo_len") is not None:
        # Sliding-window plans gather their halo whatever kv_exchange says.
        exchange = "gather"
    return Strategy(
        ring_size=row["ring"],
        ulysses_size=row["ulysses"],
        replicate_kv=row.get("replicate_kv") or 1,
        kv_exchange=exchange,
        balance="chunked" if _chunked_ring(row) else (row.get("balance") or "ptrr"),
    )


def _kernel(row: dict, all_causal: bool, fa3: bool) -> str:
    if _chunked_ring(row):
        return "dense"
    dense = bool(row.get("dense_ring")) or row.get("interval_masks") == "on"
    if row["ring"] == 1:
        if all_causal:
            return "fa3" if fa3 else "dense"
        return "dense" if row.get("interval_masks") == "on" else "sparse"
    if row.get("kv_exchange") == "gather" and dense:
        return "fa3" if (fa3 and all_causal) else "dense"
    return "dense" if dense else "sparse"


def load_rows(paths: list[str], layouts: str, fa3_files: set[str]) -> list[dict]:
    """Rows with their features (one layout analysis per (layout, window))."""
    stats_cache: dict = {}
    feat_cache: dict = {}
    out = []
    for path in paths:
        name = os.path.basename(path).removesuffix(".jsonl")
        for line in open(path):
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if (
                "median_ms" not in row
                or row.get("dp_replicas", 1) != 1
                or row.get("impl", "usp").startswith(("flashmask", "magi", "flex"))
                or "heads" not in row
            ):
                continue
            layout = row["layout"].removeprefix("file:").split("/")[-1]
            file, _, index = layout.partition("#")
            if not file.endswith(".json"):
                file += ".json"
            spec = f"{layouts}/{file}#{index or 0}"
            window = row.get("sliding_window") or None
            key = (spec, row["seq_len"], window)
            if key not in stats_cache:
                try:
                    docs, splits, modes = load_layout_file(spec, row["seq_len"])
                except (FileNotFoundError, ValueError):
                    stats_cache[key] = None
                else:
                    stats_cache[key] = mask_stats(
                        docs, splits, modes, sliding_window=window
                    )
            stats = stats_cache[key]
            if stats is None:
                continue
            strategy = _strategy(row)
            fkey = (key, strategy, row["heads"], row.get("kv_heads"), row["head_dim"])
            if fkey not in feat_cache:
                feat_cache[fkey] = features(
                    strategy,
                    stats,
                    heads=row["heads"],
                    kv_heads=row.get("kv_heads") or row["heads"],
                    head_dim=row["head_dim"],
                )
            feats = dict(feat_cache[fkey])
            if row.get("halo_len") is not None and strategy.kv_exchange == "gather":
                # The measured halo of this exact plan.
                kv_local = (row.get("kv_heads") or row["heads"]) * strategy.replicate_kv
                kv_local /= strategy.ulysses_size
                feats["gather_bytes"] = (
                    row["halo_len"] * kv_local * row["head_dim"] * 12
                )
            all_causal = all(m == "causal" for m in stats.attn_modes)
            fa3 = name in fa3_files and "--no-fa3" not in json.dumps(row)
            if row.get("fa3_on_hopper") is False:
                fa3 = False
            out.append(
                {
                    "case": (
                        _case_label(row),
                        file,
                        row["world"],
                        window or 0,
                    ),
                    "model": _case_label(row),
                    "strategy": strategy,
                    "kernel": _kernel(row, all_causal, fa3),
                    "feats": feats,
                    "ms": row["median_ms"],
                    "source": name,
                }
            )
    return out


def _hw(x: np.ndarray, base: Hardware) -> Hardware:
    return dataclasses.replace(base, **dict(zip(PARAMS, np.exp(x), strict=True)))


def _loss(x, rows, base):
    hw = _hw(x, base)
    err = [
        math.log(estimate(r["strategy"], r["feats"], hw, kernel=r["kernel"]).total_ms)
        - math.log(r["ms"])
        for r in rows
    ]
    # Block-sparse kernels cannot beat dense ones per visible pair.
    order = max(0.0, math.log(hw.sparse_tflops / hw.dense_tflops))
    return float(np.mean(np.square(err))) + 10.0 * order**2


def fit(rows: list[dict], base: Hardware | None = None, iters: int = 60) -> Hardware:
    """Coordinate descent on log-parameters (robust, no SciPy needed)."""
    base = base or PRIOR
    x = np.clip(np.log([getattr(base, p) for p in PARAMS]), _LO, _HI)
    best = _loss(x, rows, base)
    step = 1.0
    for _ in range(iters):
        improved = False
        for i in range(len(x)):
            for d in (step, -step):
                y = x.copy()
                y[i] = np.clip(y[i] + d, _LO[i], _HI[i])
                val = _loss(y, rows, base)
                if val < best:
                    x, best, improved = y, val, True
        if not improved:
            step /= 2
            if step < 1e-3:
                break
    return _hw(x, base)


def selectable(strategy: Strategy) -> bool:
    """Strategies the selector ranks (:func:`usp_attention.strategy.candidates`):
    Ulysses only, the PTRR ring, the contiguous-chunk gather and the chunked ring."""
    if strategy.ring_size == 1 or strategy.balance == "chunked":
        return True
    if strategy.kv_exchange == "gather":
        return strategy.balance == "chunks"
    return strategy.balance == "ptrr"


def score(rows: list[dict], hw: Hardware) -> list[dict]:
    """Per case: measured best (any strategy) vs the selector's pick among the
    measured strategies it ranks."""
    cases = defaultdict(list)
    for r in rows:
        cases[r["case"]].append(r)
    out = []
    for case, rs in sorted(cases.items()):
        # One row per strategy: repeats and variants the model does not
        # distinguish (e.g. halo overlap on/off) count at their fastest.
        fastest = {}
        for r in rs:
            if r["strategy"] not in fastest or r["ms"] < fastest[r["strategy"]]["ms"]:
                fastest[r["strategy"]] = r
        rs = list(fastest.values())
        pickable = [r for r in rs if selectable(r["strategy"])]
        if len(rs) < 2 or not pickable:
            continue
        best = min(rs, key=lambda r: r["ms"])
        ranked = sorted(
            pickable,
            key=lambda r: (
                estimate(r["strategy"], r["feats"], hw, kernel=r["kernel"]).total_ms
            ),
        )
        pick = ranked[0]
        out.append(
            {
                "case": case,
                "n": len(rs),
                "best": best["strategy"].label,
                "best_ms": best["ms"],
                "pick": pick["strategy"].label,
                "pick_ms": pick["ms"],
                "regret": pick["ms"] / best["ms"],
                # Timing the top k on real steps keeps the fastest of them.
                "topk_regret": [
                    min(r["ms"] for r in ranked[:k]) / best["ms"] for k in (1, 2, 3)
                ],
            }
        )
    return out


def _report(scores: list[dict], title: str) -> None:
    if not scores:
        return
    reg = np.array([s["regret"] for s in scores])
    print(
        f"\n{title}: {len(scores)} cases, picked the best in {np.mean(reg <= 1.0001):.0%}, "
        f"within 5% in {np.mean(reg <= 1.05):.0%}, within 10% in {np.mean(reg <= 1.10):.0%}; "
        f"mean regret {reg.mean():.3f}, worst {reg.max():.2f}"
    )
    for k in (2, 3):
        top = np.array([s["topk_regret"][k - 1] for s in scores])
        print(
            f"  top-{k} shortlist, fastest measured: best in {np.mean(top <= 1.0001):.0%}, "
            f"within 5% in {np.mean(top <= 1.05):.0%}, worst {top.max():.2f}"
        )
    for s in sorted(scores, key=lambda s: -s["regret"])[:8]:
        print(
            f"  {s['regret']:.2f}x  {s['case']}  picked {s['pick']} ({s['pick_ms']:.0f} ms), "
            f"best {s['best']} ({s['best_ms']:.0f} ms)"
        )


def _group(row: dict, key: str):
    """Hold-out group of a row: its model family or its GPU count."""
    return row["model"].split("_")[0] if key == "model" else row["case"][2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("results", nargs="+")
    p.add_argument("--layouts", required=True)
    p.add_argument(
        "--fa3-files", default="", help="comma-separated result names run with FA3"
    )
    p.add_argument(
        "--holdout", choices=["none", "model", "world", "both"], default="both"
    )
    args = p.parse_args()
    rows = load_rows(
        args.results, args.layouts, set(filter(None, args.fa3_files.split(",")))
    )
    print(f"{len(rows)} rows, {len({r['case'] for r in rows})} cases")
    hw = fit(rows)
    print("fitted:", {k: round(getattr(hw, k), 3) for k in PARAMS})
    print(
        f"rms log error {math.sqrt(_loss(np.log([getattr(hw, k) for k in PARAMS]), rows, hw)):.3f}"
    )
    _report(score(rows, hw), "in-sample")
    for key in ("model", "world"):
        if args.holdout not in (key, "both"):
            continue
        held = []
        for value in sorted({_group(r, key) for r in rows}):
            train = [r for r in rows if _group(r, key) != value]
            test = [r for r in rows if _group(r, key) == value]
            if not train or not test:
                continue
            held += score(test, fit(train))
        _report(held, f"held-out {key}")


if __name__ == "__main__":
    main()
