# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Summarize benchmark and accuracy results into Markdown tables.

Usage::

    python benchmarks/report.py results/ablation.jsonl results/gqa8.jsonl \
        --accuracy-logs logs/acc/*.log > report.md

Benchmark rows come from ``bench_attention.py`` (``RESULT`` JSON lines,
collected by ``sweep.py``); accuracy rows from ``check_usp.py`` logs.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def _load_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_results(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if "RESULT " in line:
                rows.append(json.loads(line.split("RESULT ", 1)[1]))
    return rows


def _group_key(label: str) -> tuple[str, str]:
    """Split ``<scenario>_<config>`` labels at the sequence-length token."""
    m = re.match(r"(.*?_\d+k)_(.*)", label)
    return (m.group(1), m.group(2)) if m else (label, "")


def bench_tables(rows: list[dict]) -> str:
    groups: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    errors = []
    for r in rows:
        scenario, config = _group_key(r.get("label", ""))
        if "error" in r:
            errors.append(f"- `{r['label']}`: {r['error']}")
            continue
        groups[scenario][config] = r
    out = []
    for scenario in sorted(groups):
        cfgs = groups[scenario]
        base_name = next((c for c in cfgs if "python_contig" in c), None)
        base = cfgs[base_name]["median_ms"] if base_name else None
        any_row = next(iter(cfgs.values()))
        out.append(
            f"\n### {scenario} (world {any_row['world']}, heads {any_row['heads']}/{any_row['kv_heads']}, "
            f"mask density {any_row['mask_density']:.3f})\n"
        )
        out.append(
            "| config | median ms | p10 / p90 | TFLOP/s/GPU | vs naive ring | ring imbalance (contiguous) | peak GiB |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for name, r in sorted(cfgs.items(), key=lambda kv: kv[1]["median_ms"]):
            speedup = f"{base / r['median_ms']:.2f}x" if base else "-"
            imb = (
                f"{r['ring_imbalance']:.3f} ({r['ring_imbalance_contiguous']:.3f})"
                if r["ring"] > 1
                else "-"
            )
            out.append(
                f"| {name} | {r['median_ms']:.1f} | {r['p10_ms']:.1f} / {r['p90_ms']:.1f} | "
                f"{r['tflops_per_gpu']:.0f} | {speedup} | {imb} | {r['peak_mem_gib']:.1f} |"
            )
    if errors:
        out.append("\n**Failed or timed-out configs**\n")
        out.extend(errors)
    return "\n".join(out)


def accuracy_table(rows: list[dict], reference: str = "fp32") -> str:
    single = {
        (r["layout"], r["seq_len"]): r for r in rows if r["check"] == "fa4_single_gpu"
    }
    cells = collections.defaultdict(dict)
    for r in rows:
        if r["check"] != "usp" or r["reference"] != reference:
            continue
        key = (r["layout"], r["seq_len"], r["ring"], r["ulysses"], r["load_balance"])
        cells[key][r["combine"]] = r
    out = [
        "| layout | seq | ring x ulysses | PTRR | out err online | flash | python | single-GPU FA4 | dq / dk / dv err (online) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(cells):
        layout, seq, ring, uly, lb = key
        c = cells[key]

        def err(mode: str) -> str:
            return f"{c[mode]['out_mean_rel']:.2e}" if mode in c else "-"

        s = single.get((layout, seq))
        sg = f"{s['out_mean_rel']:.2e}" if s else "-"
        o = c.get("online")
        grads = (
            " / ".join(
                f"{o[m]:.2e}" for m in ("dq_mean_rel", "dk_mean_rel", "dv_mean_rel")
            )
            if o
            else "-"
        )
        out.append(
            f"| {layout} | {seq // 1024}k | {ring} x {uly} | {'yes' if lb else 'no'} | "
            f"{err('online')} | {err('flash')} | {err('python')} | {sg} | {grads} |"
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bench", nargs="*", help="sweep JSONL files")
    p.add_argument("--accuracy-logs", nargs="*", default=[])
    args = p.parse_args()
    if args.accuracy_logs:
        print("## Accuracy (mean relative error vs fp32 reference)\n")
        print(accuracy_table(_load_results(args.accuracy_logs)))
    if args.bench:
        print("\n## Attention forward + backward\n")
        print(bench_tables(_load_jsonl(args.bench)))


if __name__ == "__main__":
    main()
