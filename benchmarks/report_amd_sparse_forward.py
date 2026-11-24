# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Render AMD appendix latency/throughput tables from complete JSONL runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONFIGS = ("ulysses_dp", "ring_contiguous", "ring_ptrr", "usp")


def render(paths: list[Path]) -> str:
    runs = [
        (
            path,
            [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ],
        )
        for path in paths
    ]
    modes = {
        row.get("kv_exchange", "ring")
        for _, records in runs
        for row in records
        if row["config"] == "usp"
    }
    if len(modes) != 1:
        raise ValueError("report requires a consistent USP KV exchange mode")
    usp_label = "USP 2×4 KV gather" if modes == {"gather"} else "USP 2×4"
    lines = [
        "Forward latency in ms; brackets are the range of trial medians.",
        "",
        f"| Workload | Ulysses-4 + DP | Contiguous ring-8 | PTRR ring-8 | {usp_label} | U+DP / USP latency | USP / U+DP throughput |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for path, records in runs:
        rows = {row["config"]: row for row in records}
        if len(records) != len(CONFIGS) or set(rows) != set(CONFIGS):
            raise ValueError(
                f"{path}: expected exactly one row per configuration; use a fresh output per run"
            )
        if any(not row["forward_only"] or row["world"] != 8 for row in records):
            raise ValueError(
                f"{path}: this table requires forward-only eight-GPU results"
            )
        for field in (
            "seq_len",
            "layout",
            "heads",
            "kv_heads",
            "extension_sha256",
            "git_sha",
            "amd_source_sha256",
            "metadata_sha256",
        ):
            if len({str(row[field]) for row in records}) != 1:
                raise ValueError(f"{path}: inconsistent {field}")
        if len({row.get("round_mode", "rtne") for row in records}) != 1:
            raise ValueError(f"{path}: inconsistent round_mode")
        cells = []
        for name in CONFIGS:
            row = rows[name]
            medians = row["trial_medians_ms"]
            cells.append(
                f"{row['median_ms']:.1f} [{min(medians):.1f}, {max(medians):.1f}]"
            )
        dp, usp = rows["ulysses_dp"], rows["usp"]
        label = "BAGEL pack" if dp["layout"].startswith("file:") else dp["layout"]
        label += f" {dp['seq_len'] // 1024}k"
        latency_ratio = dp["median_ms"] / usp["median_ms"]
        throughput_ratio = (
            usp["useful_tokens_per_second"] / dp["useful_tokens_per_second"]
        )
        lines.append(
            "| "
            + " | ".join(
                [label, *cells, f"{latency_ratio:.2f}×", f"{throughput_ratio:.2f}×"]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    print(render(args.results))


if __name__ == "__main__":
    main()
