# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Replay recorded MARBLE/Magi/full-model configurations on packed or named layouts.

Print a JSON plan by default. --execute requires the appropriate isolated GPU
environment and the same command on every node (NODE_RANK, MASTER_ADDR). Historical
missing environment metadata is not inferred; these are new measurements.
FlashMask balanced, hybrid, and Blackwell overlap use the archived public driver.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "benchmarks"))
import report_tables as report  # noqa: E402

TABLES = ("scale", "ablation", "long-video", "b300", "end-to-end")
# Keep planning dependency-free in isolated baseline environments. Tests check
# these names against usp_attention.layouts.LAYOUT_NAMES.
BUILTIN_LAYOUTS = (
    "causal",
    "full",
    "docs",
    "mixed",
    "tiny_kv",
    "vlm",
    "video",
    "vlm_group",
    "video_group",
)


def selected(row: dict, table: str) -> bool:
    """Match the candidate rows used by the report's five table sections.

    Args:
        row: Loaded measurement row including report source metadata.
        table: One of the five names in TABLES.

    Returns:
        Whether the row belongs to the requested table's candidate set.
    """
    model = row.get("bench") == "model"
    if table == "end-to-end":
        return model
    if model:
        return False
    if table == "b300":
        return row["_file"].startswith("b300") and report.kind(row) in (
            "ours",
            "ours_inexact",
            "fm",
            "fm_hybrid",
            "fm_overlap",
        )
    if table == "ablation":
        return (
            row["_file"] == "abl32"
            and "video_2048k" in report.layout(row)
            and report.kind(row) == "ours"
            and not row.get("sliding_window")
            and "dring" not in (row.get("label") or "")
        )
    if row["_file"].startswith("b300") or row["world"] < 32:
        return False
    if table == "long-video":
        return (
            report.model(row) == "bagel"
            and "video" in report.layout(row)
            and report.kind(row) in ("ours", "fm", "fm_hybrid")
        )
    return report.kind(row) in ("ours", "fm", "fm_hybrid", "magi")


def system(row: dict) -> str:
    """Select the environment that owns this measurement.

    Args:
        row: Measurement row whose implementation or attention backend is
            inspected.

    Returns:
        The owning environment name: "ours", "magi", or "flashmask".
    """
    impl = row.get("impl", "")
    if impl.startswith("magi"):
        return "magi"
    if impl.startswith("flashmask") or row.get("attn") == "flashmask":
        return "flashmask"
    return "ours"


def measurement_command(row: dict, python: str = sys.executable) -> list[str]:
    """Reconstruct a public benchmark command from a saved measurement.

    Args:
        row: Saved measurement with layout, shape and implementation options.
            File-backed layouts must have a committed pack; named layouts are
            passed through to drivers that support them.
        python: Python executable from the selected system's environment.

    Returns:
        An argument list for torch.distributed.run and the public benchmark
        driver. The execution wrapper supplies node topology and output
        options.

    Raises:
        ValueError: A required pack is missing or the FlashMask attention
            driver is asked to replay a built-in layout.
    """
    model = row.get("bench") == "model"
    owner = system(row)
    script = (
        "benchmarks/bench_model.py"
        if model
        else "benchmarks/baselines/flashmask_cp_bench.py"
        if owner == "flashmask"
        else "benchmarks/baselines/magi_attn.py"
        if owner == "magi"
        else "benchmarks/bench_attention.py"
    )
    layout = row["layout"]
    if layout in BUILTIN_LAYOUTS:
        if owner == "flashmask" and not model:
            raise ValueError("FlashMask attention replay requires a committed pack")
    else:
        name = layout.removeprefix("file:").split("/")[-1].split("#")[0]
        name = name.removesuffix(".gz").removesuffix(".json")
        pack = PROJECT / "benchmarks/layouts/packs" / (name + ".json.gz")
        if not pack.is_file():
            raise ValueError(f"missing committed pack: {name}")
        layout = f"file:benchmarks/layouts/packs/{name}.json#0"
    cmd = [
        python,
        "-m",
        "torch.distributed.run",
        script,
        "--layout",
        layout,
    ]
    if owner == "flashmask" and not model:
        cmd[-1] = cmd[-1].removeprefix("file:")
    fields = ["seq_len", "label", "sliding_window"]
    if owner == "flashmask" and not model:
        fields += [
            "heads",
            "kv_heads",
            "head_dim",
            "balance_chunk",
            "ulysses",
            "replicate_kv",
        ]
        cmd += ["--mode", "overlap" if "overlap" in row.get("impl", "") else "balanced"]
        if row.get("ipo"):
            cmd.append("--ipo")
    elif owner == "magi":
        fields += ["heads", "kv_heads", "head_dim"]
        config = row.get("magi_config", {})
        for key in ("chunk_size", "overlap_degree", "overlap_alg", "min_chunk_size"):
            if key in config:
                cmd += ["--" + key.replace("_", "-"), str(config[key])]
    else:
        fields += [
            "ring",
            "ulysses",
            "combine",
            "replicate_kv",
            "balance",
            "kv_exchange",
            "transport",
            "gather_max_keys",
        ]
        if model:
            fields += ["attn", "model", "layers", "global_ulysses", "dense_ring"]
            if row.get("replan"):
                cmd.append("--replan")
        else:
            fields += [
                "heads",
                "kv_heads",
                "head_dim",
                "batch",
                "interval_masks",
                "dense_ring_chunks",
            ]
        # Gather above head_dim 128 has no block-sparse FA4 kernel on H100, so
        # those runs used dense calls although their rows record dense_ring=False.
        wide_gather = (
            row.get("kv_exchange") == "gather" and (row.get("head_dim") or 0) > 128
        )
        # The chunked (generation) ring and the merged dense gather are dense-ring
        # paths: on H100 they run only with --dense-ring on, since the driver's auto
        # mode is Blackwell-only. Rows recorded before the driver stamped
        # generation_ring carry the path only in their label ("..._dring...", with
        # "_c<N>" for a forced chunk count and "_plain" for an unrefined plan).
        label = row.get("label") or ""
        legacy_dring = "_dring" in label and row.get("generation_ring") is None
        if not model and (
            row.get("dense_ring")
            or row.get("generation_ring")
            or row.get("exact_dense_ring")
            or row.get("dense_gather_merge")
            or legacy_dring
            or wide_gather
        ):
            cmd += ["--dense-ring", "on"]
        if legacy_dring:
            forced = re.search(r"_c(\d+)", label)
            if forced and row.get("dense_ring_chunks") is None:
                cmd += ["--dense-ring-chunks", forced.group(1)]
            if "_plain" in label and row.get("dense_ring_refine") is None:
                cmd.append("--no-dense-ring-refine")
        for key in ("fa3_gather", "gather_regather", "exact_dense_ring"):
            if row.get(key):
                cmd += ["--" + key.replace("_", "-")]
        for key, flag in (
            ("load_balance", "--no-load-balance"),
            ("group_reorder", "--no-group-reorder"),
        ):
            if row.get(key) is False:
                cmd.append(flag)
        if not model:
            for key, flag in (
                ("fa3_on_hopper", "--no-fa3"),
                ("comm_stream", "--no-comm-stream"),
                ("fp32_grads", "--bf16-grads"),
                ("dense_ring_refine", "--no-dense-ring-refine"),
            ):
                if row.get(key) is False:
                    cmd.append(flag)
            if row.get("forward_only"):
                cmd.append("--forward-only")
    for key in fields:
        if row.get(key) is not None:
            cmd += ["--" + key.replace("_", "-"), str(row[key])]
    # Dense-ring's recorded activation is an outcome, not its original CLI setting.
    # A recorded exact flag now reaches USPConfig (older drivers ignored it).
    return cmd


def main() -> None:
    """Print a replay plan or execute it using CLI topology and output options.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=TABLES)
    parser.add_argument("--system", choices=("ours", "magi", "flashmask"))
    parser.add_argument("--world", type=int)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report.RESULTS = report.DEFAULT_RESULTS
    rows = [
        r
        for r in report.load(extra=report.PAPER_EXTRA)
        if selected(r, args.table)
        and (args.system is None or system(r) == args.system)
        and (args.world is None or r["world"] == args.world)
    ]
    if not rows:
        parser.error("no matching measurement rows")
    plan = []
    for row in rows:
        item = {"source": report.src(row), "world": row["world"], "system": system(row)}
        try:
            item["command"] = measurement_command(row)
        except ValueError as error:
            item["blocker"] = str(error)
        if system(row) == "magi":
            item["environment"] = row.get("magi_config", {}).get("env", {})
        plan.append(item)
    if not args.execute:
        print(
            json.dumps(
                {
                    "table": args.table,
                    "historical_environment": "unknown",
                    "note": "New measurements with recorded options; not historical environment reconstruction.",
                    "rows": plan,
                },
                indent=2,
            )
        )
        return
    if not args.system or not args.world or not args.out:
        parser.error("--execute requires --system, --world, and --out")
    if args.gpus_per_node < 1 or args.world < 1:
        parser.error("GPU counts must be positive")
    if args.world % args.gpus_per_node:
        parser.error("world must be divisible by gpus-per-node")
    if any("blocker" in item for item in plan):
        parser.error("selected plan contains unavailable launchers; inspect the plan")
    nodes = args.world // args.gpus_per_node
    if nodes > 1 and not {"NODE_RANK", "MASTER_ADDR"} <= os.environ.keys():
        parser.error(
            "multi-node execution needs NODE_RANK and MASTER_ADDR on every node"
        )
    rank = int(os.environ.get("NODE_RANK", "0"))
    if rank < 0 or rank >= nodes:
        parser.error("NODE_RANK must be within the allocated node count")
    args.out.mkdir(parents=True, exist_ok=True)
    for row, item in zip(rows, plan, strict=True):
        log = args.out / f"{row['_file']}-{row['_line']}-node{rank}.log"
        if log.exists():
            parser.error(f"refusing to overwrite {log}")
        command = item["command"]
        launch = [
            f"--nproc-per-node={args.gpus_per_node}",
            f"--nnodes={nodes}",
            f"--node-rank={rank}",
            "--master-addr=" + os.environ.get("MASTER_ADDR", "127.0.0.1"),
            "--master-port=" + os.environ.get("MASTER_PORT", "29500"),
        ]
        command = command[:3] + launch + command[3:]
        with log.open("x") as output:
            completed = subprocess.run(
                command,
                cwd=PROJECT,
                env={**os.environ, **item.get("environment", {})},
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        completed.check_returncode()
        if rank == 0:
            measured = [
                json.loads(line[7:])
                for line in log.read_text().splitlines()
                if line.startswith("RESULT ")
            ]
            if not measured:
                raise RuntimeError(f"no result rows in {log}")
            with (args.out / (row["_file"] + ".jsonl")).open("a") as output:
                for result in measured:
                    result["reproduction_source"] = item["source"]
                    output.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
