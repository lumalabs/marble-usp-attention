# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Run the distributed GPU exactness gate on an H100 or B300 node.

Tests output and dQ/dK/dV against fp32 and requires their mean-relative error
not to exceed single-GPU FA4's error by more than 5% (plus 1e-6 roundoff).
Every case must produce both references; missing rows and nonfinite values fail.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TENSORS = ("out", "dq", "dk", "dv")


def validate_rows(
    rows: list[dict], *, ratio: float = 1.05, floor: float = 1e-6
) -> None:
    """Validate output and gradient errors against FA4 and fp32 references.

    Args:
        rows: One single-GPU FA4/fp32 row and two USP reference rows, with
            out/dq/dk/dv mean-relative, maximum-absolute and mean-absolute
            errors.
        ratio: Multiplier applied to the single-GPU FA4 mean-relative error.
        floor: Additive allowance for roundoff in the relative-error budget.

    Raises:
        ValueError: References are missing or duplicated, metrics are nonfinite
            or negative, or an accuracy bound is exceeded.
    """
    single = [r for r in rows if r.get("check") == "fa4_single_gpu"]
    comparisons = [r for r in rows if r.get("check") == "usp"]
    if len(single) != 1 or len(comparisons) != 2:
        raise ValueError("expected one FA4/fp32 row and two USP reference rows")
    by_reference = {r["reference"]: r for r in comparisons}
    if set(by_reference) != {"fa4", "fp32"}:
        raise ValueError("both single-GPU FA4 and fp32 references are required")
    for name in TENSORS:
        key = f"{name}_mean_rel"
        for row in rows:
            for metric in (f"{name}_mean_rel", f"{name}_max_abs", f"{name}_mean_abs"):
                value = row[metric]
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"nonfinite or negative {metric}: {value}")
        error = by_reference["fp32"][key]
        limit = ratio * single[0][key] + floor
        if error > limit:
            raise ValueError(
                f"{name}: fp32 error {error:.6g} exceeds FA4 budget {limit:.6g}"
            )
        # A broken single-GPU reference must not make the relative gate vacuous.
        if single[0][key] > 1e-2 or by_reference["fa4"][key] > 1e-2:
            raise ValueError(f"{name}: absolute mean-relative error exceeds 1e-2")


def cases(gpu: str, nproc: int) -> list[tuple[str, list[str]]]:
    """Build the hardware-specific exactness case matrix.

    Args:
        gpu: Hardware family: "h100" uses online merging; "b300" uses fp32.
        nproc: Total worker count, supported by the runner at 4 or 8.

    Returns:
        A list of (case_name, argument_list) tuples for the distributed
        correctness driver, including hardware-specific head-256 coverage.
    """
    combine = "online" if gpu == "h100" else "fp32"
    base = [
        "--seq-len",
        "4096",
        "--heads",
        "8",
        "--kv-heads",
        "8",
        "--combine",
        combine,
        "--no-fa3",
        "--interval-masks",
        "off",
    ]
    result = []
    for layout in ("causal", "full", "docs", "mixed", "vlm_group"):
        for ring in (1, nproc):
            flags = [
                *base,
                "--kv-heads",
                "8" if ring == 1 else "2",
                "--layout",
                layout,
                "--ring",
                str(ring),
                "--ulysses",
                str(nproc // ring),
                "--dense-ring",
                "off",
            ]
            result.append((f"{layout}-r{ring}", flags))
    for exchange in ("ring", "gather"):
        for dim in (128,):
            # GQA, a hybrid mesh, batch > 1, and exact dense generation steps.
            result.append(
                (
                    f"dense-{exchange}-d{dim}",
                    [
                        *base,
                        "--layout",
                        f"file:{PROJECT / 'repro/layouts/exactness.json'}#0",
                        "--ring",
                        str(nproc // 2),
                        "--ulysses",
                        "2",
                        "--kv-heads",
                        "2",
                        "--head-dim",
                        str(dim),
                        "--batch",
                        "2",
                        "--kv-exchange",
                        exchange,
                        "--dense-ring",
                        "on",
                        "--exact-dense-ring",
                        "--dense-min-span",
                        "0",
                    ],
                )
            )
    result.append(
        (
            "head256",
            [
                *base,
                "--layout",
                "docs" if gpu == "h100" else "full",
                "--dense-ring",
                "on" if gpu == "h100" else "off",
                "--ring",
                str(nproc // 2),
                "--ulysses",
                "2",
                "--kv-heads",
                "2",
                "--head-dim",
                "256",
                "--kv-exchange",
                "gather" if gpu == "h100" else "ring",
            ],
        )
    )
    result.append(
        (
            "window",
            [
                *base,
                "--layout",
                "docs",
                "--ring",
                str(nproc),
                "--ulysses",
                "1",
                "--balance",
                "chunks",
                "--kv-exchange",
                "gather",
                "--kv-heads",
                "2",
                "--sliding-window",
                "128",
                "--dense-ring",
                "off",
            ],
        )
    )
    return result


def main() -> None:
    """Run or preview the CLI exactness matrix and save worker rows and logs.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=["h100", "b300"], required=True)
    parser.add_argument("--nproc", type=int, default=8, choices=[4, 8])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", action="append", help="Run only the named case(s)")
    args = parser.parse_args()
    available = cases(args.gpu, args.nproc)
    unknown = set(args.case or ()) - {name for name, _ in available}
    if unknown:
        parser.error(f"unknown cases for nproc={args.nproc}: {sorted(unknown)}")
    matrix = [
        (name, flags) for name, flags in available if not args.case or name in args.case
    ]
    if not args.dry_run:
        import torch  # optional for command preview on CPU-only machines

        if torch.cuda.device_count() < args.nproc:
            parser.error(f"need at least {args.nproc} visible GPUs")
        expected = "H100" if args.gpu == "h100" else "B300"
        if any(
            expected not in torch.cuda.get_device_name(i) for i in range(args.nproc)
        ):
            parser.error(f"all GPUs must be {expected}")
        args.out.mkdir(parents=True, exist_ok=False)
    # Check the overlapping noise-prefix regression early.
    matrix.sort(key=lambda case: case[0] != "dense-gather-d128")
    for name, flags in matrix:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(args.nproc),
            str(PROJECT / "tests/distributed/check_usp.py"),
            *flags,
        ]
        print(name + ": " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        log = args.out / f"{name}.log"
        with log.open("w") as output:
            subprocess.run(
                command,
                cwd=PROJECT,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        rows = [
            json.loads(line.removeprefix("RESULT "))
            for line in log.read_text().splitlines()
            if line.startswith("RESULT ")
        ]
        with (args.out / "exactness.jsonl").open("a") as output:
            for row in rows:
                output.write(json.dumps({"case": name, **row}) + "\n")
        validate_rows(rows)
        if name == "head256" and args.gpu == "h100":
            if not all(
                row.get("dense_gather") for row in rows if row.get("check") == "usp"
            ):
                raise ValueError("H100 head256 did not exercise dense gather")
        if name.startswith("dense-"):
            field = (
                "generation_ring" if name.startswith("dense-ring-") else "dense_gather"
            )
            if not all(row.get(field) for row in rows if row.get("check") == "usp"):
                raise ValueError(f"{name} did not exercise the requested {field} path")
        print(f"PASS {name}", flush=True)


if __name__ == "__main__":
    main()
