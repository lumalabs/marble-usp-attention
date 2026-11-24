# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Tune MagiAttention on the cells where our lead over it is small.

Every other library in the comparison is reported at the best of a sweep over
its modes; this gives MagiAttention the same treatment. For each cell:

1. screen: chunk size x overlap degree, short timing (one process per cell);
2. env: MagiAttention's performance switches on top of the best setting;
3. confirm: the three fastest settings re-timed at full length
   (3 warm-up iterations, then 3 trials x 10, as everywhere else);
4. check: the winner must pass the same forward + backward correctness check
   as every baseline (output and dQ/dK/dV against fp32, at a short length);
   if it fails, the next fastest passing setting wins.

Winners are appended to ``magi8_tuned.jsonl`` (32-head library cells) and
``magi8_models_tuned.jsonl`` (model head layouts) under the same labels as the
default-config files, so taking the fastest row per cell keeps the faster of
the default and tuned runs. Every timed setting goes to ``magi8_sweep.jsonl``.
Run from the project root, in MagiAttention's environment, on one 8-GPU node::

    python benchmarks/baselines/sweep_magi.py --out benchmarks/results/b300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import result_json  # noqa: E402

HERE = Path(__file__).resolve().parent

# (label, layout, seq_len, heads, kv_heads, library cell?). The cells where
# MagiAttention takes at most ~15% longer than our best configuration on B300.
CELLS = [
    ("causal_512k", "causal", 524288, 32, 32, True),
    ("full_128k", "full", 131072, 32, 32, True),
    ("full_512k", "full", 524288, 32, 32, True),
    ("mixed_512k", "mixed", 524288, 32, 32, True),
    ("llama_32q8kv_docs_1024k", "docs", 1048576, 32, 8, False),
    ("wan_40q_full_128k", "full", 131072, 40, 40, False),
    ("wan_40q_full_512k", "full", 524288, 40, 40, False),
    ("mha_32q_mixed_1024k", "mixed", 1048576, 32, 32, False),
    ("qwen7b_28q4kv_mixed_256k", "mixed", 262144, 28, 4, False),
    ("qwen7b_28q4kv_mixed_1024k", "mixed", 1048576, 28, 4, False),
]
# Vision-language cells on H100 (``--cells-set vlm``), with an optional
# seventh field for the head dim: the 32-head library cells and the VLM
# head layouts (Gemma 3, Molmo2-8B). MagiAttention's Hopper kernel asserts
# head_dim <= 128, so Gemma 3 4B and 12B (head_dim 256) cannot run.
VLM_CELLS = [
    ("vlm_128k", "vlm", 131072, 32, 32, True),
    ("vlm_512k", "vlm", 524288, 32, 32, True),
    ("video_128k", "video", 131072, 32, 32, True),
    ("vlm_group_128k", "vlm_group", 131072, 32, 32, True),
    ("vlm_group_512k", "vlm_group", 524288, 32, 32, True),
    ("gemma3_27b_32q16kv_vlm_256k", "vlm", 262144, 32, 16, False),
    ("molmo2_8b_32q8kv_vlm_group_256k", "vlm_group", 262144, 32, 8, False),
    ("video_512k", "video", 524288, 32, 32, True),
    ("gemma3_27b_32q16kv_vlm_1024k", "vlm", 1048576, 32, 16, False),
    ("molmo2_8b_32q8kv_vlm_group_1024k", "vlm_group", 1048576, 32, 8, False),
]

CHUNK_SIZES = ["512", "1024", "2048"]
OVERLAP_DEGREES = ["0", "1", "2", "4", "dynamic"]
# Switches read by MagiAttention at runtime; CATGQA only applies with GQA.
ENV_SWITCHES = [
    "MAGI_ATTENTION_RANGE_MERGE",
    "MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE",
    "MAGI_ATTENTION_FLATTEN_HEAD_GROUPS",
    "MAGI_ATTENTION_CATGQA",
]
CHECK_LIMIT = 5e-3  # passing baselines land at 2e-3 to 4.5e-3


def run(cmd: list[str], env: dict[str, str], timeout: int) -> tuple[list[dict], str]:
    """Run one torchrun job; return its RESULT/CHECK rows and an error, if any.

    Args:
        cmd: Full torchrun command as an argument list.
        env: Environment variables for the subprocess.
        timeout: Maximum subprocess duration in seconds.

    Returns:
        A tuple (rows, error), where rows contains parsed RESULT/CHECK
        dictionaries and error is empty on success, "timeout" on expiry, or a
        failure summary. The command is printed to stdout.
    """
    print("+", " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return [], "timeout"
    rows = [
        json.loads(line.split(" ", 1)[1])
        for line in proc.stdout.splitlines()
        if line.startswith(("RESULT ", "CHECK "))
    ]
    err = ""
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        err = "oom" if "out of memory" in proc.stderr.lower() else " | ".join(tail)
    return rows, err


def magi_cmd(
    args: argparse.Namespace,
    cell: tuple,
    chunks: list[str],
    degrees: list[str],
    timing: list[str],
    seq_len: int | None = None,
) -> list[str]:
    """Construct a MagiAttention benchmark command for one sweep cell.

    Args:
        args: Parsed sweep options containing torchrun and nproc settings.
        cell: Tuple (label, layout, seq_len, heads, kv_heads, metadata[,
            head_dim]).
        chunks: Chunk-size arguments, including "auto" when requested.
        degrees: Overlap-degree arguments.
        timing: Additional timing flags to append.
        seq_len: Optional sequence-length override; None uses the cell's
            length.

    Returns:
        An argument list for torchrun and the MagiAttention driver.
    """
    label, layout, seq, heads, kv_heads, _, *rest = cell
    head_dim = rest[0] if rest else 128
    return [
        *args.torchrun.split(),
        "--nproc-per-node",
        str(args.nproc),
        str(HERE / "magi_attn.py"),
        "--layout",
        layout,
        "--seq-len",
        str(seq_len or seq),
        "--heads",
        str(heads),
        "--kv-heads",
        str(kv_heads),
        "--head-dim",
        str(head_dim),
        "--chunk-size",
        *chunks,
        "--overlap-degree",
        *degrees,
        "--label",
        f"{label}_magi8",
        *timing,
    ]


def main() -> None:
    """Run the requested MagiAttention sweep and save measurement logs.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", type=Path, required=True, help="results directory")
    p.add_argument("--torchrun", default="torchrun")
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument(
        "--kernel-backend",
        default="fa4",
        help="MAGI_ATTENTION_KERNEL_BACKEND (fa4 = its Blackwell FFA_FA4 backend; "
        "empty = its default, the Hopper FFA kernel).",
    )
    p.add_argument("--cells-set", choices=["b300", "vlm"], default="b300")
    p.add_argument("--cells", nargs="+", help="only these labels (default: all)")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per job")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pool = VLM_CELLS if args.cells_set == "vlm" else CELLS
    cells = [c for c in pool if not args.cells or c[0] in args.cells]
    base_env = {
        **os.environ,
        **(
            {"MAGI_ATTENTION_KERNEL_BACKEND": args.kernel_backend}
            if args.kernel_backend
            else {}
        ),
        # One cached runtime at a time: every setting builds its own key.
        "MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE": "1",
    }
    screen = ["--warmup", "2", "--trials", "1", "--iters", "5"]
    full = ["--warmup", "3", "--trials", "3", "--iters", "10"]
    if args.dry_run:
        for cell in cells:
            print(" ".join(magi_cmd(args, cell, CHUNK_SIZES, OVERLAP_DEGREES, screen)))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    sweep_log = args.out / "magi8_sweep.jsonl"
    checks_log = args.out / "magi_checks_tuned.jsonl"

    def log(path: Path, row: dict) -> None:
        with path.open("a") as f:
            f.write(result_json(row) + "\n")

    for cell in cells:
        label = cell[0]
        t0 = time.time()
        # 1. screen chunk size x overlap degree in one process.
        rows, err = run(
            magi_cmd(args, cell, CHUNK_SIZES, OVERLAP_DEGREES, screen),
            base_env,
            args.timeout,
        )
        if err == "oom":
            # One setting ran out of memory and took the process down with it:
            # screen every setting in its own process instead.
            rows = []
            for chunk, degree in [(c, d) for c in CHUNK_SIZES for d in OVERLAP_DEGREES]:
                one, one_err = run(
                    magi_cmd(args, cell, [chunk], [degree], screen),
                    base_env,
                    args.timeout,
                )
                rows += one
                if one_err:
                    log(
                        sweep_log,
                        {
                            "label": f"{label}_magi8",
                            "stage": "screen",
                            "chunk_size": chunk,
                            "overlap_degree": degree,
                            "error": one_err,
                        },
                    )
        for r in rows:
            log(sweep_log, {**r, "stage": "screen"})
        if not rows:
            print(f"{label}: screen failed ({err})", flush=True)
            log(sweep_log, {"label": f"{label}_magi8", "stage": "screen", "error": err})
            continue
        candidates = [(r["median_ms"], r["magi_config"]) for r in rows]
        best = min(candidates, key=lambda c: c[0])[1]

        # 2. env switches on top of the best chunk size and degree.
        for switch in ENV_SWITCHES:
            if switch.endswith("CATGQA") and cell[3] == cell[4]:
                continue
            env = {**base_env, switch: "1"}
            rows, err = run(
                magi_cmd(
                    args,
                    cell,
                    [str(best["chunk_size"])],
                    [best["overlap_degree"]],
                    screen,
                ),
                env,
                args.timeout,
            )
            for r in rows:
                log(sweep_log, {**r, "stage": "env"})
                candidates.append((r["median_ms"], r["magi_config"]))
            if err:
                log(
                    sweep_log,
                    {
                        "label": f"{label}_magi8",
                        "stage": "env",
                        "env": switch,
                        "error": err,
                    },
                )

        # 3. confirm the three fastest at full length; 4. check the winner.
        confirmed = []
        for _, cfg in sorted(candidates, key=lambda c: c[0])[:3]:
            env = {**base_env, **cfg["env"]}
            rows, err = run(
                magi_cmd(
                    args, cell, [str(cfg["chunk_size"])], [cfg["overlap_degree"]], full
                ),
                env,
                args.timeout,
            )
            for r in rows:
                log(sweep_log, {**r, "stage": "confirm"})
                confirmed.append(r)
        winner = None
        for r in sorted(confirmed, key=lambda r: r["median_ms"]):
            cfg = r["magi_config"]
            check_seq = max(8192, cfg["chunk_size"] * args.nproc)
            rows, err = run(
                magi_cmd(
                    args,
                    cell,
                    [str(cfg["chunk_size"])],
                    [cfg["overlap_degree"]],
                    ["--check"],
                    seq_len=check_seq,
                ),
                {**base_env, **cfg["env"]},
                args.timeout,
            )
            check = rows[0] if rows else {"error": err or "no CHECK line"}
            log(checks_log, {**check, "label": f"{label}_magi8"})
            errors = [check.get(k, float("inf")) for k in ("out", "dq", "dk", "dv")]
            if max(errors) <= CHECK_LIMIT:
                winner = r
                break
        if winner is None:
            print(f"{label}: no tuned setting passed the check", flush=True)
            continue
        out = args.out / (
            "magi8_tuned.jsonl" if cell[5] else "magi8_models_tuned.jsonl"
        )
        log(out, {**winner, "tuned": True})
        print(
            f"{label}: {winner['median_ms']:.1f} ms with {winner['magi_config']} "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
