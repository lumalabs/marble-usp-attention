# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Build the benchmark results document from a results directory.

Reads the JSONL files written by ``sweep.py`` (and ``check_usp.py`` logs for
accuracy) and prints Markdown::

    python benchmarks/summarize.py benchmarks/results > benchmarks/RESULTS.md

Expected files (any may be missing): ``accuracy.log``, ``ablation.jsonl``,
``heads8.jsonl``, ``heads16.jsonl``, ``transport8.jsonl``,
``transport16.jsonl``, ``torchcp8.jsonl``, ``torchcp8_match.jsonl``,
``te8.jsonl``, ``yunchang8.jsonl``, ``magi8.jsonl``, ``model8.jsonl``,
``collectives.jsonl``, and ``convergence/*.jsonl`` (``convergence.py`` logs).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from report import accuracy_table, bench_tables


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("RESULT "):
            line = line[len("RESULT ") :]
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def _scenario(label: str) -> tuple[str, str]:
    m = re.match(r"(.*?_\d+k)_(.*)", label)
    return (m.group(1), m.group(2)) if m else (label, "")


def _family(config: str) -> str:
    """Collapse a config name to a column across GPU counts."""
    for pattern, name in [
        (r"^ulysses\d+_dp\d+", "ulysses_dp"),
        (r"^ulysses\d+", "ulysses"),
        (r"^ring\d+_python_contig$", "naive_ring"),
        (r"^ring\d+_online_ptrr$", "ring_ptrr"),
        (r"^ring\d+_online_ptrr_nccl4py$", "ring_ptrr_nccl4py"),
        (r"^usp_.*_nccl4py$", "usp_nccl4py"),
        (r"^usp_", "usp"),
    ]:
        if re.match(pattern, config):
            return name
    return config


def scaling_table(rows8: list[dict], rows16: list[dict]) -> str:
    """Per model scenario: best time per family at 8 and 16 GPUs, and tokens/s/GPU."""
    cells: dict[str, dict[str, dict[int, tuple[float, str, float]]]] = {}
    for rows in (rows8, rows16):
        for r in rows:
            if "error" in r:
                continue
            scenario, config = _scenario(r["label"])
            fam = _family(config)
            seqs = r.get("dp_replicas", 1)
            tok_per_gpu = r["seq_len"] * seqs / (r["median_ms"] * 1e-3) / r["world"]
            cur = cells.setdefault(scenario, {}).setdefault(fam, {}).get(r["world"])
            if cur is None or r["median_ms"] < cur[0]:
                cells[scenario][fam][r["world"]] = (r["median_ms"], config, tok_per_gpu)
    families = [
        "ulysses",
        "ulysses_dp",
        "naive_ring",
        "ring_ptrr",
        "ring_ptrr_nccl4py",
        "usp",
        "usp_nccl4py",
    ]
    out = [
        "Cells: step time in ms (best layout of that family) / tokens per second per GPU, "
        "8 GPUs | 16 GPUs. `-` means the head layout does not allow it.",
        "",
        "| scenario | " + " | ".join(families) + " |",
        "|---|" + "---|" * len(families),
    ]
    for scenario in sorted(cells):
        fams = cells[scenario]
        row = [scenario]
        for fam in families:
            parts = []
            for world in (8, 16):
                v = fams.get(fam, {}).get(world)
                parts.append(f"{v[0]:.0f} / {v[2] / 1e3:.0f}k" if v else "-")
            row.append(" \\| ".join(parts))
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def transport_table(rows: list[dict]) -> str:
    out = ["| config | median ms |", "|---|---|"]
    for r in rows:
        out.append(
            f"| {r['label']} | {r.get('median_ms', float('nan')):.1f} |"
            if "error" not in r
            else f"| {r['label']} | {r['error']} |"
        )
    return "\n".join(out)


# Every implementation's best config per strategy, as (column, implementation, label regex).
_BASELINE_CELLS = [
    ("Ulysses", "ours", r"ulysses8$"),
    ("Ulysses", "TE", r"te_a2a8$"),
    ("Ulysses", "yunchang", r"yunchang_ulysses8_fa3$"),
    ("ring", "ours", r"ring8_online_ptrr$"),
    ("ring", "TE", r"te_p2p8$"),
    ("ring", "torch", r"torch_sdpa_ring_.*$"),
    ("ring", "yunchang", r"yunchang_ring8_fa3$"),
    ("ring x Ulysses", "ours", r"usp_r\du\d_online_ptrr$"),
    ("ring x Ulysses", "TE", r"te_a2a_p2p_r\du\d$"),
    ("ring x Ulysses", "yunchang", r"yunchang_usp_r\du\d_fa3$"),
    ("all-gather / other", "TE", r"te_allgather8$"),
    ("all-gather / other", "torch flex", r"torch_flex_allgather_ptrr$"),
    ("all-gather / other", "Magi", r"magi8$"),
]


def baseline_table(rows: list[dict]) -> str:
    """Per scenario and strategy: best step time of each implementation."""
    best: dict[tuple[str, str, str], float] = {}
    for r in rows:
        if "error" in r or r.get("world") != 8 or r.get("heads") != r.get("kv_heads"):
            continue
        scenario, config = _scenario(r["label"])
        for column, impl, pattern in _BASELINE_CELLS:
            if re.search(pattern, config):
                key = (scenario, column, impl)
                best[key] = min(best.get(key, float("inf")), r["median_ms"])
    columns = list(dict.fromkeys(c for c, _, _ in _BASELINE_CELLS))
    impls = {c: [i for c2, i, _ in _BASELINE_CELLS if c2 == c] for c in columns}
    order = ["mixed", "docs", "causal", "full"]
    scenarios = sorted(
        {k[0] for k in best},
        key=lambda s: (order.index(s.split("_")[0]), int(s.split("_")[1][:-1])),
    )
    out = [
        "Step time (forward + backward) in ms, 8x H100, 32 MHA heads, head dim 128; "
        "lower is better. yunchang 0.6.4 ignores `causal` in its FA3 wrappers; its causal rows "
        "use the patched wrappers in `baselines/yunchang_usp.py` (outputs and gradients checked "
        "against SDPA). Each cell lists the implementations in the order of the header; "
        "for ring x Ulysses the best of r2u4 / r4u2. `-` means the implementation does not "
        "support the mask (yunchang: plain causal or full only; TE: no mixed masks) or the "
        "run did not complete.",
        "",
        "| scenario | "
        + " | ".join(f"{c} ({' / '.join(impls[c])})" for c in columns)
        + " |",
        "|---|" + "---|" * len(columns),
    ]
    for scenario in scenarios:
        cells = []
        for c in columns:
            vals = [best.get((scenario, c, i)) for i in impls[c]]
            cells.append(" / ".join(f"{v:.0f}" if v else "-" for v in vals))
        out.append(f"| {scenario} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def convergence_table(runs: dict[str, list[dict]]) -> str:
    """Loss of each run vs the Ulysses reference, against the reference's own rerun."""
    ref = runs.get("ulysses_ref")
    if not ref:
        return ""

    def smooth(rows: list[dict], window: int = 50) -> list[float]:
        losses = [r["loss"] for r in rows]
        return [
            statistics.fmean(losses[max(0, i - window + 1) : i + 1])
            for i in range(len(losses))
        ]

    ref_s = smooth(ref)
    out = [
        "Same model init and data order; the Ulysses run computes attention for "
        "the whole sequence in one FA4 call per head subset, so it is the "
        "single-GPU numerics. `ulysses_ref_repeat` is a rerun of that reference: "
        "FA4's backward is not bitwise deterministic, so it gives the run-to-run noise floor. "
        "Loss columns are means over the last 200 steps; the gap is the largest difference "
        "of the 50-step moving averages after step 200.",
        "",
        "| run | steps | final loss | delta vs ref | max smoothed gap | mean step s |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(runs, key=lambda n: (n != "ulysses_ref", n)):
        rows = runs[name]
        n = min(len(rows), len(ref))
        if n < 400:
            continue
        tail = statistics.fmean(r["loss"] for r in rows[n - 200 : n])
        ref_n = statistics.fmean(r["loss"] for r in ref[n - 200 : n])
        s = smooth(rows)
        gap = max((abs(s[i] - ref_s[i]) for i in range(200, n)), default=0.0)
        step_s = statistics.median(r["step_s"] for r in rows[10:n])
        note = "" if n == len(ref) else f" (at step {n})"
        out.append(
            f"| {name}{note} | {len(rows)} | {tail:.4f} | {tail - ref_n:+.4f} | "
            f"{gap:.4f} | {step_s:.3f} |"
        )
    return "\n".join(out)


def kernel_table(rows: list[dict]) -> str:
    """FA3 vs FA4 on one GPU, the per-GPU shape of Ulysses-8 with 32 heads."""
    out = [
        "One H100, 4 heads x head dim 128 over the whole sequence (the per-GPU work of "
        "Ulysses-8 with 32 heads), native causal / full masks, median ms.",
        "",
        "| layout | seq | FA3 fwd | FA4 fwd | FA3 fwd+bwd | FA4 fwd+bwd | FA4 vs FA3 |",
        "|---|---|---|---|---|---|---|",
    ]
    by = {(r["layout"], r["seq_len"], r["impl"]): r for r in rows}
    for layout, seq in sorted({(r["layout"], r["seq_len"]) for r in rows}):
        a, b = by.get((layout, seq, "fa3")), by.get((layout, seq, "fa4"))
        if not (a and b):
            continue
        out.append(
            f"| {layout} | {seq // 1024}k | {a['fwd_ms']:.1f} | {b['fwd_ms']:.1f} | "
            f"{a['fwd_bwd_ms']:.1f} | {b['fwd_bwd_ms']:.1f} | "
            f"{(b['fwd_bwd_ms'] / a['fwd_bwd_ms'] - 1) * 100:+.1f}% |"
        )
    return "\n".join(out)


def baseline_check_table(rows: list[dict]) -> str:
    """Forward + backward check of every baseline against fp32 (``--check`` modes)."""
    out = [
        "Each baseline sharded with its own layout, forward + backward on 8 GPUs, "
        "8k tokens, compared with an fp32 dense reference on every rank's tokens. "
        "bf16 attention lands at about 2e-3 to 4e-3; a baseline is only benchmarked "
        "if it passes (all errors below 1e-2). `yunchang_fa3_zigzag` without "
        "`_patched` is yunchang 0.6.4 as shipped, kept to show its causal bug.",
        "",
        "| implementation | layout | config | out | dq | dk | dv | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        errs = [r[k] for k in ("out", "dq", "dk", "dv")]
        config = []
        if r.get("ring") is not None:
            config.append(f"{r['ring']} x {r['ulysses']}")
        elif r.get("ulysses") is not None:
            config.append(f"ulysses {r['ulysses']}")
        if r.get("kv_heads") not in (None, 32):
            config.append(f"{r['kv_heads']} kv heads")
        out.append(
            f"| {r['impl']} | {r['layout']} | {', '.join(config) or '-'} | "
            + " | ".join(f"{e:.2e}" for e in errs)
            + f" | {'yes' if max(errs) < 1e-2 else '**no**'} |"
        )
    return "\n".join(out)


def ring_accuracy_table(ours: list[dict], checks: list[dict]) -> str:
    """Ring-only accuracy of every implementation at the baseline-check shape."""
    lines = {}
    for r in ours:
        if r.get("reference") != "fp32":
            continue
        name = (
            "single-GPU FA4 (no parallelism)"
            if r.get("check") == "fa4_single_gpu"
            else f"ours, {r['combine']} combine"
        )
        lines.setdefault(name, {})[r["layout"]] = [
            r[f"{k}_mean_rel"] for k in ("out", "dq", "dk")
        ]
    rings = {
        "te_p2p": "TransformerEngine p2p",
        "torch_sdpa_ring": "PyTorch SDPA ring",
    }
    for r in checks:
        name = rings.get(r["impl"])
        if r["impl"].startswith("yunchang") and r["impl"].endswith("_patched"):
            if (r.get("ring"), r.get("ulysses")) == (8, 1):
                name = "yunchang ring (patched)"
        if (
            name
            and r.get("kv_heads") in (None, 32)
            and r["layout"] in ("causal", "full")
        ):
            lines.setdefault(name, {})[r["layout"]] = [r["out"], r["dq"], r["dk"]]
    out = [
        "8-way ring, 8k tokens, 32 MHA heads, head dim 128; mean relative error vs "
        "fp32 (same shape as the baseline checks).",
        "",
        "| implementation | causal out | causal dq | causal dk | full out | full dq | full dk |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, by_layout in lines.items():
        cells = []
        for layout in ("causal", "full"):
            errs = by_layout.get(layout)
            cells += [f"{e:.2e}" for e in errs] if errs else ["-"] * 3
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def model_parity_table(rows: list[dict]) -> str:
    """7B multimodal MoT decoder parity: USP vs single-GPU FA4, both against fp32."""

    def key(r: dict) -> tuple:
        return (
            r["layout"],
            r["ring"],
            r["ulysses"],
            r["fp32_grads"],
            r.get("path", "block-sparse"),
        )

    fa4 = {key(r): r for r in rows if r.get("check") == "fa4"}
    out = [
        "Two full-size 7B multimodal MoT decoder layers, 8k tokens, same weights and inputs. "
        "Mean relative error against an fp32 model with exact masked attention, as "
        "USP / single-GPU FA4 (the no-parallelism bf16 result). Parameter gradients "
        "of USP are summed over ranks in bf16, which adds about 2% on its own "
        "(see the Ulysses-only rows, whose attention is bitwise identical to FA4).",
        "",
        "| layout | ring x ulysses | mask path | combine | fp32 grads | output | input grad | param grads |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        check = r.get("check", "")
        if not check.startswith("usp_") or check.endswith("_vs_fa4"):
            continue
        ref = fa4[key(r)]
        cells = [f"{r[k]:.3e} / {ref[k]:.3e}" for k in ("out", "dx", "params")]
        out.append(
            f"| {r['layout']} | {r['ring']} x {r['ulysses']} | "
            f"{r.get('path', 'block-sparse')} | {check[4:]} | "
            f"{r['fp32_grads']} | " + " | ".join(cells) + " |"
        )
    return "\n".join(out)


def fast_path_table(native: list[dict], block_sparse: list[dict]) -> str:
    """Ulysses on plain masks: native FA4 mask vs the block-sparse path."""
    before = {r["label"]: r["median_ms"] for r in block_sparse if "error" not in r}
    out = [
        "One plain causal / full document, 8x H100, 32 heads; both columns measured "
        "in the same session. With a ring, the native path stripes a causal document "
        "over the ring ranks so every step is one causal or strictly causal FA4 call.",
        "",
        "| scenario | block-sparse mask ms | native mask ms | change |",
        "|---|---|---|---|",
    ]
    for r in native:
        if r["label"] in before and "error" not in r:
            b = before[r["label"]]
            out.append(
                f"| {r['label']} | {b:.1f} | {r['median_ms']:.1f} | "
                f"{(r['median_ms'] / b - 1) * 100:+.1f}% |"
            )
    return "\n".join(out)


def native_parity_table(rows: list[dict]) -> str:
    """Native-mask path vs block-sparse path accuracy (online combine, vs fp32)."""

    def errs(r: dict) -> str:
        return " / ".join(
            f"{r[k + '_mean_rel']:.2e}" for k in ("out", "dq", "dk", "dv")
        )

    pairs: dict[tuple, dict] = {}
    for r in rows:
        if r.get("check") != "usp" or r.get("reference") != "fp32":
            continue
        if r["combine"] != "online" or r.get("kv_heads") != r.get("heads"):
            continue
        if r.get("batch", 1) != 1 or not r.get("fp32_grads", True):
            continue
        k = (r["layout"], r["ring"], r["ulysses"])
        pairs.setdefault(k, {})["native" if r.get("native_mask") else "sparse"] = r
    out = [
        "Online combine, 16k tokens, mean relative error vs fp32 (out / dq / dk / dv).",
        "",
        "| layout | ring x ulysses | block-sparse path | native path |",
        "|---|---|---|---|",
    ]
    for (layout, ring, uly), pr in sorted(pairs.items()):
        if "native" in pr and "sparse" in pr:
            out.append(
                f"| {layout} | {ring} x {uly} | {errs(pr['sparse'])} | {errs(pr['native'])} |"
            )
    return "\n".join(out)


def model_table(rows: list[dict]) -> str:
    out = [
        "| config | step s | tokens/s/GPU | MFU | attention FLOP share | peak GiB |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['label']} | {r['error']} | | | | |")
            continue
        out.append(
            f"| {r['label']} | {r['step_ms_median'] / 1e3:.2f} | {r['tokens_per_s_per_gpu']:.0f} | "
            f"{r['mfu'] * 100:.1f}% | {r['attn_flop_share'] * 100:.0f}% | {r['peak_mem_gib']:.1f} |"
        )
    return "\n".join(out)


def online_kernel_table(rows: list[dict]) -> str:
    out = [
        "Single H100, forward only, 16k query tokens, KV split into chunks of 16k, 32 heads, "
        "dense attention. Time relative to one FA4 call over the whole KV; output error vs fp32.",
        "",
        "| chunks | one FA4 call ms | online | FA4 per chunk + flash combine | FA4 per chunk + PyTorch combine | error: one call / online / flash / PyTorch |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:

        def cell(name: str) -> str:
            return f"{r[name + '_ms']:.2f} ({r[name + '_vs_single']:.3f}x)"

        errs = " / ".join(
            f"{r[n + '_err']:.2e}"
            for n in ("fa4_single", "online", "fa4_flash_combine", "fa4_python_combine")
        )
        out.append(
            f"| {r['chunks']} | {r['fa4_single_ms']:.2f} | {cell('online')} | "
            f"{cell('fa4_flash_combine')} | {cell('fa4_python_combine')} | {errs} |"
        )
    return "\n".join(out)


def trace_table(rows: list[dict]) -> str:
    out = [
        "GPU time in one profiled forward + backward on rank 0 (ms). `nccl` counts NCCL kernels, "
        "which occupy SMs; `memcpy` is copy-engine traffic.",
        "",
        "| trace | attention | nccl | attention overlapping nccl | memcpy | other | GPU span |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in rows:
        out.append(
            f"| {t['trace']} | {t['attn_ms']:.1f} | {t['nccl_ms']:.1f} | {t['attn_overlapping_nccl_ms']:.1f} | "
            f"{t['memcpy_ms']:.1f} | {t['other_ms']:.1f} | {t['gpu_span_ms']:.1f} |"
        )
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("results_dir")
    args = p.parse_args()
    d = Path(args.results_dir)
    sections = []
    acc = _rows(d / "accuracy.log")
    if acc:
        sections.append(
            "## Accuracy vs fp32 (mean relative error)\n\n" + accuracy_table(acc)
        )
        sections.append(
            "## Deviation from single-GPU FA4 (mean relative error, longer sequences)\n\n"
            "The single-GPU column is empty here by construction.\n\n"
            + accuracy_table(acc, reference="fa4")
        )
    fp32g = _rows(d / "accuracy_fp32_grads.jsonl")
    if fp32g:
        lines = [
            "## fp32 gradient accumulation across ring steps\n",
            "Mixed layout, 32k tokens, 8 GPUs, mean relative error of the gradients vs fp32. "
            "Rows with `kv=4` use GQA (16 query / 4 KV heads); `kv=16` is MHA, where dk/dv "
            "stay per-step bf16.",
            "",
            "| kv heads | ring x ulysses | fp32 grads | dq | dk | dv |",
            "|---|---|---|---|---|---|",
        ]
        for r in fp32g:
            if r.get("check") == "usp" and r.get("reference") == "fp32":
                lines.append(
                    f"| {r['kv_heads']} | {r['ring']} x {r['ulysses']} | {r.get('fp32_grads')} | "
                    f"{r['dq_mean_rel']:.2e} | {r['dk_mean_rel']:.2e} | {r['dv_mean_rel']:.2e} |"
                )
        single = next((r for r in fp32g if r.get("check") == "fa4_single_gpu"), None)
        if single:
            lines.append(
                f"| single-GPU FA4 | - | - | {single['dq_mean_rel']:.2e} | "
                f"{single['dk_mean_rel']:.2e} | {single['dv_mean_rel']:.2e} |"
            )
        speed = _rows(d / "fp32_grads_speed.jsonl")
        if speed:
            lines += [
                "",
                "| 256k step, 8 GPUs | bf16 grads ms | fp32 grads ms |",
                "|---|---|---|",
            ]
            lines += [
                f"| {r['label']} | {r['bf16_grads_ms']:.1f} | {r['fp32_grads_ms']:.1f} |"
                for r in speed
            ]
        sections.append("\n".join(lines))
    abl = _rows(d / "ablation.jsonl")
    if abl:
        sections.append("## Ablation on 8 GPUs (32 MHA heads)\n" + bench_tables(abl))
    h8, h16 = _rows(d / "heads8.jsonl"), _rows(d / "heads16.jsonl")
    if h8 or h16:
        sections.append(
            "## Real head layouts, one node vs two nodes\n\n" + scaling_table(h8, h16)
        )
    for name in ("transport8", "transport16"):
        rows = _rows(d / f"{name}.jsonl")
        if rows:
            sections.append(f"## Transports ({name})\n\n" + transport_table(rows))
    tcp = _rows(d / "torchcp8.jsonl") + _rows(d / "torchcp8_match.jsonl")
    if tcp:
        sections.append(
            "## PyTorch-native context parallelism vs ours (8 GPUs)\n"
            + bench_tables(tcp)
        )
    external = [
        r
        for name in (
            "ablation",
            "torchcp8",
            "torchcp8_match",
            "te8",
            "yunchang8",
            "magi8",
        )
        for r in _rows(d / f"{name}.jsonl")
    ]
    if any("te_" in r["label"] or "yunchang" in r["label"] for r in external):
        sections.append(
            "## Ours vs other context-parallel attention libraries (8 GPUs)\n\n"
            + baseline_table(external)
        )
    checks = _rows(d / "baseline_checks.jsonl")
    if checks:
        sections.append(
            "## Correctness of the baselines (forward + backward)\n\n"
            + baseline_check_table(checks)
        )
    ring_acc = _rows(d / "ring8_8k_accuracy.jsonl")
    if ring_acc and checks:
        sections.append(
            "## Ring accuracy vs other libraries\n\n"
            + ring_accuracy_table(ring_acc, checks)
        )
    parity = _rows(d / "model_parity.jsonl")
    if parity:
        sections.append(
            "## Model-level parity: 7B multimodal MoT decoder\n\n"
            + model_parity_table(parity)
        )
    no_native = _rows(d / "torchcp8_match_no_native.jsonl")
    if no_native:
        sections.append(
            "## Native FA4 mask for plain layouts\n\n"
            + fast_path_table(_rows(d / "torchcp8_match.jsonl"), no_native)
        )
    native_check = _rows(d / "native_ring_check.jsonl")
    if native_check:
        sections.append(
            "## Native masks: accuracy vs the block-sparse path\n\n"
            + native_parity_table(native_check)
        )
    kernels = _rows(d / "kernel_fa3_fa4.jsonl")
    if kernels:
        sections.append("## FA3 vs FA4 kernels on one GPU\n\n" + kernel_table(kernels))
    conv = {p.stem: _rows(p) for p in sorted((d / "convergence").glob("*.jsonl"))}
    if conv.get("ulysses_ref"):
        sections.append(
            "## Convergence: ~200M LM, 64k packed tokens per step, 8 GPUs\n\n"
            + convergence_table(conv)
        )
    gqa = _rows(d / "gqa8.jsonl")
    if gqa:
        sections.append("## GQA head layouts on 8 GPUs\n" + bench_tables(gqa))
    attrib = _rows(d / "attrib16.jsonl")
    if attrib:
        sections.append(
            "## Ulysses-16 vs USP across two nodes (profiled pairs)\n\n"
            + transport_table(attrib)
        )
    traces = _rows(d / "trace_breakdown.jsonl")
    if traces:
        sections.append("## Where the GPU time goes\n\n" + trace_table(traces))
    kernel = _rows(d / "online_kernel.jsonl")
    if kernel:
        sections.append(
            "## Online FA4 kernel vs chunked FA4\n\n" + online_kernel_table(kernel)
        )
    for name, title in (("model8", "8 GPUs"), ("model16", "16 GPUs, 2 nodes")):
        model = _rows(d / f"{name}.jsonl")
        if model:
            sections.append(
                f"## 7B multimodal MoT decoder, 28 layers, {title}\n\n"
                + model_table(model)
            )
    coll = _rows(d / "collectives.jsonl")
    if coll:
        sections.append(
            "## Collectives (2 nodes)\n\n```\n"
            + "\n".join(json.dumps(r) for r in coll)
            + "\n```"
        )
    print("# usp-attention benchmark results\n\n" + "\n\n".join(sections))


if __name__ == "__main__":
    main()
