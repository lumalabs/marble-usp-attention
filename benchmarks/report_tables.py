# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""MLSYS_RESULTS.md, generated from the archived JSONL result rows and nothing else.

    python benchmarks/report_tables.py > benchmarks/MLSYS_RESULTS.md
    python benchmarks/report_tables.py path/to/results --extra 'path/to/more/*.jsonl'

With no arguments it reads the rows behind the paper's tables:
``results/rerun_oct28/mlsys`` plus the 2026-10-30 grid gap-fill
(``results/rerun_oct30_gapfill``) and the MagiAttention reruns
(``results/rerun_oct28_scaling/rerun_magi*``), i.e. ``PAPER_EXTRA``.
A results directory given on the command line is read on its own unless
``--extra`` is also given. ``--extra`` merges more JSONL files (paths or globs)
into the inputs without touching ``results/``; rows that carry a
``launch_command`` get the launcher's job id from its ``--master-addr``
appended to their source (``file:line@job``).

Every number in the headline and every table cell is computed here from the
raw rows; each "best" cell names the file and label of the row it came from,
so any number in the paper can be traced to one JSONL line.

Ours = the fastest *exact* configuration per case: rows that use the B300 /
H100 interval two-call path (``interval``), the bf16-merging dense ring
(``dense_ring`` without ``exact_dense_ring``) or FA3 gathers for head_dim 256
(bf16 dK/dV sums) are excluded. Baselines: FlashMask CP (public all-gather
path, plus its single-node Blackwell overlap path where it ran), FlashMask +
Ulysses (our hybrid construction) and MagiAttention.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import statistics
from collections import defaultdict
from collections.abc import Iterable

# Default input directory; ``main`` and the importing scripts (repro/tables.py,
# analysis/*.py) override it by assigning the module global before ``load()``.
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "rerun_oct28", "mlsys")
DEFAULT_RESULTS = RESULTS
# Rows merged into the default inputs to reproduce the paper's tables.
PAPER_EXTRA = (
    os.path.join(HERE, "results", "rerun_oct30_gapfill", "*.jsonl"),
    os.path.join(HERE, "results", "rerun_oct28_scaling", "rerun_magi*.jsonl"),
)
# Runs where FA3 was the default for dense gathers (inexact dK/dV for head_dim 256).
FA3_DEFAULT_FILES = {"g32g", "g32e2", "g64", "g128", "abl32"}
# Speedup band reported as "level" rather than a win or a loss.
LEVEL = (0.98, 1.02)
ALIASES = {
    "qwen38": "q38",
    "q38fa3": "q38",
    "qwen3vl30": "q3vl30",
    "qwen3vl": "q3vl30",
}
NAMES = {
    "q3vl30": "Qwen3-VL-30B-A3B",
    "q38": "Qwen3.8-27B",
    "molmo2": "Molmo2-8B",
    "gemma3": "Gemma 3 27B (global)",
    "gemma3_swa": "Gemma 3 27B (sliding window)",
    "bagel": "BAGEL-7B",
    "mha": "32-head MHA",
}

# Labels of the paper-gap sweeps (sweep.py ``_paper_sweeps``: ``_ours``,
# ``_fm``, ``_magi``, ``_e2e``, ``_e2e_fm``) name one config each:
#   gemma3_mixed_4096k_32g_r4u8_gather_chunks[_dense_ring_on...]   ours
#   gemma3_mixed_1024k_fm[_u16]                                    FlashMask CP / FM+U
#   magi_q3vl30_video_4096k_128g_c4096                             MagiAttention
#   e2e_gemma3_mixed_2048k_32g_r4u8_gather                         ours, end to end
#   e2e_fm_bagel_video_4096k_32g / e2e_fmu_bagel_video_4096k_32g_u4
# The older labels (a bare model name, "fm_<model>", "magi_<model>",
# "abl_<model>", "e2e_<model>", "bagel_dring_k8", ...) never contain the
# "<model>_<mix>_<tokens>k" stem, so they fall through to the older rules.
SWEEP_LABEL = re.compile(
    r"^(?P<prefix>e2e_fmu_|e2e_fm_|e2e_|magi_)?"
    r"(?P<model>gemma3_swa|gemma3|molmo2|bagel|q3vl30|q38|mha)"
    r"_(?P<mix>mixed|video|mix)_(?P<tokens>\d+)k"
    r"(?:_(?P<world>\d+)g)?"
    r"(?P<fm>_fm)?"
    r"(?:_r(?P<ring>\d+)u(?P<ulysses>\d+))?"
    r"(?:_u(?P<fm_ulysses>\d+))?"
    r"(?:_(?P<exchange>ring|gather))?"
    r"(?:_(?P<balance>ptrr|chunks))?"
    r"(?:_c(?P<chunk>\d+))?"
    r"(?P<extra>(?:_[a-z0-9]+)*)$"
)
SWEEP_KINDS = {
    "e2e_fmu_": "e2e_fmu",
    "e2e_fm_": "e2e_fm",
    "e2e_": "e2e",
    "magi_": "magi",
}


def launch_command(r: dict) -> list[str]:
    """The recorded launch command as argv, or ``[]`` when the row has none."""
    cmd = r.get("launch_command") or []
    return shlex.split(cmd) if isinstance(cmd, str) else list(cmd)


def launch_flag(r: dict, flag: str, *, first: bool = False) -> str | None:
    """Value of ``flag`` in the launch command.

    Args:
        r: A JSONL row.
        flag: The option, e.g. ``"--layout"``.
        first: Return the first occurrence instead of the last (argparse keeps
            the last; the sweep's unique name is the first ``--label``).

    Returns:
        The value, or None when the flag is absent.
    """
    argv = launch_command(r)
    hits = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]
    if not hits:
        return None
    return hits[0] if first else hits[-1]


def job_id(r: dict) -> str | None:
    """Launcher job id from the launch command's ``--master-addr`` pod name.

    ``user-marble-oom32-bagel-c-k33ae-n0-0-worker-0`` ->
    ``user-marble-oom32-bagel-c-k33ae``.
    """
    addr = launch_flag(r, "--master-addr")
    if addr is None:
        return None
    return re.sub(r"-n\d+-\d+-worker-\d+$", "", addr)


def sweep_match(r: dict) -> re.Match | None:
    """``SWEEP_LABEL`` match of the row's label, else of its launch ``--label``.

    Rows of the first sweep runs carry only the model name as ``label`` (the
    driver kept the last of two ``--label`` options); the unique name is the
    first ``--label`` of the launch command.
    """
    for lab in (r.get("label") or "", launch_flag(r, "--label", first=True) or ""):
        m = SWEEP_LABEL.match(lab)
        if m is not None:
            return m
    return None


def sweep_name(r: dict) -> str:
    """The unique per-config sweep label when there is one, else the label."""
    m = sweep_match(r)
    return m.string if m is not None else (r.get("label") or "")


def is_oom(r: dict) -> bool:
    """True for a failed run whose recorded output tail is a CUDA OOM."""
    if not r.get("error"):
        return False
    text = f"{r.get('error', '')}\n{r.get('tail') or ''}"
    return "OutOfMemory" in text or "out of memory" in text


def world(r: dict) -> int | None:
    """GPUs of the run: the row's ``world``, else the label, else the launcher."""
    if r.get("world") is not None:
        return int(r["world"])
    m = sweep_match(r)
    if m is not None and m["world"]:
        return int(m["world"])
    nnodes, nproc = launch_flag(r, "--nnodes"), launch_flag(r, "--nproc-per-node")
    if nnodes and nproc:
        return int(nnodes) * int(nproc)
    return None


HEADER = """
## What is compared

| System | What it does | Notes |
|---|---|---|
| **Ours** | Ulysses (heads, inside a node) x a mask-aware sequence dimension: ring (PTRR + online fp32 FA4), gather (only the KV the mask needs; block-sparse or dense calls), halo (sliding windows), dense ring (long generation, per-step balanced chunks) | Exact mode only: out / dq / dk / dv at single-GPU FA4 accuracy |
| FlashMask CP | FlashMask-4 (PyTorch release) balanced all-gather context parallelism with its own load balancer | Concurrent work (FlashMask-3). Its overlap path runs only on one Blackwell node |
| FM + Ulysses | Our construction: Ulysses inside the node, FlashMask CP across nodes | Strongest FlashMask-based baseline; not shipped by FlashMask |
| MagiAttention | Dispatch solver + group-collective communication | Default bf16 reduce (1.22x dK/dV error) |

Models (q / kv heads, head dim): Qwen3-VL-30B-A3B 32/4/128, Qwen3.8-27B 24/4/256,
Molmo2-8B 32/8/128 (vision groups), Gemma 3 27B 32/16/128 (1024-token sliding window
on 5 of 6 layers), BAGEL-7B 28/4/128. Layouts are real packs built from each model's
tokenizer rules (`realpack/`), 1M-8M tokens. Single-node results (synthetic masks,
accuracy, library comparisons) are in `RESULTS.md`.
"""

CAVEATS = """
## Caveats

- FlashMask's public H100 context parallelism does not overlap communication; the
  FM + Ulysses hybrid is our construction. FlashMask-3's Hopper kernel ships only in
  PaddlePaddle.
- Peak memory is attention-only in kernel benchmarks; gather keeps the gathered KV for
  the backward unless `gather_regather=True`.
- NCCL: every 32-128 GPU run (ours, FlashMask, MagiAttention) unsets the image's
  `LD_PRELOAD` and uses torch's bundled NCCL 2.28.9 with send/recv transport and
  `NCCL_NET_PLUGIN=none`. The one-sided nccl4py transport results in `RESULTS.md`
  ran on the image's NCCL 2.29.7 (`/opt/nccl`), which its RMA calls require.
- Runs are 3-5 timed iterations after warm-up; repeated runs agree within 1-2%.
  `bagel_fm64b` and `bagel64e` re-measure the 8M baselines and dense ring in one session.
"""


def load_rows(path: str) -> list[dict]:
    """Every JSON row of one file, with ``_file``, ``_line`` and ``_source``.

    ``_source`` is the launcher's job id (``job_id``) or None; error
    rows (no ``median_ms``) are included, so callers can report OOMs.
    """
    name = os.path.basename(path).removesuffix(".jsonl")
    rows = []
    for i, line in enumerate(open(path)):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        r["_file"], r["_line"], r["_source"] = name, i + 1, job_id(r)
        rows.append(r)
    return rows


def extra_files(patterns: Iterable[str]) -> list[str]:
    """Expand ``--extra`` paths or globs in order, each file once."""
    out: list[str] = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or [pat]
        out += [h for h in hits if h not in out]
    return out


def load(results: str | None = None, extra: Iterable[str] = ()) -> list[dict]:
    """Measured rows of ``results``/*.jsonl plus the ``extra`` files, in that order.

    Args:
        results: Directory of JSONL files; the module's ``RESULTS`` by default.
        extra: More JSONL paths or globs (e.g. rows from another cluster),
            merged after the directory's rows without copying them into it.

    Returns:
        Rows with a timing (attention rows with ``median_ms`` and one data
        replica; model rows with ``step_ms_median``).
    """
    files = sorted(glob.glob(os.path.join(results or RESULTS, "*.jsonl")))
    files += extra_files(extra)
    rows = []
    for f in files:
        for r in load_rows(f):
            if r.get("bench") == "model":
                if "step_ms_median" not in r:
                    continue
            elif "median_ms" not in r or r.get("dp_replicas", 1) != 1:
                continue
            rows.append(r)
    return rows


def model(r: dict) -> str:
    """Canonical model key (``NAMES``) of a row.

    Paper-gap sweep labels (``SWEEP_LABEL``) name the model outright; the
    q3vl30 rows run on the ``qwen38_*`` packs, so the label wins over the
    layout path. Other labels keep the older prefix rules.
    """
    m = sweep_match(r)
    if m is not None:
        return ALIASES.get(m["model"], m["model"])
    lab = (r.get("label") or "").removeprefix("fm_").removeprefix("abl_")
    lab = lab.removeprefix("magi_").removesuffix("_fa4")
    if lab.startswith("bagel"):
        return "bagel"
    return ALIASES.get(lab, lab)


def layout(r: dict) -> str:
    """Pack name (``gemma3_mixed_4096k``) from the row, else from its launcher.

    Error rows record no ``layout``; the launch command's ``--layout`` has it.
    """
    lay = r.get("layout") or launch_flag(r, "--layout") or ""
    return lay.split("/")[-1].removesuffix("#0").removesuffix(".json")


def kind(r: dict) -> str | None:
    """System of a row: ours / ours_inexact / fm / fm_hybrid / fm_overlap / magi,
    e2e / e2e_fm / e2e_fmu for full-model rows, None for rows no table uses.

    ``impl`` (set by the FlashMask and Magi drivers) decides first; for rows
    without it (ours, and error rows that recorded only the label) the
    sweep label's ``_fm[_uN]`` / ``magi_`` / ``e2e_`` parts decide; the full
    model rows fall back to ``attn`` and ``ulysses``.
    """
    impl, lab = r.get("impl", ""), r.get("label") or ""
    m = sweep_match(r)
    if impl.startswith("flashmask"):
        if "overlap" in impl:
            return "fm_overlap"
        return "fm_hybrid" if "ulysses" in impl else "fm"
    if impl.startswith("magi") or lab.startswith("magi"):
        return None if lab.startswith("magi_hp") else "magi"
    if r.get("bench") == "model":
        if m is not None and m["prefix"] in SWEEP_KINDS:
            return SWEEP_KINDS[m["prefix"]]
        if lab.startswith("e2e_fmu_"):
            return "e2e_fmu"
        if lab.startswith("e2e_fm_"):
            return "e2e_fm"
        if lab.startswith("e2e_") or r.get("attn") == "usp":
            return "e2e"
        if r.get("attn") == "flashmask":
            return "e2e_fmu" if (r.get("ulysses") or 1) > 1 else "e2e_fm"
        return None
    if m is not None and m["fm"]:
        # An error row of the FlashMask ladders: no impl was recorded.
        return "fm_hybrid" if m["fm_ulysses"] else "fm"
    if impl or lab in ("flex", "ours", "cornstarch"):
        return None
    # From the plan the row ran: the interval two-call path and the dense ring
    # merge bf16 partials. (Before 2026-10-16, bench_attention recorded
    # --exact-dense-ring without applying it, so the flag says nothing.)
    exact = not r.get("interval") and not r.get("dense_ring")
    if r.get("head_dim") == 256 and r.get("kv_exchange") == "gather":
        if r.get("fa3_gather") or (
            r["_file"] in FA3_DEFAULT_FILES and "fa4" not in lab
        ):
            exact = False
    if "fa3" in lab:
        exact = False
    return "ours" if exact else "ours_inexact"


def cfg(r: dict) -> str:
    s = f"R{r['ring']}U{r['ulysses']}"
    if r.get("sliding_window") and r.get("kv_exchange") == "gather":
        s += " halo"
    elif r.get("kv_exchange") == "gather":
        s += " gather"
    elif r["ring"] > 1:
        s += " dense ring" if "dring" in (r.get("label") or "") else " ring"
    return s


def src(r: dict | None) -> str:
    """``file:line`` of a row, plus ``@job`` when the row recorded its launch."""
    if r is None:
        return ""
    s = f"{r['_file']}:{r['_line']}"
    return f"{s}@{r['_source']}" if r.get("_source") else s


def best(rs: list[dict]) -> dict | None:
    return min(rs, key=lambda r: r["median_ms"]) if rs else None


def ratio(a: dict | None, b: dict | None) -> float | None:
    return None if a is None or b is None else b["median_ms"] / a["median_ms"]


def ms(r: dict | None, missing: str = "–") -> str:
    return missing if r is None else f"{r['median_ms']:.0f}"


def fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}×"


def cases_for(rows: list[dict]) -> dict:
    cases: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = kind(r)
        if k is not None:
            cases[(model(r), layout(r), r["world"], r.get("sliding_window") or 0)][
                k
            ].append(r)
    return cases


def scale_section(rows: list[dict]) -> list[str]:
    cases = cases_for([r for r in rows if not r["_file"].startswith("b300")])
    out, table = [], []
    vs_fmu, vs_cp, vs_magi = [], [], []
    for key in sorted(cases, key=lambda k: (k[0], k[2], k[1])):
        c = cases[key]
        o = best(c["ours"])
        fm, fh, mg = best(c["fm"]), best(c["fm_hybrid"]), best(c["magi"])
        if o is None or (fm is None and fh is None and mg is None):
            continue
        m, lay, w, win = key
        if w < 32:
            continue
        rh, rc, rm = ratio(o, fh), ratio(o, fm), ratio(o, mg)
        if rh is not None:
            vs_fmu.append((rh, w, key))
        if rc is not None:
            vs_cp.append((rc, w, key))
        if rm is not None:
            vs_magi.append((rm, w, key))
        table.append(
            f"| {NAMES.get(m, m)} | {lay} | {w} | {o['median_ms']:.0f} | {cfg(o)} | "
            f"{o['peak_mem_gib']:.1f} | {src(o)} | {ms(fm)} | "
            f"{ms(fh)} | "
            f"{ms(mg)} | "
            f"{fmt(rc)} | {fmt(rh)} | {fmt(rm)} | {src(fh)} |"
        )

    def counts(v):
        win = [x for x, _, _ in v if x >= LEVEL[1]]
        level = [x for x, _, _ in v if LEVEL[0] <= x < LEVEL[1]]
        loss = [x for x, _, _ in v if x < LEVEL[0]]
        return win, level, loss

    w_, l_, s_ = counts(vs_fmu)
    out.append("## Headline (32-128 H100s, generated)\n")
    out.append(
        f"- vs FM + Ulysses: {len(vs_fmu)} cases; faster in {len(w_)}"
        f" ({min(w_):.2f}-{max(w_):.1f}×), level in {len(l_)}"
        f" ({', '.join(f'{x:.2f}×' for x in l_)}), slower in {len(s_)}"
        + (f" ({', '.join(f'{x:.2f}×' for x in s_)})" if s_ else "")
        + "."
    )
    for w in sorted({w for _, w, _ in vs_fmu}):
        xs = [x for x, ww, _ in vs_fmu if ww == w]
        out.append(
            f"  - {w} GPUs: {len(xs)} cases, median {statistics.median(xs):.2f}×"
        )
    w_, l_, s_ = counts(vs_cp)
    out.append(
        f"- vs FlashMask CP: {len(vs_cp)} cases with a result; faster in {len(w_)}"
        f" ({min(w_):.2f}-{max(w_):.1f}×), level in {len(l_)}, slower in {len(s_)}."
    )
    if vs_magi:
        w_, l_, s_ = counts(vs_magi)
        gpus = sorted({w for _, w, _ in vs_magi})
        out.append(
            f"- vs MagiAttention: {len(vs_magi)} cases at {'/'.join(map(str, gpus))} GPUs;"
            f" faster in {len(w_)} ({min(w_):.2f}-{max(w_):.1f}×), level in {len(l_)},"
            f" slower in {len(s_)}"
            + (f" ({', '.join(f'{x:.2f}×' for x in s_)})" if s_ else "")
            + "."
        )
    out.append(
        f"\nLevel = speedup in [{LEVEL[0]}, {LEVEL[1]}). Ours = fastest exact configuration.\n"
    )
    out.append("## Every case at 32-128 H100s\n")
    out.append(
        "| model | layout | GPUs | ours ms | config | mem GiB | ours source | FlashMask CP |"
        " FM+Ulysses | MagiAttention | vs CP | vs FM+U | vs Magi | FM+U source |"
    )
    out.append("|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    out += table
    return out


def video_section(rows: list[dict]) -> list[str]:
    out = ["\n## BAGEL long video (dense ring)\n"]
    out.append(
        "| tokens | GPUs | label | config | chunks | refined | ms | mem GiB | source |"
    )
    out.append("|---:|---:|---|---|---:|---|---:|---:|---|")
    sel = [
        r
        for r in rows
        if model(r) == "bagel"
        and "video" in layout(r)
        and not r["_file"].startswith("b300")
        and kind(r) in ("ours", "fm", "fm_hybrid")
        and r["world"] >= 32
    ]
    for r in sorted(sel, key=lambda r: (r["seq_len"], r["world"], r["median_ms"])):
        out.append(
            f"| {r['seq_len'] // 1024}k | {r['world']} | {r.get('label')} | "
            f"{cfg(r) if kind(r) == 'ours' else r.get('impl')} | "
            f"{r.get('dense_ring_chunks') or ''} | {r.get('dense_ring_refine', '')} | "
            f"{r['median_ms']:.0f} | {r['peak_mem_gib']:.1f} | {src(r)} |"
        )
    return out


def b300_section(rows: list[dict]) -> list[str]:
    cases = cases_for([r for r in rows if r["_file"].startswith("b300")])
    out = ["\n## 8x B300, real 1M-token packs (exact paths only)\n"]
    out.append(
        "| model | layout | ours ms | config | source | FlashMask best ms | which |"
        " source | ours vs FlashMask | fastest inexact ours |"
    )
    out.append("|---|---|---:|---|---|---:|---|---|---:|---:|")
    for key in sorted(cases):
        c = cases[key]
        o = best(c["ours"])
        f = best(c["fm"] + c["fm_hybrid"] + c["fm_overlap"])
        if o is None or f is None:
            continue
        which = f.get("impl", "").removeprefix("flashmask_cp_")
        ix = best(c["ours_inexact"])
        out.append(
            f"| {NAMES.get(key[0], key[0])} | {key[1]} | {o['median_ms']:.0f} | {cfg(o)} | "
            f"{src(o)} | {f['median_ms']:.0f} | {which} | {src(f)} | {fmt(ratio(o, f))} | "
            f"{ms(ix, '')} |"
        )
    return out


B300_PATH_FILES = (
    "b300_eval6",
    "b300_eval7",
    "b300_eval8",
    "b300_eval9",
    "b300_eval10",
)


def _b300_path(r: dict) -> str:
    if "magi" in (r.get("label") or ""):
        return "MagiAttention (tuned)"
    if r.get("dp_replicas", 1) != 1:
        return f"Ulysses-{r['ulysses']} + DP{r['dp_replicas']}"
    if r.get("generation_ring"):
        return "exact generation ring" if r["ring"] > 1 else "exact, one rank"
    if r.get("interval"):
        return "interval (bf16 merges)"
    if r.get("dense_ring"):
        return "dense ring (bf16 merges)"
    return "block-sparse"


def b300_paths_section() -> list[str]:
    """Exact paths against the bf16-merging defaults, same session per file."""
    out = [
        "\n## 8x B300, exact paths vs the bf16-merging defaults (synthetic layouts)\n",
        "Rows from one file share a session. eval6 predates the bench fix that"
        " applies `--exact-dense-ring`, so its path column comes from the plan.\n",
        "| layout | tokens | heads q/kv | mesh | path | ms | source |",
        "|---|---:|---|---|---|---:|---|",
    ]
    rows = []
    for name in B300_PATH_FILES:
        f = os.path.join(RESULTS, f"{name}.jsonl")
        if not os.path.exists(f):
            continue
        for i, line in enumerate(open(f)):
            r = json.loads(line)
            if r.get("bench") == "model" or "median_ms" not in r:
                continue
            r["_file"], r["_line"] = name, i + 1
            rows.append(r)
    for r in sorted(
        rows,
        key=lambda r: (
            r["layout"],
            r["seq_len"],
            r.get("kv_heads") or 0,
            r["_file"],
            r["median_ms"],
        ),
    ):
        out.append(
            f"| {layout(r)} | {r['seq_len'] // 1024}k | {r['heads']}/{r.get('kv_heads') or r['heads']} | "
            f"R{r['ring']}U{r['ulysses']} | {_b300_path(r)} | {r['median_ms']:.1f} | {src(r)} |"
        )
    return out


def b300_accuracy_section() -> list[str]:
    """Error against an fp32 reference, relative to one GPU's FA4 error."""
    out = [
        "\n## 8x B300 accuracy, 32k tokens, 32/8 heads (error vs fp32 / one-GPU FA4 error)\n",
        "| args | path | out | dQ | dK | dV | source |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name in ("b300_acc4", "b300_acc5"):
        f = os.path.join(RESULTS, f"{name}.jsonl")
        if not os.path.exists(f):
            continue
        rows = [json.loads(line) for line in open(f)]
        for i, r in enumerate(rows):
            if r.get("check") != "usp" or r.get("reference") != "fp32":
                continue
            single = next(
                x
                for x in rows[i:]
                if x.get("check") == "fa4_single_gpu" and x["args"] == r["args"]
            )
            path = _b300_path({**r, "median_ms": 0})
            cells = " | ".join(
                f"{r[f'{t}_mean_rel'] / single[f'{t}_mean_rel']:.2f}x"
                for t in ("out", "dq", "dk", "dv")
            )
            args = r["args"].replace("file:layouts/", "")
            out.append(f"| `{args}` | {path} | {cells} | {name}:{i + 1} |")
    return out


def ablation_section(rows: list[dict]) -> list[str]:
    out = [
        "\n## Exchange x token order (32 GPUs, 2M video layouts, one session: abl32)\n"
    ]
    out.append(
        "| model | ring + PTRR | ring + chunks | gather + PTRR | gather + chunks |"
    )
    out.append("|---|---:|---:|---:|---:|")
    cells: dict = defaultdict(list)
    for r in rows:
        if r["_file"] != "abl32" or "video_2048k" not in layout(r) or kind(r) != "ours":
            continue
        if "dring" in (r.get("label") or "") or r.get("sliding_window"):
            continue
        ex = "gather" if r.get("kv_exchange") == "gather" else "ring"
        cells[(model(r), ex, r.get("balance", "ptrr"))].append(r)
    for m in sorted({k[0] for k in cells}):
        row = [
            best(cells[(m, ex, bal)])
            for ex in ("ring", "gather")
            for bal in ("ptrr", "chunks")
        ]
        out.append(
            f"| {NAMES.get(m, m)} | "
            + " | ".join(
                "–" if r is None else f"{r['median_ms']:.0f} ({src(r)})" for r in row
            )
            + " |"
        )
    return out


def e2e_section(rows: list[dict]) -> list[str]:
    out = ["\n## End to end (full model forward + backward, every run)\n"]
    out.append(
        "| label | layout | GPUs | attention | mesh | step s | MFU | mem GiB | source |"
    )
    out.append("|---|---|---:|---|---|---:|---:|---:|---|")
    sel = [r for r in rows if r.get("bench") == "model"]
    for r in sorted(
        sel,
        key=lambda r: (
            r.get("label") or "",
            r["seq_len"],
            r["world"],
            r["step_ms_median"],
        ),
    ):
        mesh = f"R{r['ring']}U{r['ulysses']} {r.get('kv_exchange', '')}".strip()
        mfu = r.get("mfu")
        out.append(
            f"| {r.get('label')} | {layout(r)} | {r['world']} | {r.get('attn')} | {mesh} | "
            f"{r['step_ms_median'] / 1e3:.2f} | {'' if mfu is None else f'{100 * mfu:.0f}%'} | "
            f"{r['peak_mem_gib']:.0f} | {src(r)} |"
        )
    return out


def _headline_cells(rows: list[dict]) -> list[tuple[tuple, dict]]:
    """The cells of the "Every case" table, in its order (same selection rule)."""
    cases = cases_for([r for r in rows if not r["_file"].startswith("b300")])
    out = []
    for key in sorted(cases, key=lambda k: (k[0], k[2], k[1])):
        c = cases[key]
        o = best(c["ours"])
        if o is None or key[2] < 32:
            continue
        if all(best(c[k]) is None for k in ("fm", "fm_hybrid", "magi")):
            continue
        out.append((key, c))
    return out


def _family(lay: str) -> tuple[str, int]:
    """'gemma3_mixed_4096k' -> ('gemma3_mixed', 4096)."""
    stem, _, tok = lay.rpartition("_")
    return stem, int(tok.removesuffix("k"))


def gib(r: dict | None) -> str:
    return "–" if r is None else f"{r['peak_mem_gib']:.1f}"


def memory_section(rows: list[dict]) -> list[str]:
    """Peak attention memory per GPU for every cell of the table above."""
    out = ["\n## Peak attention memory per GPU, every case (GiB, same rows as above)\n"]
    out.append(
        "`torch.cuda.max_memory_allocated` around the attention call, attention-only"
        " (inputs, outputs and workspace of one layer). Each value is the row that"
        " gives the ms in the table above; baselines with a result but no memory"
        " column there get a source here. OOM attempts leave no row (`–`).\n"
    )
    out.append(
        "| model | layout | GPUs | ours GiB | config | FlashMask CP | FM+Ulysses |"
        " FM+U mesh | MagiAttention | FM+U / ours | CP / ours | CP source | Magi source |"
    )
    out.append("|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---|---|")
    for (m, lay, w, _), c in _headline_cells(rows):
        o = best(c["ours"])
        fm, fh, mg = best(c["fm"]), best(c["fm_hybrid"]), best(c["magi"])

        def rel(b):
            return "–" if b is None else f"{b['peak_mem_gib'] / o['peak_mem_gib']:.2f}×"

        mesh = "" if fh is None else f"R{fh['ring']}U{fh['ulysses']}"
        out.append(
            f"| {NAMES.get(m, m)} | {lay} | {w} | {gib(o)} | {cfg(o)} | {gib(fm)} | "
            f"{gib(fh)} | {mesh} | "
            f"{gib(mg)} | {rel(fh)} | {rel(fm)} | {src(fm)} | {src(mg)} |"
        )
    return out


SCALING_SYSTEMS = (
    ("ours", "ours"),
    ("fm_hybrid", "FM+U"),
    ("fm", "FlashMask CP"),
    ("magi", "Magi"),
)


def _by_series(rows: list[dict]) -> dict:
    """(model, layout family, window) -> {system: {(tokens k, GPUs): best row}}."""
    series: dict = defaultdict(lambda: defaultdict(dict))
    for (m, lay, w, win), c in _headline_cells(rows):
        fam, tok = _family(lay)
        for k, s in SCALING_SYSTEMS:
            r = best(c[k])
            if r is not None:
                series[(m, fam, win)][s][(tok, w)] = r
    return series


def strong_scaling_section(rows: list[dict]) -> list[str]:
    """Same layout, GPUs doubled: T(P) / T(2P), ideal 2.00×."""
    out = ["\n## Strong scaling (same layout, GPUs doubled; ideal 2.00×)\n"]
    out.append(
        "Ratio = ms at P / ms at 2P for each system's fastest row at both counts;"
        " efficiency = ratio / 2. Medians per system are over the layouts with a"
        " result at both counts.\n"
    )
    out.append(
        "| model | layout | tokens | GPUs | "
        + " | ".join(f"{s} ms (ratio)" for _, s in SCALING_SYSTEMS)
        + " |"
    )
    out.append("|---|---|---:|---|" + "---:|" * len(SCALING_SYSTEMS))
    med: dict = defaultdict(lambda: defaultdict(list))
    series = _by_series(rows)
    for (m, fam, _), by_sys in sorted(series.items()):
        pairs = sorted(
            {(tok, p) for d in by_sys.values() for (tok, p) in d if (tok, 2 * p) in d}
        )
        for tok, p in pairs:
            cells = []
            for _, s in SCALING_SYSTEMS:
                d = by_sys.get(s, {})
                a, b = d.get((tok, p)), d.get((tok, 2 * p))
                if a is None or b is None:
                    cells.append(
                        "–" if a is None and b is None else f"{ms(a)} → {ms(b)}"
                    )
                    continue
                x = a["median_ms"] / b["median_ms"]
                med[(tok, p)][s].append(x)
                cells.append(f"{ms(a)} → {ms(b)} ({x:.2f}×)")
            out.append(
                f"| {NAMES.get(m, m)} | {fam} | {tok}k | {p}→{2 * p} | "
                + " | ".join(cells)
                + " |"
            )
    for (tok, p), by_sys in sorted(med.items()):
        out.append(
            f"\n- {tok}k tokens, {p}→{2 * p} GPUs, median ratio (n): "
            + "; ".join(
                f"{s} {statistics.median(v):.2f}× ({len(v)})" for s, v in by_sys.items()
            )
        )
    return out


def weak_scaling_section(rows: list[dict]) -> list[str]:
    """Tokens per GPU fixed, GPUs doubled: T(2N, 2P) / T(N, P), ideal 1.00×."""
    out = ["\n## Weak scaling (tokens per GPU fixed; ideal 1.00×)\n"]
    out.append(
        "Each row is one system on one layout family at a fixed tokens-per-GPU;"
        " ratio = ms at the largest count / ms at the smallest count with a result."
        " On dense masks the work per GPU grows with N, so 1.00× is not reachable;"
        " the comparison between systems at one row is what matters.\n"
    )
    out.append(
        "| model | layout | tokens/GPU | system | 32 GPUs | 64 GPUs | 128 GPUs | ratio |"
    )
    out.append("|---|---|---:|---|---:|---:|---:|---:|")
    med: dict = defaultdict(lambda: defaultdict(list))
    for (m, fam, _), by_sys in sorted(_by_series(rows).items()):
        per_gpu = sorted({tok * 1024 // p for d in by_sys.values() for (tok, p) in d})
        for tpg in per_gpu:
            for _, s in SCALING_SYSTEMS:
                d = by_sys.get(s, {})
                pts = {p: r for (tok, p), r in d.items() if tok * 1024 // p == tpg}
                if len(pts) < 2:
                    continue
                lo, hi = min(pts), max(pts)
                x = pts[hi]["median_ms"] / pts[lo]["median_ms"]
                med[tpg][s].append(x)
                out.append(
                    f"| {NAMES.get(m, m)} | {fam} | {tpg // 1024}k | {s} | "
                    + " | ".join(ms(pts.get(p)) for p in (32, 64, 128))
                    + f" | {x:.2f}× |"
                )
    for tpg, by_sys in sorted(med.items()):
        out.append(
            f"\n- {tpg // 1024}k tokens/GPU, median ratio (n): "
            + "; ".join(
                f"{s} {statistics.median(v):.2f}× ({len(v)})" for s, v in by_sys.items()
            )
        )
    return out


def b300_all_section(rows: list[dict]) -> list[str]:
    """Every real-pack layout in the B300 rows, with memory; dashes where a side is missing."""
    cases = cases_for(
        [
            r
            for r in rows
            if r["_file"].startswith("b300") and layout(r).endswith("_1024k")
        ]
    )
    out = ["\n## 8x B300, real 1M-token packs: every layout in the rows, with memory\n"]
    out.append(
        "Same cells as the table above plus the layouts it omits because one side has"
        " no exact row. `–` = no row. Memory = `peak_mem_gib` of the fastest row.\n"
    )
    out.append(
        "| model | layout | ours ms | ours GiB | config | source | FlashMask CP ms | GiB |"
        " FM+Ulysses ms | GiB | FM overlap ms | GiB | ours vs best FlashMask |"
        " inexact ours ms | GiB |"
    )
    out.append(
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for key in sorted(cases):
        c = cases[key]
        o, fm, fh, fo = (best(c[k]) for k in ("ours", "fm", "fm_hybrid", "fm_overlap"))
        f = best(c["fm"] + c["fm_hybrid"] + c["fm_overlap"])
        ix = best(c["ours_inexact"])
        if o is None and f is None and ix is None:
            continue
        out.append(
            f"| {NAMES.get(key[0], key[0])} | {key[1]} | {ms(o)} | {gib(o)} | "
            f"{'' if o is None else cfg(o)} | {src(o)} | {ms(fm)} | {gib(fm)} | "
            f"{ms(fh)} | {gib(fh)} | {ms(fo)} | {gib(fo)} | {fmt(ratio(o, f))} | "
            f"{ms(ix)} | {gib(ix)} |"
        )
    return out


def _shown(path: str) -> str:
    """``path`` relative to this directory when inside it, for generated headers."""
    rel = os.path.relpath(os.path.abspath(path), HERE)
    return path if rel.startswith("..") else rel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """``[results_dir] [--extra PATH_OR_GLOB ...]``; see the module docstring."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", nargs="?", default=RESULTS, help="JSONL directory")
    p.add_argument(
        "--extra",
        nargs="*",
        default=None,
        metavar="JSONL",
        help="more JSONL files or globs merged after the directory's rows"
        " (default: PAPER_EXTRA when no results directory is given)",
    )
    return p.parse_args(argv)


def main() -> None:
    global RESULTS
    args = parse_args()
    RESULTS = args.results
    extra = args.extra
    if extra is None:
        default = os.path.abspath(RESULTS) == os.path.abspath(DEFAULT_RESULTS)
        extra = list(PAPER_EXTRA) if default else []
    rows = load(extra=extra)
    merged = sorted({r["_source"] for r in rows if r.get("_source")})
    lines = [
        "# Mask-aware USP: evaluation results (generated)\n",
        f"Generated by `benchmarks/report_tables.py` from `{_shown(RESULTS)}/*.jsonl`;"
        " do not edit by hand. Attention forward + backward per step (median ms,"
        " slowest rank), bf16, H100 80GB nodes of 8 GPUs unless marked B300."
        " `source` is `file:line` of the JSONL row.\n",
    ]
    if extra:
        lines.append(
            f"Merged `--extra` inputs: {', '.join(f'`{_shown(f)}`' for f in extra)};"
            " their sources carry the launching job id as `file:line@job`"
            f" (jobs: {', '.join(merged) or 'none recorded'}).\n"
        )
    lines.append(HEADER)
    attn = [r for r in rows if r.get("bench") != "model"]
    lines += scale_section(attn)
    lines += memory_section(attn)
    lines += strong_scaling_section(attn)
    lines += weak_scaling_section(attn)
    lines += ablation_section(attn)
    lines += video_section(attn)
    lines += b300_section(attn)
    lines += b300_all_section(attn)
    lines += b300_paths_section()
    lines += b300_accuracy_section()
    lines += e2e_section(rows)
    lines.append(CAVEATS)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
