# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Run a named benchmark sweep: one torchrun per config, results appended to JSONL.

Usage on one 8-GPU node::

    python benchmarks/sweep.py ablation --out results/ablation.jsonl

and on two nodes (run on both, same arguments except ``--node-rank``)::

    python benchmarks/sweep.py multinode16 --out results/multinode16.jsonl \
        --nnodes 2 --node-rank 0 --master-addr <node-0 IP>

Each config runs ``bench_attention.py`` in a fresh process group. Failures and
timeouts are recorded as rows with ``"error"`` so a sweep never stops early.
On single-node sweeps, configs already present in the output file (same label)
are skipped, so an interrupted sweep resumes by rerunning the same command.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from provenance import result_json

HERE = Path(__file__).resolve().parent


def cp(
    ring: int, ulysses: int, combine: str, *, ptrr: bool = True, transport: str = "nccl"
) -> list[str]:
    """Build attention-benchmark flags for one parallel configuration.

    Args:
        ring: Number of ranks in the ring dimension.
        ulysses: Number of ranks in the Ulysses dimension.
        combine: Ring partial-output combine mode.
        ptrr: Whether to keep load balancing enabled.
        transport: Ring communication transport name.

    Returns:
        A list of benchmark command-line arguments.
    """
    flags = [
        "--ring",
        str(ring),
        "--ulysses",
        str(ulysses),
        "--combine",
        combine,
        "--transport",
        transport,
    ]
    return flags if ptrr else [*flags, "--no-load-balance"]


# (label suffix, flags) for the 8-GPU ablation ladder: same kernels, adding one
# piece of the stack at a time.
LADDER = [
    ("ulysses8", cp(1, 8, "online")),
    ("ring8_python_contig", cp(8, 1, "python", ptrr=False)),
    ("ring8_flash_contig", cp(8, 1, "flash", ptrr=False)),
    ("ring8_online_contig", cp(8, 1, "online", ptrr=False)),
    ("ring8_online_ptrr", cp(8, 1, "online")),
    ("usp_r4u2_online_ptrr", cp(4, 2, "online")),
    ("usp_r2u4_online_ptrr", cp(2, 4, "online")),
    ("usp_r2u4_flash_ptrr", cp(2, 4, "flash")),
    ("usp_r2u4_python_ptrr", cp(2, 4, "python")),
    ("usp_r2u4_online_ptrr_symm", cp(2, 4, "online", transport="symm_mem_direct")),
    ("ring8_online_ptrr_symm", cp(8, 1, "online", transport="symm_mem_direct")),
]

HEADS = {
    "mha_32q": ["--heads", "32"],
    "llama_32q8kv": ["--heads", "32", "--kv-heads", "8"],
    "qwen7b_28q4kv": ["--heads", "28", "--kv-heads", "4"],
}


def _grid(heads, layouts, seqs, configs) -> list[tuple[str, list[str]]]:
    out = []
    for hname, layout, seq in itertools.product(heads, layouts, seqs):
        for name, flags in configs(hname):
            label = f"{hname}_{layout}_{seq // 1024}k_{name}"
            out.append(
                (
                    label,
                    ["--layout", layout, "--seq-len", str(seq), *HEADS[hname], *flags],
                )
            )
    return out


def _gqa8(hname: str) -> list[tuple[str, list[str]]]:
    # Ulysses degree is capped by the KV head count and must divide the query
    # heads, so Qwen2.5-7B's 28q/4kv cannot run Ulysses-8 at all.
    head = (
        [("ulysses8", cp(1, 8, "online"))]
        if hname == "llama_32q8kv"
        else [("usp_r2u4_python_contig", cp(2, 4, "python", ptrr=False))]
    )
    return [
        *head,
        ("ring8_python_contig", cp(8, 1, "python", ptrr=False)),
        ("ring8_online_ptrr", cp(8, 1, "online")),
        ("usp_r2u4_online_ptrr", cp(2, 4, "online")),
        ("usp_r2u4_online_ptrr_symm", cp(2, 4, "online", transport="symm_mem_direct")),
    ]


def _multi16(hname: str) -> list[tuple[str, list[str]]]:
    # Ulysses inside a node (NVLink all-to-all), ring across nodes.
    head = [("ulysses16", cp(1, 16, "online"))] if hname == "mha_32q" else []
    return [
        *head,
        ("ring16_python_contig", cp(16, 1, "python", ptrr=False)),
        ("ring16_online_ptrr", cp(16, 1, "online")),
        ("usp_r2u8_python_contig", cp(2, 8, "python", ptrr=False)),
        ("usp_r2u8_online_ptrr", cp(2, 8, "online")),
        ("usp_r2u8_online_ptrr_nccl4py", cp(2, 8, "online", transport="nccl4py")),
        ("usp_r4u4_online_ptrr", cp(4, 4, "online")),
    ]


# Head layouts of real open models (from their Hugging Face configs): query
# heads, KV heads, head dim, and the packed layout each is usually trained or
# served with.
MODELS = {
    # Llama-3.1-8B/70B, Qwen3-8B/32B, Qwen2.5-72B, Mistral-Small-24B
    "llama_32q8kv": (32, 8, 128, "docs", [262144, 1048576]),
    # Qwen2.5-7B (and Qwen3-235B-A22B shares the 4 KV-head cap)
    "qwen7b_28q4kv": (28, 4, 128, "mixed", [262144, 1048576]),
    # Wan2.1 / Wan2.2 14B video DiT: 40 MHA heads, bidirectional attention
    "wan_40q": (40, 40, 128, "full", [131072, 524288]),
    # 32 MHA heads: the best case for Ulysses
    "mha_32q": (32, 32, 128, "mixed", [262144, 1048576]),
}

# Vision-language models with a bidirectional mask over each image crop in
# the language model (Gemma 3's global layers), on packed interleaved
# image-text documents and on long videos.
VLM_MODELS = {
    # Gemma 3 4B: 4 KV heads cap Ulysses at 4 GPUs
    "gemma3_4b_8q4kv": (8, 4, 256, "vlm", [262144, 1048576]),
    # Gemma 3 12B: 8 KV heads cap Ulysses at 8 GPUs
    "gemma3_12b_16q8kv": (16, 8, 256, "vlm", [262144, 1048576]),
    # Gemma 3 27B
    "gemma3_27b_32q16kv": (32, 16, 128, "vlm", [262144, 1048576]),
    # Molmo2-8B (Qwen3-8B language model): all image tokens of a sample form
    # one bidirectional vision group; 8 KV heads cap Ulysses at 8 GPUs
    "molmo2_8b_32q8kv": (32, 8, 128, "vlm_group", [262144, 1048576]),
}
VIDEO_MODELS = {
    name: (q, kv, hd, "video", seqs)
    for name, (q, kv, hd, layout, seqs) in VLM_MODELS.items()
    if layout == "vlm"
}
# Forces the dense varlen paths (section 09 of the post): two dense calls per
# image-text sample without a ring, dense call shapes on every ring step.
DENSE = ["--interval-masks", "on", "--dense-ring", "on"]


def _divides(u: int, q: int, kv: int) -> bool:
    return q % u == 0 and kv % u == 0


def _model_grid(world: int, models=None) -> list[tuple[str, list[str]]]:
    """Every legal parallel layout for each model at a given GPU count."""
    out = []
    for name, (q, kv, hd, layout, seqs) in (models or MODELS).items():
        heads = ["--heads", str(q), "--kv-heads", str(kv), "--head-dim", str(hd)]
        if hd > 128:
            # FA4's SM90 hdim-256 kernel uses 80-key tiles, which 128-token
            # sparse blocks cannot use, so these heads take the dense paths.
            heads += DENSE
        configs: list[tuple[str, list[str]]] = []
        if _divides(world, q, kv):
            configs.append((f"ulysses{world}", cp(1, world, "online")))
        elif q % world == 0:
            rep = world // kv
            configs.append(
                (
                    f"ulysses{world}_kvrep{rep}",
                    [*cp(1, world, "online"), "--replicate-kv", str(rep)],
                )
            )
        # Ulysses on fewer GPUs plus data parallelism: the largest in-node
        # Ulysses degree the heads allow, data-parallel across the rest. No
        # attention traffic leaves the Ulysses group, if the sequence fits.
        u_dp = next((u for u in (8, 4, 2) if u < world and _divides(u, q, kv)), None)
        if u_dp and not (world <= 8 and _divides(world, q, kv)):
            configs.append((f"ulysses{u_dp}_dp{world // u_dp}", cp(1, u_dp, "online")))
        configs += [
            (f"ring{world}_python_contig", cp(world, 1, "python", ptrr=False)),
            (f"ring{world}_online_ptrr", cp(world, 1, "online")),
        ]
        best_u = None
        for u in [world // 2, world // 4]:
            if u > 1 and _divides(u, q, kv):
                configs.append(
                    (f"usp_r{world // u}u{u}_online_ptrr", cp(world // u, u, "online"))
                )
                best_u = best_u or u
        if world > 8:
            configs.append(
                (
                    f"ring{world}_online_ptrr_nccl4py",
                    cp(world, 1, "online", transport="nccl4py"),
                )
            )
            if best_u:
                configs.append(
                    (
                        f"usp_r{world // best_u}u{best_u}_online_ptrr_nccl4py",
                        cp(world // best_u, best_u, "online", transport="nccl4py"),
                    )
                )
        for seq in seqs:
            for cfg, flags in configs:
                out.append(
                    (
                        f"{name}_{layout}_{seq // 1024}k_{cfg}",
                        ["--layout", layout, "--seq-len", str(seq), *heads, *flags],
                    )
                )
    return out


# Real-model packs behind the 32-128 GPU tables (benchmarks/layouts/packs,
# MLSYS_RESULTS.md): query heads, KV heads, head dim, and flags every path
# shares. report_tables.py keys rows by their --label (the model name, "fm_"
# for FlashMask, "abl_" for the exchange x order ablation), so these configs
# end with that label; argparse keeps the last --label, and the sweep's own
# label only names the progress line.
PACK_MODELS = {
    "gemma3": (32, 16, 128, []),
    "gemma3_swa": (32, 16, 128, ["--sliding-window", "1024"]),
    "molmo2": (32, 8, 128, []),
    "bagel": (28, 4, 128, []),
    # Qwen3-VL-30B-A3B: 32q/4kv/128 on the qwen38 packs (MLSYS_RESULTS.md).
    "q3vl30": (32, 4, 128, []),
    "q38": (24, 4, 256, []),
}
# Pack-file prefix where it differs from the model name.
PACK_PREFIX = {"q3vl30": "qwen38", "q38": "qwen38"}
# Packs each model's sweeps read, by mix; lengths beyond the committed packs
# (gemma3 8M and 16M need pack.py and the public shards) are skipped until
# the pack exists, so every node walks the same list.
PACK_SIZES = {
    "gemma3": ("mixed", "video"),
    "gemma3_swa": ("mixed", "video"),
    "molmo2": ("mixed", "video"),
    "bagel": ("mix", "video"),
    "q3vl30": ("mixed", "video"),
}
OOM_LADDER = [1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24]


def _pack(model: str, mix: str, seq: int) -> str | None:
    """Committed pack name for a model, mix and length, or None if absent."""
    base = model.removesuffix("_swa")
    name = f"{PACK_PREFIX.get(base, base)}_{mix}_{seq // 1024}k"
    return name if (HERE / "layouts/packs" / f"{name}.json.gz").is_file() else None


def _pack_flags(model: str, pack: str, seq: int) -> list[str]:
    q, kv, hd, extra = PACK_MODELS[model]
    heads = ["--heads", str(q), "--kv-heads", str(kv), "--head-dim", str(hd)]
    layout = f"file:benchmarks/layouts/packs/{pack}.json#0"
    return ["--layout", layout, "--seq-len", str(seq), *heads, *extra]


def _ours(
    model: str,
    pack: str,
    seq: int,
    world: int,
    *,
    ulysses: int,
    exchange: str,
    balance: str = "chunks",
    label: str | None = None,
    extra: list[str] | None = None,
    combine: str = "online",
) -> tuple[str, list[str]]:
    """One of our configs on a pack: Ulysses inside the node, the rest ring."""
    ring = world // ulysses
    # The sliding-window variant shares the global layers' pack.
    mix = pack.split("_", 1)[1]
    # The sweep loop passes the returned name as the driver's --label; a
    # caller-supplied label replaces it (argparse would keep only the last
    # of two --label options).
    name = f"{model}_{mix}_{world}g_r{ring}u{ulysses}_{exchange}_{balance}"
    if extra:
        name += "_" + "_".join(a.strip("-").replace("-", "_") for a in extra)
    flags = [
        *_pack_flags(model, pack, seq),
        *cp(ring, ulysses, combine),
        "--kv-exchange",
        exchange,
        "--balance",
        balance,
        *(extra or []),
    ]
    if combine != "online":
        name += f"_combine_{combine}"
    return label or name, flags


# Best exact path per model and world size in MLSYS_RESULTS.md: gather on the
# image masks, halo ring for the sliding window, and the dense generation
# ring (chosen by the planner for the noise spans) on long BAGEL video.
def _best(model: str, pack: str, seq: int, world: int) -> tuple[str, list[str]]:
    if model == "bagel":
        exchange = "gather" if "mix" in pack else "ring"
        # --dense-ring defaults to "auto" (Blackwell only); the H100 long-video
        # rows ran the dense generation ring, so switch it on explicitly.
        extra = ["--dense-ring", "on"] if exchange == "ring" else []
        return _ours(model, pack, seq, world, ulysses=4, exchange=exchange, extra=extra)
    exchange = "ring" if model == "gemma3_swa" else "gather"
    return _ours(model, pack, seq, world, ulysses=8, exchange=exchange)


def _fm(
    model: str, pack: str, seq: int, *, ulysses: int | None = None
) -> tuple[str, list[str]]:
    """FlashMask CP, or FM + Ulysses with ``ulysses``, as the run lists did."""
    q, kv, hd, extra = PACK_MODELS[model]
    flags = _pack_flags(model, pack, seq)
    flags[1] = flags[1].removeprefix("file:")
    name = f"{model}_{pack.split('_', 1)[1]}_fm"
    if ulysses:
        flags += ["--ulysses", str(ulysses)]
        name += f"_u{ulysses}"
        if ulysses > kv:
            flags += ["--replicate-kv", str(ulysses // kv)]
    iters = ["--iters", "3", "--warmup", "1"] if seq >= 1 << 23 else []
    return name, [*flags, *iters, "--ipo"]


# End to end (bench_model.py, full model forward + backward): the decoder
# preset of each pack family, the Ulysses degree inside the node (the rest
# ring) and the KV exchange of the e2e rows in MLSYS_RESULTS.md: gather on the
# image masks, the dense generation ring on BAGEL video.
E2E_MODELS = {
    "gemma3": ("gemma3_27b", 8, "gather"),
    "molmo2": ("qwen3_8b", 8, "gather"),
    "bagel": ("mot7b", 4, "ring"),
}


def _e2e_flags(model: str, pack: str, seq: int) -> list[str]:
    """Build the bench_model.py flags every end-to-end run of a pack shares."""
    preset = E2E_MODELS[model][0]
    layout = f"file:benchmarks/layouts/packs/{pack}.json#0"
    # Fewer timed steps from 8M on, as the FlashMask attention ladders use.
    iters = ["--warmup", "1", "--iters", "3"] if seq >= 1 << 23 else []
    return ["--model", preset, "--layout", layout, "--seq-len", str(seq), *iters]


def _e2e(model: str, pack: str, seq: int, world: int) -> tuple[str, list[str]]:
    """Our full-model config on a pack: the attention tables' best path, end to end."""
    _, ulysses, exchange = E2E_MODELS[model]
    ring = world // ulysses
    mix = pack.split("_", 1)[1]
    name = f"e2e_{model}_{mix}_{world}g_r{ring}u{ulysses}_{exchange}"
    flags = [
        *_e2e_flags(model, pack, seq),
        "--attn",
        "usp",
        *cp(ring, ulysses, "online"),
        "--kv-exchange",
        exchange,
        "--balance",
        "chunks",
    ]
    if exchange == "ring":
        # The H100 default of --dense-ring is "auto" (off): switch the dense
        # generation ring on, as the recorded BAGEL e2e rows did.
        flags += ["--dense-ring", "on"]
    return name, flags


def _e2e_fm(
    model: str, pack: str, seq: int, world: int, *, ulysses: int = 1
) -> tuple[str, list[str]]:
    """FlashMask CP end to end, or FM + Ulysses with ``ulysses``, via bench_model.py."""
    mix = pack.split("_", 1)[1]
    name = f"e2e_{'fmu' if ulysses > 1 else 'fm'}_{model}_{mix}_{world}g"
    flags = [*_e2e_flags(model, pack, seq), "--attn", "flashmask"]
    if ulysses > 1:
        flags += ["--ulysses", str(ulysses)]
        name += f"_u{ulysses}"
    return name, flags


# MagiAttention settings for the 32-128 GPU cells. The MLSYS_RESULTS.md Magi
# rows are the fastest of chunk size {4096, 8192} x overlap degree {1, 4} at
# 32 GPUs (results/mlsys/magitune32a,b) and {4096, 8192} x {1} at 64 GPUs
# (magitune64b): chunk 8192 with degree 1 won every 2M cell and every 64-GPU
# cell, chunk 4096 within 0.1% at 64 GPUs; degree 4 never won at 2M. The
# overlap algorithm and min chunk size stayed at the driver's defaults
# (uniform, 512). One process per chunk size, so an OOM costs one setting.
MAGI_CHUNK_SIZES = (4096, 8192)
MAGI_OVERLAP_DEGREE = "1"


def _magi(
    model: str, pack: str, seq: int, world: int, chunk: int
) -> tuple[str, list[str]]:
    """MagiAttention on a pack at one of the tuned chunk sizes (magi_attn.py)."""
    mix = pack.split("_", 1)[1]
    name = f"magi_{model}_{mix}_{world}g_c{chunk}"
    flags = [
        *_pack_flags(model, pack, seq),
        "--chunk-size",
        str(chunk),
        "--overlap-degree",
        MAGI_OVERLAP_DEGREE,
    ]
    return name, flags


def _paper_sweeps() -> dict[str, list[tuple[str, list[str]]]]:
    """Sweeps filling the gaps left by the 32-128 GPU tables."""
    out: dict[str, list[tuple[str, list[str]]]] = {}
    # (a) Push to OOM: double the tokens at a fixed GPU count until a config
    # fails (failures are recorded as error rows), for Gemma 3 mixed and
    # BAGEL long video on our best path, FM + Ulysses and FlashMask CP.
    for world in (32, 64):
        ours, fm = [], []
        for model, mix, ulysses in [
            ("gemma3", "mixed", (8, 16)),
            ("bagel", "video", (4,)),
        ]:
            for seq in OOM_LADDER:
                pack = _pack(model, mix, seq)
                if pack is None:
                    continue
                ours.append(_best(model, pack, seq, world))
                fm.append(_fm(model, pack, seq))
                fm += [_fm(model, pack, seq, ulysses=u) for u in ulysses]
        out[f"oom{world}"], out[f"oom{world}_fm"] = ours, fm
    # (a') Ulysses only, as a ceiling probe: possible only where the Ulysses
    # degree equals the GPU count and divides the query heads, i.e. Gemma 3
    # (32 heads) at 32 GPUs with every K/V head replicated twice. BAGEL (28
    # heads) cannot be split 32 ways and 64 GPUs exceed every model's heads:
    # that limit is the point.
    out["oom32_u"] = [
        _ours(
            "gemma3",
            pack,
            seq,
            32,
            ulysses=32,
            exchange="gather",
            extra=["--replicate-kv", "2"],
        )
        for seq in OOM_LADDER
        if (pack := _pack("gemma3", "mixed", seq)) is not None
    ]
    # (a'') MagiAttention on the same OOM ladders at 32 GPUs, at the chunk
    # size that won every 2M cell (8192), so the max-sequence table has a Magi
    # column.
    out["oom32_magi"] = [
        _magi(model, pack, seq, 32, 8192)
        for model, mix in (("gemma3", "mixed"), ("bagel", "video"))
        for seq in OOM_LADDER
        if (pack := _pack(model, mix, seq)) is not None
    ]
    # (b) Strong-scaling fill-ins: 2M at 128 GPUs and 4M at 32 GPUs for Gemma 3
    # and Molmo2, BAGEL video 4M at 64 and 8M at 32.
    fills = {
        32: [("gemma3", 1 << 22), ("molmo2", 1 << 22), ("bagel", 1 << 23)],
        64: [("bagel", 1 << 22)],
        128: [("gemma3", 1 << 21), ("molmo2", 1 << 21)],
        # 256 GPUs (training queue, borrowed capacity): 4M for Gemma 3 and
        # Molmo2 pairs with the 4M/128 cells; BAGEL 8M and 16M pair with the
        # 64-GPU ladder two doublings away.
        256: [
            ("gemma3", 1 << 22),
            ("molmo2", 1 << 22),
            ("bagel", 1 << 23),
            ("bagel", 1 << 24),
        ],
    }
    for world, cells in fills.items():
        ours, fm = [], []
        for model, seq in cells:
            for mix in PACK_SIZES[model]:
                if model == "bagel" and mix != "video":
                    continue
                pack = _pack(model, mix, seq)
                if pack is None:
                    continue
                ours.append(_best(model, pack, seq, world))
                if model == "bagel":
                    fm += [_fm(model, pack, seq), _fm(model, pack, seq, ulysses=4)]
                else:
                    fm.append(_fm(model, pack, seq, ulysses=8))
                    if model == "gemma3" and world < 128:
                        fm.append(_fm(model, pack, seq, ulysses=16))
        out[f"fill{world}"], out[f"fill{world}_fm"] = ours, fm
    # (c) Chunked ring: chunks per rank k, with and without the per-step
    # refinement, on BAGEL video (dense generation ring) at 4M/32 and 8M/64.
    for world, seq in [(32, 1 << 22), (64, 1 << 23)]:
        pack = _pack("bagel", "video", seq)
        assert pack is not None
        chunks = [
            _ours(
                "bagel",
                pack,
                seq,
                world,
                ulysses=4,
                exchange="ring",
                extra=["--dense-ring", "on"],
            )
        ]
        for k in (1, 2, 4, 8, 16, 32, 64):
            for refine in (True, False):
                extra = ["--dense-ring", "on", "--dense-ring-chunks", str(k)]
                label = f"bagel_dring_k{k}"
                if not refine:
                    extra.append("--no-dense-ring-refine")
                    label += "_norefine"
                chunks.append(
                    _ours(
                        "bagel",
                        pack,
                        seq,
                        world,
                        ulysses=4,
                        exchange="ring",
                        extra=extra,
                        label=label,
                    )
                )
        out[f"chunks{world}"] = chunks
    # (d) Exchange x token order at 64 GPUs (the 32-GPU table used 2M video
    # at R4U8): 4M video at R8U8 for Molmo2, Gemma 3 global and sliding window.
    abl64 = []
    for model in ("molmo2", "gemma3", "gemma3_swa"):
        pack = _pack(model, "video", 1 << 22)
        assert pack is not None
        for exchange in ("ring", "gather"):
            for balance in ("ptrr", "chunks"):
                abl64.append(
                    _ours(
                        model,
                        pack,
                        1 << 22,
                        64,
                        ulysses=8,
                        exchange=exchange,
                        balance=balance,
                    )
                )
    out["abl64"] = abl64
    # (e) Collective bandwidth check on two nodes (bench_collectives.py), to
    # compare a cluster against results/collectives.jsonl before trusting its
    # multi-node attention timings.
    out["collectives16"] = [("collectives", ["--mib", "512"])]
    # (f) End to end (bench_model.py): the OOM ladders of (a) on the full
    # decoder, ours vs FlashMask CP and FM + Ulysses, plus Molmo2 mixed 1M at
    # 64 GPUs to pair with the e2e table's 32-GPU row (Gemma 3 mixed 1M at 64
    # is the first rung of its ladder).
    for world in (32, 64):
        ours, fm = [], []
        for model, mix, seqs in [
            ("gemma3", "mixed", OOM_LADDER),
            ("bagel", "video", OOM_LADDER),
            ("molmo2", "mixed", [1 << 20] if world == 64 else []),
        ]:
            for seq in seqs:
                pack = _pack(model, mix, seq)
                if pack is None:
                    continue
                ours.append(_e2e(model, pack, seq, world))
                fm.append(_e2e_fm(model, pack, seq, world))
                fm.append(
                    _e2e_fm(model, pack, seq, world, ulysses=E2E_MODELS[model][1])
                )
        out[f"e2e{world}"], out[f"e2e{world}_fm"] = ours, fm
    # (j) The 128-GPU points that complete the 2M 32 -> 64 -> 128 strong-scaling
    # curves for every system: FlashMask CP (plain all-gather, no Ulysses) on the
    # Gemma 3 / Molmo2 2M layouts, and MagiAttention at 2M on the eight layouts
    # of magi32 / magi64 (both chunk sizes, degree 1).
    # (k) Grid gap-fill (2026-10-30): the cells Table 4 left empty. Gemma 3's
    # sliding-window layers at 4M on 64 and 128 GPUs (ours at U = 4 and 8, FM +
    # Ulysses at U = 8 and 16; FlashMask CP cannot fit Gemma 3 at 4M), FM +
    # Ulysses on the sliding-window 2M packs at 64 GPUs, and BAGEL long video at
    # 8M on 128 GPUs (dense generation ring and sparse all-to-all, FlashMask CP
    # and FM + Ulysses U = 4).
    swa4m = [
        p
        for mix in PACK_SIZES["gemma3_swa"]
        if (p := _pack("gemma3_swa", mix, 1 << 22))
    ]
    swa2m = [
        p
        for mix in PACK_SIZES["gemma3_swa"]
        if (p := _pack("gemma3_swa", mix, 1 << 21))
    ]
    bagel8m = _pack("bagel", "video", 1 << 23)
    assert bagel8m is not None and len(swa4m) == 2 and len(swa2m) == 2
    for world in (64, 128):
        out[f"gap{world}"] = [
            _ours("gemma3_swa", pack, 1 << 22, world, ulysses=u, exchange="ring")
            for pack in swa4m
            for u in (8, 4)
        ]
        out[f"gap{world}_fm"] = [
            _fm("gemma3_swa", pack, 1 << 22, ulysses=u)
            for pack in swa4m
            for u in (8, 16)
        ]
    swa1m = [
        p
        for mix in PACK_SIZES["gemma3_swa"]
        if (p := _pack("gemma3_swa", mix, 1 << 20))
    ]
    assert len(swa1m) == 2
    out["gap32_fm"] = [
        _fm("gemma3_swa", pack, 1 << 20, ulysses=u) for pack in swa1m for u in (8, 16)
    ]
    # Second round: FlashMask CP cells that were never run (not out of memory),
    # and BAGEL's packed mix at 64 and 128 GPUs (2M/4M at 64, 4M at 128, as the
    # other models), ours (gather, U = 4), FlashMask CP and FM + Ulysses U = 4.
    bmix = {s: _pack("bagel", "mix", s) for s in (1 << 21, 1 << 22)}
    assert all(bmix.values())
    out["gapb64"] = [_best("bagel", bmix[s], s, 64) for s in (1 << 21, 1 << 22)]
    out["gapb128"] = [_best("bagel", bmix[1 << 22], 1 << 22, 128)]
    out["gapb64_fm"] = [
        _fm(m, _pack(m, mix, 1 << 22), 1 << 22)
        for m, mix in (("molmo2", "mixed"), ("molmo2", "video"))
    ] + [
        f
        for s in (1 << 21, 1 << 22)
        for f in (_fm("bagel", bmix[s], s), _fm("bagel", bmix[s], s, ulysses=4))
    ]
    out["gapb128_fm"] = [
        _fm(m, _pack(m, "mixed", 1 << 22), 1 << 22) for m in ("molmo2", "q38", "q3vl30")
    ] + [
        _fm("bagel", bmix[1 << 22], 1 << 22),
        _fm("bagel", bmix[1 << 22], 1 << 22, ulysses=4),
    ]
    out["gap64_fm"] += [
        _fm("gemma3_swa", pack, 1 << 21, ulysses=u) for pack in swa2m for u in (8, 16)
    ]
    out["gap128"] += [
        _ours(
            "bagel",
            bagel8m,
            1 << 23,
            128,
            ulysses=4,
            exchange="ring",
            extra=["--dense-ring", "on"],
        ),
        _ours("bagel", bagel8m, 1 << 23, 128, ulysses=4, exchange="gather"),
    ]
    out["gap128_fm"] += [
        _fm("bagel", bagel8m, 1 << 23),
        _fm("bagel", bagel8m, 1 << 23, ulysses=4),
    ]
    out["fill128cp_fm"] = [
        _fm(model, pack, 1 << 21)
        for model in ("gemma3", "molmo2")
        for mix in PACK_SIZES[model]
        if (pack := _pack(model, mix, 1 << 21)) is not None
    ]
    out["magi128_2m"] = [
        _magi(model, pack, 1 << 21, 128, chunk)
        for model in ("gemma3", "gemma3_swa", "molmo2", "q3vl30")
        for mix in PACK_SIZES[model]
        if (pack := _pack(model, mix, 1 << 21)) is not None
        for chunk in MAGI_CHUNK_SIZES
    ]
    # (h) Online FA4 kernel vs per-step merges, single GPU (bench_online_kernel.py):
    # time and mean relative error vs fp32 for W = 1..32 chunks at three query
    # block sizes (what one rank sees over W ring steps). Runs on 8 GPUs of one
    # node; every rank repeats the same measurement.
    out["online8"] = [
        (
            f"online_q{q >> 10}k",
            [
                "--seq-q",
                str(q),
                "--chunks",
                "1",
                "2",
                "4",
                "8",
                "16",
                "32",
                "--heads",
                "32",
                "--head-dim",
                "128",
                "--iters",
                "20",
                "--warmup",
                "5",
            ],
        )
        for q in (16384, 65536, 131072)
    ]
    # (i) Ring combine mode at 32 GPUs: the same ring configs with the online
    # kernel (fp32 state inside the kernel), the fused flash_attn_combine
    # merge of bf16 partials, and the PyTorch merge. "fp32" merges fp32
    # partials outside the kernel (on Hopper the fresh-state output of the
    # online kernel; split-KV partials on Blackwell) and is only listed for
    # the configs that merge per-step partials: the dense ring never uses the
    # combine mode.
    combine = []
    for seq in (1 << 20, 1 << 21, 1 << 22):
        pack = _pack("bagel", "video", seq)
        for mode in ("online", "flash", "python"):
            combine.append(
                _ours(
                    "bagel",
                    pack,
                    seq,
                    32,
                    ulysses=4,
                    exchange="ring",
                    extra=["--dense-ring", "on"],
                    combine=mode,
                )
            )
    # the block-sparse ring (dense ring off) is the path that merges per-step partials
    for seq in (1 << 21, 1 << 22):
        pack = _pack("bagel", "video", seq)
        for mode in ("online", "flash", "python"):
            combine.append(
                _ours(
                    "bagel",
                    pack,
                    seq,
                    32,
                    ulysses=4,
                    exchange="ring",
                    extra=["--dense-ring", "off"],
                    combine=mode,
                )
            )
    pack = _pack("gemma3_swa", "video", 1 << 22)
    for mode in ("online", "flash", "python"):
        combine.append(
            _ours(
                "gemma3_swa",
                pack,
                1 << 22,
                32,
                ulysses=8,
                exchange="ring",
                combine=mode,
            )
        )
    out["combine32"] = combine
    # (g) MagiAttention strong-scaling pairs on the layouts of the strong-scaling
    # table: 2M at 32 and 64 GPUs, 4M at 64 and 128, for Gemma 3 27B (global
    # and sliding window), Molmo2-8B and Qwen3-VL-30B-A3B, mixed and video.
    # Qwen3.8-27B (head_dim 256) is out: MagiAttention's Hopper kernel asserts
    # head_dim <= 128 (sweep_magi.py). Labels start with "magi_", which
    # report_tables.py reads as the MagiAttention column.
    magi_cells = {32: [1 << 21], 64: [1 << 21, 1 << 22], 128: [1 << 22]}
    for world, seqs in magi_cells.items():
        magi = []
        for seq in seqs:
            for model in ("gemma3", "gemma3_swa", "molmo2", "q3vl30"):
                for mix in PACK_SIZES[model]:
                    pack = _pack(model, mix, seq)
                    if pack is None:
                        continue
                    for chunk in MAGI_CHUNK_SIZES:
                        magi.append(_magi(model, pack, seq, world, chunk))
        out[f"magi{world}"] = magi
    return out


# Multi-node sweeps and the GPU count each one is built for.
PAPER_WORLDS = {
    "oom32": 32,
    "oom32_fm": 32,
    "oom32_u": 32,
    "oom32_magi": 32,
    "oom64": 64,
    "oom64_fm": 64,
    "fill32": 32,
    "fill32_fm": 32,
    "fill64": 64,
    "fill64_fm": 64,
    "fill128": 128,
    "fill256": 256,
    "online8": 8,
    "fill128cp_fm": 128,
    "magi128_2m": 128,
    "combine32": 32,
    "fill256_fm": 256,
    "fill128_fm": 128,
    "gap32_fm": 32,
    "gapb64": 64,
    "gapb64_fm": 64,
    "gapb128": 128,
    "gapb128_fm": 128,
    "gap64": 64,
    "gap64_fm": 64,
    "gap128": 128,
    "gap128_fm": 128,
    "chunks32": 32,
    "chunks64": 64,
    "abl64": 64,
    "collectives16": 16,
    "magi32": 32,
    "magi64": 64,
    "magi128": 128,
    "e2e32": 32,
    "e2e32_fm": 32,
    "e2e64": 64,
    "e2e64_fm": 64,
}


# Fewer timed iterations for the model grids: full-attention configs take
# seconds per step.
SWEEP_ARGS = {
    name: ["--warmup", "2", "--iters", "5", "--trials", "2"]
    for name in (
        "heads8",
        "heads16",
        "heads16_dp",
        "vlm8",
        "vlm16",
        "video8",
        "video16",
        "vlm8_dense",
        "vlm8_hd128",
        "vlm16_hd128",
        "video_group8",
    )
}
SWEEP_ARGS.update(
    {
        name: ["--warmup", "2", "--iters", "5", "--trials", "1"]
        for name in PAPER_WORLDS
        # bench_collectives.py has its own fixed iteration count and takes no
        # --warmup/--iters/--trials; bench_model.py (e2e*) takes no --trials
        # and gets its --warmup/--iters per config.
        if not name.endswith("_fm")
        and name != "online8"
        and name != "collectives16"
        and not name.startswith("e2e")
    }
)


def _sweeps() -> dict[str, list[tuple[str, list[str]]]]:
    ablation = []
    for layout, seq in itertools.chain(
        itertools.product(["mixed", "docs", "causal", "full"], [131072, 524288]),
        [("mixed", 1048576)],
    ):
        for name, flags in LADDER:
            ablation.append(
                (
                    f"{layout}_{seq // 1024}k_{name}",
                    ["--layout", layout, "--seq-len", str(seq), *flags],
                )
            )
    smoke = [
        (f"smoke_{n}", ["--layout", "mixed", "--seq-len", "65536", *f])
        for n, f in LADDER
    ]
    gqa8 = _grid(
        ["llama_32q8kv", "qwen7b_28q4kv"], ["mixed", "causal"], [262144, 1048576], _gqa8
    )
    multinode16 = _grid(
        ["mha_32q", "llama_32q8kv"],
        ["mixed", "causal"],
        [524288, 1048576, 2097152],
        _multi16,
    )
    # 7B multimodal MoT decoder (bench_model.py). 28 query heads cannot split 8 ways,
    # so Ulysses-only is not an option on 8 GPUs; the no-parallelism baselines
    # run one full sequence per GPU at lengths that still fit.
    model8 = []
    for seq in [131072, 262144, 524288, 1048576]:
        for name, flags in [
            ("ring8_python_contig", cp(8, 1, "python", ptrr=False)),
            ("ring8_online_ptrr", cp(8, 1, "online")),
            ("usp_r4u2_online_ptrr", cp(4, 2, "online")),
            ("usp_r2u4_online_ptrr", cp(2, 4, "online")),
            ("usp_r2u4_python_contig", cp(2, 4, "python", ptrr=False)),
        ]:
            label = f"mm7b_mixed_{seq // 1024}k_{name}"
            model8.append((label, ["--seq-len", str(seq), "--attn", "usp", *flags]))
    for seq in [16384, 32768, 65536]:
        for attn in ["flex", "fa4"]:
            label = f"mm7b_mixed_{seq // 1024}k_single_{attn}"
            model8.append((label, ["--seq-len", str(seq), "--attn", attn]))
    # Molmo2-8B's language model (Qwen3-8B dimensions, 32 query / 8 KV heads)
    # end to end on its vision-group mask. 8 KV heads allow Ulysses-8 here.
    model8_vlm = []
    vlm_model = ["--model", "qwen3_8b", "--layout", "vlm_group"]
    # 1M takes several minutes per step for the slower layouts; 512k is the cap.
    for seq in [131072, 262144, 524288]:
        for name, flags in [
            ("ulysses8", cp(1, 8, "online")),
            ("ring8_python_contig", cp(8, 1, "python", ptrr=False)),
            ("ring8_online_ptrr", cp(8, 1, "online")),
            ("usp_r4u2_online_ptrr", cp(4, 2, "online")),
            ("usp_r2u4_online_ptrr", cp(2, 4, "online")),
        ]:
            label = f"qwen3_8b_vlm_group_{seq // 1024}k_{name}"
            model8_vlm.append(
                (label, [*vlm_model, "--seq-len", str(seq), "--attn", "usp", *flags])
            )
    for seq in [16384, 32768, 65536]:
        for attn in ["flex", "fa4"]:
            label = f"qwen3_8b_vlm_group_{seq // 1024}k_single_{attn}"
            model8_vlm.append(
                (label, [*vlm_model, "--seq-len", str(seq), "--attn", attn])
            )
    # PyTorch-native context parallelism on the same 8 GPUs and MHA heads.
    torchcp8 = []
    for layout, seq in itertools.product(["mixed", "causal", "full"], [131072, 524288]):
        tag = f"{layout}_{seq // 1024}k"
        base = ["--layout", layout, "--seq-len", str(seq)]
        torchcp8.append(
            (f"{tag}_torch_flex_allgather_ptrr", [*base, "--mode", "flex_allgather"])
        )
        if layout != "mixed":
            for rotate in ["allgather", "alltoall"]:
                torchcp8.append(
                    (
                        f"{tag}_torch_sdpa_ring_{rotate}",
                        [*base, "--mode", "sdpa_ring", "--rotate", rotate],
                    )
                )

    # Transport comparison with profiler traces (run with --profile-dir):
    # does moving KV slow attention down? "serial" runs the transport on the
    # compute stream, so attention never shares the GPU with it.
    def transports(
        ring: int, uly: int, seq: int, names: list[str]
    ) -> list[tuple[str, list[str]]]:
        base = [
            "--layout",
            "mixed",
            "--seq-len",
            str(seq),
            "--heads",
            "32",
            "--kv-heads",
            "8",
        ]
        out = []
        for t in names:
            flags = cp(ring, uly, "online", transport=t)
            out.append((f"mixed_{seq // 1024}k_r{ring}u{uly}_{t}", [*base, *flags]))
        out.append(
            (
                f"mixed_{seq // 1024}k_r{ring}u{uly}_nccl_serial",
                [*base, *cp(ring, uly, "online"), "--no-comm-stream"],
            )
        )
        return out

    transport8 = transports(
        8, 1, 524288, ["nccl", "nccl4py", "symm_mem", "symm_mem_direct"]
    )
    transport16 = transports(2, 8, 1048576, ["nccl", "nccl4py"]) + transports(
        16, 1, 1048576, ["nccl", "nccl4py"]
    )
    # Profiled pairs explaining Ulysses-16 vs USP across two nodes; the
    # llama-heads-on-mixed pair separates the head layout from the mask.
    attrib16 = []
    for name, heads, layout, uly in [
        (
            "llama_32q8kv",
            ["--heads", "32", "--kv-heads", "8"],
            "docs",
            [*cp(1, 16, "online"), "--replicate-kv", "2"],
        ),
        (
            "llama_32q8kv",
            ["--heads", "32", "--kv-heads", "8"],
            "mixed",
            [*cp(1, 16, "online"), "--replicate-kv", "2"],
        ),
        (
            "mha_32q",
            ["--heads", "32", "--kv-heads", "32"],
            "mixed",
            cp(1, 16, "online"),
        ),
    ]:
        base = ["--layout", layout, "--seq-len", "262144", *heads]
        attrib16.append((f"{name}_{layout}_256k_ulysses16", [*base, *uly]))
        attrib16.append(
            (f"{name}_{layout}_256k_usp_r2u8_online_ptrr", [*base, *cp(2, 8, "online")])
        )
    heads8 = _model_grid(8)
    heads16 = _model_grid(16)
    # Added after heads16 started: only the data-parallel Ulysses entries.
    heads16_dp = [(label, flags) for label, flags in heads16 if "_dp" in label]
    # Our side of torchcp8 for the layouts the ablation did not cover.
    torchcp8_match = []
    for layout, seq in itertools.product(["causal", "full"], [131072, 524288]):
        for name, flags in [
            ("ulysses8", cp(1, 8, "online")),
            ("usp_r2u4_online_ptrr", cp(2, 4, "online")),
            ("ring8_online_ptrr", cp(8, 1, "online")),
            ("ring8_python_contig", cp(8, 1, "python", ptrr=False)),
        ]:
            label = f"{layout}_{seq // 1024}k_{name}"
            torchcp8_match.append(
                (label, ["--layout", layout, "--seq-len", str(seq), *flags])
            )
    # The 7B multimodal decoder at lengths that need two nodes (1M OOMs on 8 GPUs); Ulysses is
    # capped at 4 by the 4 KV heads, so the options are ring or USP.
    model16 = []
    for seq in [524288, 1048576]:
        for name, flags in [
            ("ring16_python_contig", cp(16, 1, "python", ptrr=False)),
            ("ring16_online_ptrr", cp(16, 1, "online")),
            ("usp_r8u2_online_ptrr", cp(8, 2, "online")),
            ("usp_r4u4_online_ptrr", cp(4, 4, "online")),
            ("usp_r4u4_online_ptrr_nccl4py", cp(4, 4, "online", transport="nccl4py")),
        ]:
            label = f"mm7b_mixed_{seq // 1024}k_{name}"
            model16.append((label, ["--seq-len", str(seq), "--attn", "usp", *flags]))
    # yunchang (xDiT's USP) on the plain masks it supports, 8 GPUs, 32 MHA heads.
    yunchang8 = []
    for layout, seq in itertools.product(["causal", "full"], [131072, 524288]):
        for name, (ring, uly) in {
            "ulysses8": (1, 8),
            "ring8": (8, 1),
            "usp_r2u4": (2, 4),
            "usp_r4u2": (4, 2),
        }.items():
            label = f"{layout}_{seq // 1024}k_yunchang_{name}_fa3"
            flags = [
                "--layout",
                layout,
                "--seq-len",
                str(seq),
                "--ring",
                str(ring),
                "--ulysses",
                str(uly),
                "--attn-type",
                "fa3",
            ]
            yunchang8.append((label, flags))
    # TransformerEngine context parallelism (Megatron's), 8 GPUs, 32 MHA heads.
    te8 = []
    for layout, seq in itertools.product(["causal", "full", "docs"], [131072, 524288]):
        for name, flags in {
            "a2a8": ["--comm", "a2a"],
            "p2p8": ["--comm", "p2p"],
            "allgather8": ["--comm", "all_gather"],
            "a2a_p2p_r2u4": ["--comm", "a2a+p2p", "--ulysses", "4"],
            "a2a_p2p_r4u2": ["--comm", "a2a+p2p", "--ulysses", "2"],
        }.items():
            label = f"{layout}_{seq // 1024}k_te_{name}"
            te8.append((label, ["--layout", layout, "--seq-len", str(seq), *flags]))
    # MagiAttention on every layout, 8 GPUs, 32 MHA heads.
    magi8 = [
        (f"{layout}_{seq // 1024}k_magi8", ["--layout", layout, "--seq-len", str(seq)])
        for layout, seq in itertools.product(
            ["mixed", "docs", "causal", "full"], [131072, 524288]
        )
    ]
    # MagiAttention on the real model head layouts of heads8 (same labels).
    magi8_models = [
        (
            f"{name}_{layout}_{seq // 1024}k_magi8",
            [
                "--heads",
                str(q),
                "--kv-heads",
                str(kv),
                "--layout",
                layout,
                "--seq-len",
                str(seq),
            ],
        )
        for name, (q, kv, _, layout, seqs) in MODELS.items()
        for seq in seqs
    ]
    # Our VLM grids and MagiAttention on the same VLM head layouts and masks.
    vlm8 = _model_grid(8, VLM_MODELS)
    vlm16 = _model_grid(16, VLM_MODELS)
    video8 = _model_grid(8, VIDEO_MODELS)
    video16 = _model_grid(16, VIDEO_MODELS)
    # Control for FA4's head_dim 256 backward, which slows down with call
    # length and so penalizes long-call layouts (Ulysses) most: Gemma 3 4B and
    # 12B head counts at head_dim 128.
    hd128 = {
        f"{name}_hd128": (q, kv, 128, layout, seqs)
        for name, (q, kv, hd, layout, seqs) in VLM_MODELS.items()
        if hd == 256
    }
    vlm8_hd128 = _model_grid(8, hd128)
    vlm16_hd128 = _model_grid(16, hd128)
    # The same grids with the dense paths forced on (H100 defaults them off
    # for image masks); data-parallel Ulysses rows are the same as vlm8's.
    vlm8_dense = [
        (f"{label}_dense", [*flags, *DENSE])
        for label, flags in [*vlm8, *video8]
        if "_dp" not in label
        and "python_contig" not in label
        and DENSE[0] not in flags
        and "vlm_group" not in label
    ]
    magi8_vlm = [
        (
            f"{name}_{layout}_{seq // 1024}k_magi8",
            [
                "--heads",
                str(q),
                "--kv-heads",
                str(kv),
                "--head-dim",
                str(hd),
                "--layout",
                layout,
                "--seq-len",
                str(seq),
            ],
        )
        for name, (q, kv, hd, layout, seqs) in [
            *VLM_MODELS.items(),
            *VIDEO_MODELS.items(),
        ]
        for seq in seqs
    ]
    # Molmo2-style video: all frames of a video form one vision group, so the
    # block-sparse path is the only one (no dense paths). 32 heads and Molmo2-8B.
    video_group8 = [
        (
            f"video_group_{seq // 1024}k_{name}",
            ["--layout", "video_group", "--seq-len", str(seq), *flags],
        )
        for seq in [131072, 524288]
        for name, flags in LADDER
        if "symm" not in name
    ] + _model_grid(
        8, {"molmo2_8b_32q8kv": (32, 8, 128, "video_group", [262144, 1048576])}
    )
    # The 32-head library comparison and ablation ladder on the VLM masks,
    # without USP over symmetric memory (its backward still deadlocks).
    ladder = [(n, f) for n, f in LADDER if n != "usp_r2u4_online_ptrr_symm"]
    vlm_ablation = [
        (
            f"{layout}_{seq // 1024}k_{name}",
            ["--layout", layout, "--seq-len", str(seq), *flags],
        )
        for layout, seq in itertools.product(
            ["vlm", "video", "vlm_group"], [131072, 524288]
        )
        for name, flags in ladder
    ] + [
        (f"vlm_1024k_{name}", ["--layout", "vlm", "--seq-len", "1048576", *flags])
        for name, flags in ladder
    ]
    magi8_vlm32 = [
        (f"{layout}_{seq // 1024}k_magi8", ["--layout", layout, "--seq-len", str(seq)])
        for layout, seq in itertools.product(
            ["vlm", "video", "vlm_group"], [131072, 524288]
        )
    ] + [
        # A video vision group is one slice per pair of frames: 128k only.
        ("video_group_128k_magi8", ["--layout", "video_group", "--seq-len", "131072"])
    ]
    torchcp8_vlm = [
        (
            f"{layout}_{seq // 1024}k_torch_flex_allgather_ptrr",
            ["--layout", layout, "--seq-len", str(seq), "--mode", "flex_allgather"],
        )
        for layout, seq in itertools.product(
            ["vlm", "video", "vlm_group", "video_group"], [131072, 524288]
        )
    ]
    return {
        "vlm8": vlm8,
        "model8_vlm": model8_vlm,
        "vlm16": vlm16,
        "video8": video8,
        "video16": video16,
        "vlm8_dense": vlm8_dense,
        "vlm8_hd128": vlm8_hd128,
        "video_group8": video_group8,
        "vlm16_hd128": vlm16_hd128,
        "magi8_vlm": magi8_vlm,
        "magi8_vlm32": magi8_vlm32,
        "vlm_ablation": vlm_ablation,
        "torchcp8_vlm": torchcp8_vlm,
        "magi8": magi8,
        "magi8_models": magi8_models,
        "te8": te8,
        "yunchang8": yunchang8,
        "model16": model16,
        "torchcp8_match": torchcp8_match,
        "attrib16": attrib16,
        "heads8": heads8,
        "heads16": heads16,
        "heads16_dp": heads16_dp,
        "transport8": transport8,
        "transport16": transport16,
        "torchcp8": torchcp8,
        "ablation": ablation,
        "smoke": smoke,
        "gqa8": gqa8,
        "multinode16": multinode16,
        "model8": model8,
        **_paper_sweeps(),
    }


# Sweeps that drive a script other than bench_attention.py.
SCRIPTS = {
    "collectives16": "bench_collectives.py",
    "model8": "bench_model.py",
    "model16": "bench_model.py",
    "model8_vlm": "bench_model.py",
    "torchcp8": "baselines/torch_cp.py",
    "yunchang8": "baselines/yunchang_usp.py",
    "te8": "baselines/te_cp.py",
    "magi8": "baselines/magi_attn.py",
    "magi8_models": "baselines/magi_attn.py",
    "magi8_vlm": "baselines/magi_attn.py",
    "magi8_vlm32": "baselines/magi_attn.py",
    # Multi-node MagiAttention (the cluster launcher builds its environment on
    # every node for sweeps named magi*).
    "magi32": "baselines/magi_attn.py",
    "magi64": "baselines/magi_attn.py",
    "magi128": "baselines/magi_attn.py",
    "oom32_magi": "baselines/magi_attn.py",
    "magi128_2m": "baselines/magi_attn.py",
    "online8": "bench_online_kernel.py",
    "torchcp8_vlm": "baselines/torch_cp.py",
    **{
        name: "baselines/flashmask_cp_bench.py"
        for name in PAPER_WORLDS
        if name.endswith("_fm")
    },
    # FlashMask end to end is bench_model.py --attn flashmask; the _fm suffix
    # still selects FlashMask's environment in the cluster launcher.
    "e2e32": "bench_model.py",
    "e2e32_fm": "bench_model.py",
    "e2e64": "bench_model.py",
    "e2e64_fm": "bench_model.py",
}


def main() -> None:
    """Launch the selected benchmark sweep and save worker result rows.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    sweeps = _sweeps()
    p = argparse.ArgumentParser()
    p.add_argument("sweep", choices=sorted(sweeps))
    p.add_argument("--out", required=True)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument(
        "--combine",
        default=None,
        help="replace every config's ring combine mode (e.g. fp32 on Blackwell, "
        "where the online kernel does not run); labels are unchanged",
    )
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--node-rank", type=int, default=0)
    p.add_argument("--master-addr", default=None)
    p.add_argument("--master-port", type=int, default=29611)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--profile-dir", default=None)
    p.add_argument(
        "--filter", default=None, help="Only run labels matching this regex."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each driver command as a JSON list instead of running it "
        "(benchmarks/preflight.py checks them against the drivers' parsers).",
    )
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    # Every node must walk the same config list in lockstep, so only resume
    # (skip finished labels) on single-node sweeps.
    if out.exists() and args.nnodes == 1:
        # finished labels only: a config whose earlier attempt failed is rerun
        done = {
            row.get("label")
            for row in (
                json.loads(line)
                for line in out.read_text().splitlines()
                if line.strip()
            )
            if "error" not in row
        }

    launch = ["--nproc-per-node", str(args.nproc)]

    def _launch_args(index: int) -> list[str]:
        """torchrun arguments for the ``index``-th config of this sweep.

        Multi-node sweeps give every config its own rendezvous port and id:
        nodes finish a config at slightly different times, and a node that
        reuses the port would join the previous run's closing rendezvous on
        rank 0 (``RendezvousClosedError``).
        """
        if args.nnodes <= 1:
            return list(launch)
        return [
            *launch,
            "--nnodes",
            str(args.nnodes),
            "--node-rank",
            str(args.node_rank),
            "--master-addr",
            args.master_addr,
            "--master-port",
            str(args.master_port + index),
            "--rdzv-id",
            f"{args.sweep}-{index}",
        ]

    for index, (label, flags) in enumerate(sweeps[args.sweep]):
        if args.combine and "--combine" in flags:
            flags = list(flags)
            flags[flags.index("--combine") + 1] = args.combine
        if label in done or (args.filter and not re.search(args.filter, label)):
            continue
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            *_launch_args(index),
            str(HERE / SCRIPTS.get(args.sweep, "bench_attention.py")),
            "--label",
            label,
            *flags,
            *SWEEP_ARGS.get(args.sweep, []),
            *args.extra,
        ]
        if args.profile_dir:
            cmd += ["--profile-dir", args.profile_dir]
        if args.dry_run:
            print(json.dumps(cmd))
            continue
        start = time.time()
        returncode = None
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.timeout
            )
            returncode = proc.returncode
            rows = [
                json.loads(line.split("RESULT ", 1)[1])
                for line in proc.stdout.splitlines()
                if "RESULT " in line
            ]
            if not rows:
                text = proc.stderr or proc.stdout
                # torchrun's failure summary fills the tail; keep the first
                # exception lines from the workers too.
                errors = [
                    line
                    for line in text.splitlines()
                    if re.search(r"(Error|Exception)\b|usage:|error:", line, re.I)
                    and "elastic" not in line
                ]
                tail = "\n".join(dict.fromkeys(errors[:8])) + "\n---\n" + text[-1500:]
                rows = [
                    {
                        "label": label,
                        "error": f"exit={proc.returncode}",
                        "tail": tail,
                        # torchrun's summary hides the worker's own message
                        # (e.g. argparse); keep the start of stderr as well.
                        "stderr_head": "\n".join((proc.stderr or "").splitlines()[:40]),
                    }
                ]
        except subprocess.TimeoutExpired:
            rows = [{"label": label, "error": "timeout"}]
        if args.node_rank != 0:
            # Only node 0 hosts global rank 0, which prints the result row.
            print(f"{label}: {rows[0].get('error', 'ok')}", flush=True)
            # Keep this node's failure on the shared log: node 0's row only
            # sees the remote side of a failure that starts on another node.
            if rows[0].get("tail"):
                print(rows[0]["tail"], flush=True)
            continue
        for row in rows:
            row["launch_command"] = cmd
            if returncode is not None and returncode != 0:
                row.setdefault("error", f"exit={returncode}")
            row["wall_s"] = round(time.time() - start, 1)
            with out.open("a") as f:
                f.write(result_json(row) + "\n")
        ms = rows[0].get("median_ms", rows[0].get("step_ms_median"))
        # drivers without a median_ms (bench_online_kernel.py) report row counts
        status = rows[0].get("error") or (
            f"{ms:.2f} ms" if ms is not None else f"{len(rows)} rows"
        )
        print(f"{label}: {status}", flush=True)


if __name__ == "__main__":
    main()
