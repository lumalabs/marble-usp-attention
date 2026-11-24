# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Break down GPU time in profiler traces written by ``bench_attention.py``.

For each Chrome trace (``--profile-dir`` output, one attention forward +
backward on rank 0) reports milliseconds of:

- ``attn``: FlashAttention / online FA4 kernels,
- ``nccl``: NCCL kernels (these occupy SMs while they progress),
- ``memcpy``: copy-engine transfers (no SMs),
- ``other``: every other kernel,
- ``attn_overlapping_nccl``: attention time that runs while an NCCL kernel is
  resident on the same GPU,

plus the wall span of GPU activity. Comparing ``attn`` between transports, and
against a run with ``--no-comm-stream`` (transport serialized, so attention
never shares the GPU), measures how much the transport slows attention down::

    python benchmarks/analyze_trace.py traces/*.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from provenance import result_json

ATTN = re.compile(r"flash|attn|attention", re.IGNORECASE)


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        total += max(0.0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path: Path) -> dict:
    """Summarize GPU activity and attention/NCCL overlap in a trace.

    Args:
        path: Path to a Chrome-format JSON trace with traceEvents.

    Returns:
        A row with the trace stem and attention, NCCL, memcpy, other-kernel,
        overlap and total GPU-span durations in milliseconds.
    """
    events = json.loads(path.read_text()).get("traceEvents", [])
    spans: dict[str, list[tuple[float, float]]] = {
        "attn": [],
        "nccl": [],
        "memcpy": [],
        "other": [],
    }
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        span = (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        if cat == "gpu_memcpy":
            spans["memcpy"].append(span)
        elif cat == "kernel":
            if "nccl" in name.lower():
                spans["nccl"].append(span)
            elif ATTN.search(name):
                spans["attn"].append(span)
            else:
                spans["other"].append(span)
    gpu = [s for v in spans.values() for s in v]
    row = {"trace": path.stem}
    for key, value in spans.items():
        row[f"{key}_ms"] = sum(end - start for start, end in value) / 1e3
    row["attn_overlapping_nccl_ms"] = (
        _overlap(_merge(spans["attn"]), _merge(spans["nccl"])) / 1e3
    )
    row["gpu_span_ms"] = (
        (max(e for _, e in gpu) - min(s for s, _ in gpu)) / 1e3 if gpu else 0.0
    )
    return row


def main() -> None:
    """Read trace paths from sys.argv and print stamped timing summaries.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("traces", nargs="+")
    args = p.parse_args()
    for path in args.traces:
        print(result_json(analyze(Path(path))))


if __name__ == "__main__":
    main()
