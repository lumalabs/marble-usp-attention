# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Regenerate one MLSys table from archived rows, with source row references.

This rebuilds the reported table; it does not rerun a GPU measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "benchmarks"))
import report_tables  # noqa: E402

TABLES = {
    "scale": report_tables.scale_section,
    "ablation": report_tables.ablation_section,
    "long-video": report_tables.video_section,
    "b300": report_tables.b300_section,
    "end-to-end": report_tables.e2e_section,
}


def main() -> None:
    """Print the selected paper table using CLI results-directory options.

    Raises:
        SystemExit: Argument parsing requests help or rejects invalid CLI
            options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=sorted(TABLES))
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="JSONL directory (default: the rows behind the paper's tables)",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=None,
        metavar="JSONL",
        help="more JSONL files or globs (default: the paper's gap-fill and"
        " MagiAttention reruns when --results is not given)",
    )
    args = parser.parse_args()
    extra = args.extra
    if args.results is None:
        report_tables.RESULTS = report_tables.DEFAULT_RESULTS
        if extra is None:
            extra = list(report_tables.PAPER_EXTRA)
    else:
        report_tables.RESULTS = str(args.results)
    rows = report_tables.load(extra=extra or ())
    if args.table != "end-to-end":
        rows = [r for r in rows if r.get("bench") != "model"]
    if not rows:
        parser.error("no measurement rows found")
    print("\n".join(TABLES[args.table](rows)))


if __name__ == "__main__":
    main()
