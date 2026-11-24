# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for measurement provenance and the GPU accuracy gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "benchmarks"))
sys.path.insert(0, str(PROJECT / "repro"))
from check_exactness import cases, validate_rows  # noqa: E402
from provenance import git_state, result_json, source_revision, stamp_row  # noqa: E402


def test_worker_provenance_is_not_replaced() -> None:
    worker = {"git": {"sha": "worker", "dirty": False}}
    assert stamp_row({"provenance": worker, "median_ms": 3})["provenance"] is worker
    row = json.loads(result_json({"median_ms": 3}))
    assert row["median_ms"] == 3
    assert "dirty" in row["provenance"]["git"]
    assert "nvidia-cutlass-dsl" in row["provenance"]["packages"]
    assert "driver" in row["provenance"]


def test_git_stamp_tracks_uncommitted_and_untracked_sources(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        ],
        check=True,
    )
    assert git_state(tmp_path)["dirty"] is False
    (tmp_path / "new.py").write_text("# untracked source\n")
    assert git_state(tmp_path)["dirty"] is True
    assert git_state(tmp_path)["sha"]


def _rows() -> list[dict]:
    return [
        {
            "check": check,
            "reference": ref,
            **{
                f"{name}_{metric}": 0.002
                for name in ("out", "dq", "dk", "dv")
                for metric in ("mean_rel", "mean_abs", "max_abs")
            },
        }
        for check, ref in (("usp", "fp32"), ("usp", "fa4"), ("fa4_single_gpu", "fp32"))
    ]


def test_exactness_gate_rejects_gradient_regressions_and_missing_references() -> None:
    validate_rows(_rows())
    rows = _rows()
    rows[0]["dk_mean_rel"] *= 1.2
    with pytest.raises(ValueError, match="dk"):
        validate_rows(rows)
    with pytest.raises(ValueError, match="expected"):
        validate_rows(_rows()[:2])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_exactness_gate_rejects_nonfinite_values(value: float) -> None:
    rows = _rows()
    rows[1]["out_max_abs"] = value
    with pytest.raises(ValueError, match="nonfinite"):
        validate_rows(rows)


def test_exactness_matrix_covers_hardware_specific_combine() -> None:
    for gpu, combine in (("h100", "online"), ("b300", "fp32")):
        matrix = dict(cases(gpu, 8))
        assert combine in matrix["mixed-r8"]
        assert "--exact-dense-ring" in matrix["dense-ring-d128"]
        assert "--sliding-window" in matrix["window"]
        flags = matrix["head256"]
        assert flags[flags.index("--kv-exchange") + 1] == (
            "gather" if gpu == "h100" else "ring"
        )
        for name, flags in matrix.items():
            # FA4 preserves fp32 dK/dV partials only for GQA on ring paths.
            kv = flags[len(flags) - 1 - flags[::-1].index("--kv-heads") + 1]
            assert kv == ("8" if name.endswith("-r1") else "2")


def test_submitted_revision_takes_precedence_over_container_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha = "a" * 40
    monkeypatch.setenv("MARBLE_GIT_COMMIT", sha + "-dirty")
    assert source_revision()["sha"] == sha
    assert source_revision()["dirty"] is True
    assert source_revision()["source"] == "submitted_tree"


@pytest.mark.parametrize("failed", [False, True])
def test_sweep_preserves_worker_stamp_and_marks_partial_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failed: bool
) -> None:
    import sweep

    output = tmp_path / "rows.jsonl"
    row = {"label": "test", "median_ms": 1.0, "provenance": {"git": {"sha": "worker"}}}
    monkeypatch.setattr(sweep, "_sweeps", lambda: {"smoke": [("test", [])]})
    monkeypatch.setattr(sys, "argv", ["sweep.py", "smoke", "--out", str(output)])
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=int(failed),
            stdout="RESULT " + json.dumps(row),
            stderr="",
        ),
    )
    sweep.main()
    saved = json.loads(output.read_text())
    assert saved["provenance"]["git"]["sha"] == "worker"
    assert "launch_command" in saved
    assert ("error" in saved) is failed


def test_replay_resolves_every_reported_pack_and_baseline_driver() -> None:
    import replay

    from usp_attention.layouts import LAYOUT_NAMES, build_layout, load_layout_file

    assert replay.BUILTIN_LAYOUTS == LAYOUT_NAMES
    replay.report.RESULTS = str(PROJECT / "benchmarks/results/mlsys")
    rows = replay.report.load()
    assert rows
    for row in rows:
        if row.get("impl") != "torch_flex_allgather":
            command = replay.measurement_command(row)
            layout = command[command.index("--layout") + 1]
            if row["layout"] in LAYOUT_NAMES:
                assert layout == row["layout"]
                docs, _, _ = build_layout(
                    layout, row["seq_len"], ring_size=row.get("ring", 1)
                )
                assert sum(docs) == row["seq_len"]
                continue
            spec = layout.removeprefix("file:").split("#")[0]
            load_layout_file(str(PROJECT / spec) + "#0", row["seq_len"])
            assert (
                PROJECT / (layout.removeprefix("file:").split("#")[0] + ".gz")
            ).is_file()
    ablation = [r for r in rows if replay.selected(r, "ablation")]
    assert ablation
    assert {r["kv_exchange"] for r in ablation} == {"ring", "gather"}
    assert {r["balance"] for r in ablation} == {"ptrr", "chunks"}


def test_replay_missing_pack_is_not_treated_as_a_builtin() -> None:
    import replay

    with pytest.raises(ValueError, match="missing committed pack"):
        replay.measurement_command({"layout": "file:missing/vlm.json#0"})


def test_case_filter_rejects_cases_for_a_different_mesh(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "repro/check_exactness.py"),
            "--gpu",
            "h100",
            "--nproc",
            "4",
            "--case",
            "causal-r8",
            "--dry-run",
            "--out",
            str(tmp_path / "unused"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unknown cases" in result.stderr
    assert not (tmp_path / "unused").exists()
