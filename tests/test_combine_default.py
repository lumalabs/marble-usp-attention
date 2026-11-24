# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""USPConfig picks a ring combine mode the GPU can run."""

from __future__ import annotations

import pytest
import torch

from usp_attention.usp import USPConfig


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), "online"),
        ((10, 0), "fp32"),
        ((10, 3), "fp32"),
        ((11, 0), "fp32"),
        ((8, 0), "flash"),
        ((12, 0), "flash"),
    ],
)
def test_default_combine_mode_per_arch(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], expected: str
) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    assert USPConfig().resolved_combine_mode("cuda:0") == expected


def test_explicit_combine_mode_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (10, 0)
    )
    assert USPConfig(combine_mode="online").resolved_combine_mode("cuda:0") == "online"


def test_rocm_and_cpu_do_not_pick_the_sm90_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.2")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 4))
    assert USPConfig().resolved_combine_mode("cuda:0") == "flash"
    assert USPConfig().resolved_combine_mode("cpu") == "flash"


@pytest.mark.parametrize(
    ("head_dim", "expected"),
    [
        (None, "online"),
        (128, "online"),
        (192, "online"),
        (256, "online"),
        (320, "python"),
    ],
)
def test_hopper_falls_back_above_online_head_dim(
    monkeypatch: pytest.MonkeyPatch, head_dim: int | None, expected: str
) -> None:
    # The online kernel covers FA4's whole SM90 range (head_dim <= 256). Beyond
    # it the resolver still answers "python", but FA4 itself rejects such heads
    # on SM90, so 320 only exercises the resolver, not a runnable fallback.
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    assert USPConfig().resolved_combine_mode("cuda:0", head_dim=head_dim) == expected


def test_explicit_online_rejects_wide_heads() -> None:
    with pytest.raises(ValueError, match="head_dim <= 256"):
        USPConfig(combine_mode="online").resolved_combine_mode("cuda:0", head_dim=320)


def test_blackwell_keeps_fp32_for_wide_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (10, 0)
    )
    assert USPConfig().resolved_combine_mode("cuda:0", head_dim=256) == "fp32"


def test_online_kernel_interface_rejects_wide_heads() -> None:
    pytest.importorskip("cutlass")
    from usp_attention.online_flash.interface import _validate_online_head_dims
    from usp_attention.types import ONLINE_MAX_HEAD_DIM

    _validate_online_head_dims(ONLINE_MAX_HEAD_DIM, ONLINE_MAX_HEAD_DIM, 8)
    with pytest.raises(ValueError, match="supports head_dim"):
        _validate_online_head_dims(ONLINE_MAX_HEAD_DIM + 8, 128, 8)
    with pytest.raises(ValueError, match="supports head_dim"):
        _validate_online_head_dims(128, ONLINE_MAX_HEAD_DIM + 8, 8)
