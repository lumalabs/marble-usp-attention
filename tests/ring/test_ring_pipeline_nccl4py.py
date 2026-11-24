# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from usp_attention.ring import ring_pipeline_nccl4py as nccl4py_ring
from usp_attention.ring.ring_pipeline_nccl4py import (
    NCCL4PyGradRing,
    NCCL4PyRingDirect,
    _format_version,
    _normalize_nccl_version,
    _peer_for_step,
    close_nccl4py_ring_buffers,
    destroy_all_nccl4py_ring_communicators,
)


@pytest.fixture(autouse=True)
def clear_nccl4py_ring_caches() -> Iterator[None]:
    nccl4py_ring._comm_cache.clear()
    nccl4py_ring._pair_buffer_cache.clear()
    nccl4py_ring._direct_cache.clear()
    nccl4py_ring._grad_cache.clear()
    yield
    nccl4py_ring._comm_cache.clear()
    nccl4py_ring._pair_buffer_cache.clear()
    nccl4py_ring._direct_cache.clear()
    nccl4py_ring._grad_cache.clear()


class TestNCCLVersionHelpers:
    @pytest.mark.parametrize(
        ("raw_version", "expected"),
        [
            (22907, (2, 29, 7)),
            (21905, (2, 19, 5)),
            ((2, 29, 7), (2, 29, 7)),
            ((2, 29), (2, 29, 0)),
            ([2, 29, 7], (2, 29, 7)),
            ([2, "29"], (2, 29, 0)),
        ],
    )
    def test_normalize_nccl_version(
        self, raw_version: object, expected: tuple[int, int, int]
    ) -> None:
        assert _normalize_nccl_version(raw_version) == expected

    @pytest.mark.parametrize("raw_version", ["2.29.7", object()])
    def test_normalize_nccl_version_rejects_unknown_format(
        self, raw_version: object
    ) -> None:
        with pytest.raises(TypeError, match="Unsupported NCCL version format"):
            _normalize_nccl_version(raw_version)

    def test_format_version(self) -> None:
        assert _format_version((2, 29, 7)) == "2.29.7"


class TestNCCL4PyVersionGate:
    def test_accepts_minimum_runtime_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nccl4py_ring, "nccl4py_version_tuple", lambda: (2, 29, 7))
        monkeypatch.setattr(
            nccl4py_ring, "torch_nccl_version_tuple", lambda: (2, 29, 7)
        )

        nccl4py_ring.require_nccl4py_one_sided()

    def test_rejects_old_nccl4py_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nccl4py_ring, "nccl4py_version_tuple", lambda: (2, 29, 6))
        monkeypatch.setattr(
            nccl4py_ring, "torch_nccl_version_tuple", lambda: (2, 29, 7)
        )

        with pytest.raises(RuntimeError, match="at least 2.29.7, got 2.29.6"):
            nccl4py_ring.require_nccl4py_one_sided()

    def test_allows_old_torch_nccl_when_nccl4py_runtime_is_new_enough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nccl4py_ring, "nccl4py_version_tuple", lambda: (2, 29, 7))
        monkeypatch.setattr(
            nccl4py_ring, "torch_nccl_version_tuple", lambda: (2, 29, 6)
        )

        nccl4py_ring.require_nccl4py_one_sided()

    def test_allows_unavailable_torch_nccl_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_torch_nccl_unavailable() -> tuple[int, int, int]:
            raise RuntimeError("torch nccl unavailable")

        monkeypatch.setattr(nccl4py_ring, "nccl4py_version_tuple", lambda: (2, 29, 7))
        monkeypatch.setattr(
            nccl4py_ring,
            "torch_nccl_version_tuple",
            raise_torch_nccl_unavailable,
        )

        nccl4py_ring.require_nccl4py_one_sided()


class TestRingStepHelpers:
    @pytest.mark.parametrize(
        ("rank", "step", "world_size", "expected"),
        [
            (0, 0, 4, 0),
            (0, 1, 4, 3),
            (0, 2, 4, 2),
            (2, 3, 4, 3),
            (4, 2, 8, 2),
        ],
    )
    def test_peer_for_step(
        self, rank: int, step: int, world_size: int, expected: int
    ) -> None:
        assert _peer_for_step(rank, step, world_size) == expected

    @pytest.mark.parametrize(
        ("step", "expected"),
        [
            (1, 0),
            (2, 1),
            (3, 0),
            (4, 1),
        ],
    )
    def test_direct_recv_slot_for_step(self, step: int, expected: int) -> None:
        assert NCCL4PyRingDirect._recv_slot_for_step(step) == expected

    @pytest.mark.parametrize(
        ("step", "expected"),
        [
            (0, 0),
            (1, 1),
            (2, 0),
            (3, 1),
        ],
    )
    def test_grad_slot_for_step(self, step: int, expected: int) -> None:
        assert NCCL4PyGradRing._slot_for_step(step) == expected

    def test_direct_source_and_consumer_for_step(self) -> None:
        direct = SimpleNamespace(rank=1, world_size=4)

        assert NCCL4PyRingDirect._source_for_step(direct, 1) == 0
        assert NCCL4PyRingDirect._source_for_step(direct, 2) == 3
        assert NCCL4PyRingDirect._consumer_for_step(direct, 1) == 2
        assert NCCL4PyRingDirect._consumer_for_step(direct, 2) == 3

    @pytest.mark.parametrize("step", [0, 4])
    def test_direct_source_rejects_invalid_step(self, step: int) -> None:
        direct = SimpleNamespace(rank=1, world_size=4)

        with pytest.raises(ValueError, match="Expected step"):
            NCCL4PyRingDirect._source_for_step(direct, step)

    @pytest.mark.parametrize("step", [0, 4])
    def test_direct_consumer_rejects_invalid_step(self, step: int) -> None:
        direct = SimpleNamespace(rank=1, world_size=4)

        with pytest.raises(ValueError, match="Expected step"):
            NCCL4PyRingDirect._consumer_for_step(direct, step)


class _FakePairBuffers:
    def __init__(self, comm: object):
        self.comm = comm
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeComm:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


class TestNCCL4PyCacheCleanup:
    def test_close_ring_buffers_clears_matching_transport_caches(self) -> None:
        comm = object()
        other_comm = object()
        buffers = _FakePairBuffers(comm)
        other_buffers = _FakePairBuffers(other_comm)
        nccl4py_ring._pair_buffer_cache[("buffers",)] = buffers
        nccl4py_ring._pair_buffer_cache[("other_buffers",)] = other_buffers
        nccl4py_ring._direct_cache[("direct",)] = SimpleNamespace(comm=comm)
        nccl4py_ring._direct_cache[("other_direct",)] = SimpleNamespace(comm=other_comm)
        nccl4py_ring._grad_cache[("grad",)] = SimpleNamespace(comm=comm)
        nccl4py_ring._grad_cache[("other_grad",)] = SimpleNamespace(comm=other_comm)

        close_nccl4py_ring_buffers(comm)

        assert buffers.closed
        assert not other_buffers.closed
        assert list(nccl4py_ring._pair_buffer_cache) == [("other_buffers",)]
        assert list(nccl4py_ring._direct_cache) == [("other_direct",)]
        assert list(nccl4py_ring._grad_cache) == [("other_grad",)]

    def test_destroy_all_communicators_closes_windows_and_destroys_comms(self) -> None:
        comm_a = _FakeComm()
        comm_b = _FakeComm()
        buffers_a = _FakePairBuffers(comm_a)
        buffers_b = _FakePairBuffers(comm_b)
        nccl4py_ring._comm_cache["a"] = comm_a
        nccl4py_ring._comm_cache["b"] = comm_b
        nccl4py_ring._pair_buffer_cache[("a",)] = buffers_a
        nccl4py_ring._pair_buffer_cache[("b",)] = buffers_b
        nccl4py_ring._direct_cache[("a",)] = SimpleNamespace(comm=comm_a)
        nccl4py_ring._grad_cache[("b",)] = SimpleNamespace(comm=comm_b)

        destroy_all_nccl4py_ring_communicators()

        assert buffers_a.closed
        assert buffers_b.closed
        assert comm_a.destroyed
        assert comm_b.destroyed
        assert nccl4py_ring._comm_cache == {}
        assert nccl4py_ring._pair_buffer_cache == {}
        assert nccl4py_ring._direct_cache == {}
        assert nccl4py_ring._grad_cache == {}
