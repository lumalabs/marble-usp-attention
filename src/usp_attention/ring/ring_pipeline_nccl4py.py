# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""NCCL4Py one-sided ring pipeline for KV and gradient exchange.

This module is a drop-in transport sibling of :mod:`ring_pipeline_symm`,
but uses NCCL4Py RMA windows instead of ``torch.distributed._symmetric_memory``.
The forward KV path follows a receiver-driven direct-send schedule:

* rank ``r`` signals the owner of step ``s`` that receive slot
  ``(s - 1) % 2`` is free;
* the owner ``(r - s + W) % W`` pushes K/V into that slot with
  ``put_signal``;
* rank ``r`` waits for the two K/V arrival signals before running the step.

The dk/dv accumulator path still rotates next/prev like the standard ring
backward. It uses direct pushes into alternating receive slots plus DONE
signals so slots are not reused before the consumer has copied them into local
staging.

Arrival waits and local bookkeeping run on the caller's current CUDA stream.
The owner's "slot free" wait and its one-sided ``put_signal`` push run on a
dedicated send stream, in that order; callers only join that stream in
``finalize`` (before the send slot is restaged), because the receiver's
arrival signal is the dependency that matters for subsequent attention work.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist

loguru = logging.getLogger(__name__)
from torch import Tensor

from usp_attention.ring.ring_pipeline import (  # noqa: F401
    PendingGradExchange as PendingGradExchange,
)
from usp_attention.ring.ring_pipeline import (
    RingStepCallback as RingStepCallback,
)
from usp_attention.ring.ring_pipeline import (
    RingStepKernel as RingStepKernel,
)
from usp_attention.ring.ring_pipeline import (
    _launch_async_grad_rotation as _launch_async_grad_rotation,
)
from usp_attention.ring.ring_pipeline import (
    _wait_pending_grad_exchange as _wait_pending_grad_exchange,
)
from usp_attention.ring.ring_planning import RankRingPlan

if TYPE_CHECKING:
    import nccl.core as nccl
    from nccl.core import Communicator
    from nccl.core.interop import torch as nccl_torch
else:
    nccl: Any | None = None
    nccl_torch: Any | None = None

_MIN_NCCL4PY_ONE_SIDED_VERSION = (2, 29, 7)
_NCCL4PY_RING_DEBUG = os.environ.get("NCCL4PY_RING_DEBUG", "0").lower() in {
    "1",
    "true",
    "yes",
}

NCCL4PyBufferMode = Literal["direct"]

_comm_cache: dict[tuple[object, str], Any] = {}
_pair_buffer_cache: dict[tuple[object, ...], NCCL4PyPairBuffers] = {}
_direct_cache: dict[tuple[object, ...], NCCL4PyRingDirect] = {}
_grad_cache: dict[tuple[object, ...], NCCL4PyGradRing] = {}


def _log(rank: int, msg: str) -> None:
    if not _NCCL4PY_RING_DEBUG:
        return
    t = time.perf_counter()
    global_rank: int | None = None
    if dist.is_initialized():
        try:
            global_rank = dist.get_rank()
        except RuntimeError:
            global_rank = None
    rank_str = f"rank={rank}"
    if global_rank is not None:
        rank_str += f" global_rank={global_rank}"
    loguru.info(f"[nccl4py_ring {rank_str} t={t:.6f}] {msg}")


def _require_nccl4py() -> None:
    global nccl, nccl_torch

    if nccl is not None and nccl_torch is not None:
        return

    try:
        import nccl.core as loaded_nccl
        from nccl.core.interop import torch as loaded_nccl_torch
    except ImportError as exc:
        raise RuntimeError(
            "use_nccl4py_ring=True requires nccl4py to be installed in the "
            "runtime environment."
        ) from exc

    nccl = loaded_nccl
    nccl_torch = loaded_nccl_torch


def _normalize_nccl_version(raw_version: object) -> tuple[int, int, int]:
    if isinstance(raw_version, int):
        return (
            raw_version // 10000,
            (raw_version % 10000) // 100,
            raw_version % 100,
        )
    if isinstance(raw_version, tuple):
        padded = tuple(int(part) for part in raw_version) + (0, 0, 0)
        return padded[:3]
    if isinstance(raw_version, list):
        padded = tuple(int(part) for part in raw_version) + (0, 0, 0)
        return padded[:3]
    raise TypeError(f"Unsupported NCCL version format: {raw_version!r}.")


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def torch_nccl_version_tuple() -> tuple[int, int, int]:
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "nccl"):
        raise RuntimeError("torch.cuda.nccl is unavailable; cannot use NCCL4Py RMA.")
    try:
        return _normalize_nccl_version(torch.cuda.nccl.version())
    except Exception as exc:
        raise RuntimeError("Failed to query torch.cuda.nccl.version().") from exc


def nccl4py_version_tuple() -> tuple[int, int, int]:
    _require_nccl4py()
    version = nccl.get_version().nccl_version
    return (version.major, version.minor, version.micro)


def require_nccl4py_one_sided() -> None:
    """Validate that NCCL4Py loaded a NCCL runtime with RMA APIs."""

    runtime_version = nccl4py_version_tuple()
    if runtime_version < _MIN_NCCL4PY_ONE_SIDED_VERSION:
        raise RuntimeError(
            "NCCL4Py one-sided ring communication requires NCCL4Py's loaded "
            "NCCL runtime to be at least "
            f"{_format_version(_MIN_NCCL4PY_ONE_SIDED_VERSION)}, got "
            f"{_format_version(runtime_version)} from nccl4py."
        )

    try:
        torch_version = torch_nccl_version_tuple()
    except RuntimeError as exc:
        loguru.warning(f"Could not query torch.cuda.nccl.version(): {exc}")
        return
    if torch_version < _MIN_NCCL4PY_ONE_SIDED_VERSION:
        loguru.warning(
            "PyTorch is bound to NCCL "
            f"{_format_version(torch_version)}, but NCCL4Py loaded runtime NCCL "
            f"{_format_version(runtime_version)}. Allowing NCCL4Py RMA because "
            "the one-sided APIs are provided by the NCCL4Py runtime."
        )


def _group_key(process_group: dist.ProcessGroup) -> object:
    return getattr(process_group, "group_name", None) or id(process_group)


def init_nccl4py_ring_communicator(
    process_group: dist.ProcessGroup,
    *,
    tag: str,
) -> "Communicator":
    """Create or return a cached NCCL4Py communicator for ``process_group``.

    NCCL4Py one-sided signals share one signal lane per communicator.  Forward
    direct KV, backward direct KV, and dk/dv grad rotation use different signal
    protocols, so they must not share a communicator or stale signals from one
    protocol can satisfy waits in another.
    """

    _require_nccl4py()
    require_nccl4py_one_sided()

    key = (_group_key(process_group), tag)
    cached = _comm_cache.get(key)
    if cached is not None:
        return cached

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    id_list: list[Any] = [nccl.get_unique_id() if rank == 0 else None]
    src_global_rank = dist.get_global_rank(process_group, 0)
    dist.broadcast_object_list(
        id_list,
        src=src_global_rank,
        group=process_group,
    )
    unique_id = id_list[0]
    if unique_id is None:
        raise RuntimeError("Failed to broadcast NCCL4Py communicator unique ID")

    comm = nccl.Communicator.init(
        nranks=world_size,
        rank=rank,
        unique_id=unique_id,
    )
    _comm_cache[key] = comm
    loguru.info(
        "Initialized NCCL4Py ring communicator for group "
        f"{key!r} rank={rank}/{world_size} torch_nccl="
        f"{_format_version(torch_nccl_version_tuple())} nccl4py_runtime="
        f"{_format_version(nccl4py_version_tuple())}"
    )
    return comm


def get_nccl4py_ring_communicator_for_tag(
    process_group: dist.ProcessGroup,
    *,
    tag: str,
) -> "Communicator":
    cached = _comm_cache.get((_group_key(process_group), tag))
    if cached is not None:
        return cached
    return init_nccl4py_ring_communicator(process_group, tag=tag)


def destroy_nccl4py_ring_communicator(process_group: dist.ProcessGroup) -> None:
    group_key = _group_key(process_group)
    comms_to_destroy = []
    for key, comm in list(_comm_cache.items()):
        if key[0] != group_key:
            continue
        _comm_cache.pop(key)
        comms_to_destroy.append(comm)
    close_nccl4py_ring_buffers_for_comms(comms_to_destroy)
    for comm in comms_to_destroy:
        comm.destroy()


def destroy_all_nccl4py_ring_communicators() -> None:
    """Close cached NCCL4Py windows and communicators for process teardown.

    Long-lived workers should call this when the process group layout is no
    longer valid so subsequent jobs cannot reuse stale NCCL4Py RMA windows.
    """

    comms_to_destroy = list(_comm_cache.values())
    _comm_cache.clear()
    close_nccl4py_ring_buffers_for_comms(comms_to_destroy)
    for comm in comms_to_destroy:
        comm.destroy()


def _peer_for_step(rank: int, step: int, world_size: int) -> int:
    return (rank - step + world_size) % world_size


@dataclass(kw_only=True, slots=True)
class NCCL4PyPairBuffers:
    """Registered NCCL4Py windows for a pair of same-shaped tensors."""

    comm: Any
    mode: NCCL4PyBufferMode
    tensor_shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    numel: int
    num_send_slots: int
    num_recv_slots: int
    send_first_slots: list[Tensor]
    send_second_slots: list[Tensor]
    recv_first_slots: Tensor
    recv_second_slots: Tensor
    send_first_windows: list[Any]
    send_second_windows: list[Any]
    recv_first_window: Any
    recv_second_window: Any
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        for window in self.send_first_windows:
            window.close()
        for window in self.send_second_windows:
            window.close()
        self.recv_first_window.close()
        self.recv_second_window.close()
        self.closed = True


def _allocate_pair_buffers(
    *,
    comm: "Communicator",
    tensor_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    mode: NCCL4PyBufferMode,
    num_send_slots: int,
    num_recv_slots: int,
) -> NCCL4PyPairBuffers:
    _require_nccl4py()

    send_first_slots = [
        nccl_torch.empty(tensor_shape, dtype=dtype, device=device)
        for _ in range(num_send_slots)
    ]
    send_second_slots = [
        nccl_torch.empty(tensor_shape, dtype=dtype, device=device)
        for _ in range(num_send_slots)
    ]
    recv_first_slots = nccl_torch.empty(
        (num_recv_slots, *tensor_shape),
        dtype=dtype,
        device=device,
    )
    recv_second_slots = nccl_torch.empty(
        (num_recv_slots, *tensor_shape),
        dtype=dtype,
        device=device,
    )

    send_first_windows = [
        comm.register_window(slot, nccl.WindowFlag.CollSymmetric)
        for slot in send_first_slots
    ]
    send_second_windows = [
        comm.register_window(slot, nccl.WindowFlag.CollSymmetric)
        for slot in send_second_slots
    ]
    recv_first_window = comm.register_window(
        recv_first_slots,
        nccl.WindowFlag.CollSymmetric,
    )
    recv_second_window = comm.register_window(
        recv_second_slots,
        nccl.WindowFlag.CollSymmetric,
    )
    if (
        any(window is None for window in send_first_windows)
        or any(window is None for window in send_second_windows)
        or recv_first_window is None
        or recv_second_window is None
    ):
        raise RuntimeError("NCCL4Py register_window returned None.")

    return NCCL4PyPairBuffers(
        comm=comm,
        mode=mode,
        tensor_shape=tensor_shape,
        dtype=dtype,
        device=device,
        numel=tensor_shape.numel(),
        num_send_slots=num_send_slots,
        num_recv_slots=num_recv_slots,
        send_first_slots=send_first_slots,
        send_second_slots=send_second_slots,
        recv_first_slots=recv_first_slots,
        recv_second_slots=recv_second_slots,
        send_first_windows=send_first_windows,
        send_second_windows=send_second_windows,
        recv_first_window=recv_first_window,
        recv_second_window=recv_second_window,
    )


def _pair_buffer_key(
    *,
    comm: "Communicator",
    process_group: dist.ProcessGroup,
    tensor_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    mode: NCCL4PyBufferMode,
    num_send_slots: int,
    num_recv_slots: int,
    tag: str,
) -> tuple[object, ...]:
    return (
        mode,
        id(comm),
        _group_key(process_group),
        dist.get_rank(process_group),
        dist.get_world_size(process_group),
        tuple(tensor_shape),
        dtype,
        str(device),
        num_send_slots,
        num_recv_slots,
        tag,
    )


def _get_or_create_pair_buffers(
    *,
    comm: "Communicator",
    process_group: dist.ProcessGroup,
    tensor_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    mode: NCCL4PyBufferMode,
    num_send_slots: int,
    num_recv_slots: int,
    tag: str,
) -> NCCL4PyPairBuffers:
    key = _pair_buffer_key(
        comm=comm,
        process_group=process_group,
        tensor_shape=tensor_shape,
        dtype=dtype,
        device=device,
        mode=mode,
        num_send_slots=num_send_slots,
        num_recv_slots=num_recv_slots,
        tag=tag,
    )
    cached = _pair_buffer_cache.get(key)
    if cached is not None and not cached.closed:
        return cached

    buffers = _allocate_pair_buffers(
        comm=comm,
        tensor_shape=tensor_shape,
        dtype=dtype,
        device=device,
        mode=mode,
        num_send_slots=num_send_slots,
        num_recv_slots=num_recv_slots,
    )
    _pair_buffer_cache[key] = buffers
    return buffers


def close_nccl4py_ring_buffers(comm: "Communicator" | None = None) -> None:
    close_nccl4py_ring_buffers_for_comms(None if comm is None else [comm])


def close_nccl4py_ring_buffers_for_comms(comms: list[object] | None = None) -> None:
    comm_ids = None if comms is None else {id(comm) for comm in comms}
    for key, direct in list(_direct_cache.items()):
        if comm_ids is not None and id(direct.comm) not in comm_ids:
            continue
        _direct_cache.pop(key)
    for key, grad_ring in list(_grad_cache.items()):
        if comm_ids is not None and id(grad_ring.comm) not in comm_ids:
            continue
        _grad_cache.pop(key)
        close = getattr(grad_ring, "close", None)
        if close is not None:
            close()
    for key, buffers in list(_pair_buffer_cache.items()):
        if comm_ids is not None and id(buffers.comm) not in comm_ids:
            continue
        _pair_buffer_cache.pop(key)
        buffers.close()


class NCCL4PyRingDirect:
    """Direct owner-to-consumer push transport for forward/backward KV."""

    def __init__(
        self,
        kv_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        process_group: dist.ProcessGroup,
        comm_stream: torch.cuda.Stream | None = None,
        *,
        tag: str = "direct",
    ):
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.device = device
        self.comm: "Communicator" = get_nccl4py_ring_communicator_for_tag(
            process_group,
            tag=f"{tag}:data",
        )
        self._ctrl_comm: "Communicator" = get_nccl4py_ring_communicator_for_tag(
            process_group,
            tag=f"{tag}:ctrl",
        )
        self._comm_stream = (
            comm_stream if comm_stream is not None else torch.cuda.Stream(device=device)
        )
        self._published = False
        self._stage_event: torch.cuda.Event | None = None
        self._pending_step: int | None = None

        self._buffers = _get_or_create_pair_buffers(
            comm=self.comm,
            process_group=process_group,
            tensor_shape=kv_shape,
            dtype=dtype,
            device=device,
            mode="direct",
            num_send_slots=1,
            num_recv_slots=2,
            tag=tag,
        )
        self._validate_buffers(kv_shape, dtype, device)

    def _validate_buffers(
        self,
        kv_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self._buffers.closed:
            raise ValueError("Cannot use closed NCCL4Py direct buffers.")
        if self._buffers.tensor_shape != kv_shape:
            raise ValueError(
                f"NCCL4Py direct buffer shape mismatch: "
                f"{self._buffers.tensor_shape=} {kv_shape=}."
            )
        if self._buffers.dtype != dtype or self._buffers.device != device:
            raise ValueError(
                "NCCL4Py direct buffer dtype/device mismatch: "
                f"{self._buffers.dtype=} {dtype=} "
                f"{self._buffers.device=} {device=}."
            )

    def _source_for_step(self, step: int) -> int:
        if step <= 0 or step >= self.world_size:
            raise ValueError(f"Expected step in [1, {self.world_size}), got {step=}.")
        return _peer_for_step(self.rank, step, self.world_size)

    def _consumer_for_step(self, step: int) -> int:
        if step <= 0 or step >= self.world_size:
            raise ValueError(f"Expected step in [1, {self.world_size}), got {step=}.")
        return (self.rank + step) % self.world_size

    @staticmethod
    def _recv_slot_for_step(step: int) -> int:
        return (step - 1) % 2

    def initial_publish(self, k: Tensor, v: Tensor) -> None:
        if not k.is_cuda or not v.is_cuda:
            raise ValueError("NCCL4Py direct ring expects CUDA tensors.")
        if not k.is_contiguous() or not v.is_contiguous():
            raise ValueError("NCCL4Py direct ring expects contiguous K/V tensors.")
        if tuple(k.shape) != tuple(self._buffers.tensor_shape):
            raise ValueError(
                f"NCCL4Py direct ring shape mismatch: {k.shape=} "
                f"{self._buffers.tensor_shape=}."
            )

        self._buffers.send_first_slots[0].copy_(k)
        self._buffers.send_second_slots[0].copy_(v)
        self._stage_event = torch.cuda.Event()
        self._stage_event.record(torch.cuda.current_stream(device=self.device))
        self._published = True
        _log(self.rank, "NCCL4PyRingDirect.initial_publish")

    def prepare_prefetch(self, step: int) -> torch.cuda.Event:
        owner = self._source_for_step(step)
        current_stream = torch.cuda.current_stream(device=self.device)
        self._ctrl_comm.signal(owner, stream=current_stream.cuda_stream)
        event = torch.cuda.Event()
        event.record(current_stream)
        self._pending_step = step
        _log(
            self.rank,
            f"NCCL4PyRingDirect.prepare_prefetch: step={step} owner={owner}",
        )
        return event

    def start_prefetch(
        self, step: int, _ready_event: torch.cuda.Event | None = None
    ) -> None:
        del _ready_event
        if not self._published:
            raise RuntimeError("start_prefetch called before initial_publish.")
        if self._stage_event is None:
            raise RuntimeError("start_prefetch called before initial_publish.")

        _require_nccl4py()
        consumer = self._consumer_for_step(step)
        consumer_slot = self._recv_slot_for_step(step)
        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(self._stage_event)
            # Wait for the consumer's "slot free" signal on the push stream
            # itself, so the put cannot overwrite a slot the consumer is still
            # reading (and the owner's compute stream is not stalled on it).
            self._ctrl_comm.wait_signal(
                nccl.WaitSignalDesc(consumer),
                stream=self._comm_stream.cuda_stream,
            )
            self.comm.put_signal(
                self._buffers.send_first_slots[0],
                consumer,
                self._buffers.recv_first_window,
                peer_window_offset=consumer_slot * self._buffers.numel,
                stream=self._comm_stream.cuda_stream,
            )
            self.comm.put_signal(
                self._buffers.send_second_slots[0],
                consumer,
                self._buffers.recv_second_window,
                peer_window_offset=consumer_slot * self._buffers.numel,
                stream=self._comm_stream.cuda_stream,
            )
        _log(
            self.rank,
            f"NCCL4PyRingDirect.start_prefetch: step={step} consumer={consumer}",
        )

    def wait_prefetch(self) -> None:
        if self._pending_step is None:
            raise RuntimeError("wait_prefetch called before prepare_prefetch.")
        _require_nccl4py()
        source = self._source_for_step(self._pending_step)
        current_stream = torch.cuda.current_stream(device=self.device)
        self.comm.wait_signal(
            nccl.WaitSignalDesc(source, op_count=2),
            stream=current_stream.cuda_stream,
        )
        _log(
            self.rank,
            f"NCCL4PyRingDirect.wait_prefetch: step={self._pending_step} "
            f"source={source}",
        )

    def ack_prefetch(self, _step: int) -> None:
        return None

    def staging(self, step: int) -> tuple[Tensor, Tensor]:
        idx = self._recv_slot_for_step(step)
        return self._buffers.recv_first_slots[idx], self._buffers.recv_second_slots[idx]

    def finalize(self) -> None:
        torch.cuda.current_stream(device=self.device).wait_stream(self._comm_stream)
        self._pending_step = None
        _log(self.rank, "NCCL4PyRingDirect.finalize")


def _get_or_create_direct(
    kv_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream | None,
    *,
    tag: str = "direct",
) -> NCCL4PyRingDirect:
    key = (
        tuple(kv_shape),
        dtype,
        str(device),
        _group_key(process_group),
        dist.get_rank(process_group),
        tag,
    )
    cached = _direct_cache.get(key)
    if cached is not None:
        return cached
    direct = NCCL4PyRingDirect(
        kv_shape=kv_shape,
        dtype=dtype,
        device=device,
        process_group=process_group,
        comm_stream=comm_stream,
        tag=tag,
    )
    _direct_cache[key] = direct
    return direct


class NCCL4PyGradRing:
    """NCCL4Py one-sided dk/dv accumulator rotation.

    The grad ring sends dk/dv as one stacked registered buffer and one
    ``put_signal``.  Keeping the two tensors under one signal avoids a subtle
    race where a receiver can observe the first signal and copy before the
    second tensor's signal/data is visible on some NCCL4Py runtimes.
    """

    def __init__(
        self,
        grad_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        process_group: dist.ProcessGroup,
        comm_stream: torch.cuda.Stream | None = None,
    ):
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.device = device
        self._prev_rank = (self.rank - 1 + self.world_size) % self.world_size
        self._next_rank = (self.rank + 1) % self.world_size
        self.comm = get_nccl4py_ring_communicator_for_tag(
            process_group,
            tag="grad:data",
        )
        self._ctrl_comm = get_nccl4py_ring_communicator_for_tag(
            process_group,
            tag="grad:ctrl",
        )
        self._comm_stream = (
            comm_stream if comm_stream is not None else torch.cuda.Stream(device=device)
        )
        _require_nccl4py()
        self._numel = grad_shape.numel()
        self._send_slots = [
            nccl_torch.empty((2, *grad_shape), dtype=dtype, device=device)
            for _ in range(2)
        ]
        self._recv_slots = nccl_torch.empty(
            (2, 2, *grad_shape),
            dtype=dtype,
            device=device,
        )
        self._send_windows = [
            self.comm.register_window(slot, nccl.WindowFlag.CollSymmetric)
            for slot in self._send_slots
        ]
        self._recv_window = self.comm.register_window(
            self._recv_slots,
            nccl.WindowFlag.CollSymmetric,
        )
        if any(window is None for window in self._send_windows):
            raise RuntimeError("NCCL4Py grad send register_window returned None.")
        if self._recv_window is None:
            raise RuntimeError("NCCL4Py grad recv register_window returned None.")
        self._staging_first = [
            torch.empty(grad_shape, dtype=dtype, device=device),
            torch.empty(grad_shape, dtype=dtype, device=device),
        ]
        self._staging_second = [
            torch.empty(grad_shape, dtype=dtype, device=device),
            torch.empty(grad_shape, dtype=dtype, device=device),
        ]
        self._last_read_step: int | None = None
        self._outstanding_sends = 0
        self._closed = False

    @staticmethod
    def _slot_for_step(step: int) -> int:
        return step % 2

    def _wait_next_done(self) -> None:
        _require_nccl4py()
        current_stream = torch.cuda.current_stream(device=self.device)
        self._ctrl_comm.wait_signal(
            nccl.WaitSignalDesc(self._next_rank),
            stream=current_stream.cuda_stream,
        )
        self._outstanding_sends -= 1

    def _publish_to_next(self, first: Tensor, second: Tensor, step: int) -> None:
        if self._outstanding_sends >= len(self._send_slots):
            self._wait_next_done()

        slot = self._slot_for_step(step)
        self._send_slots[slot][0].copy_(first, non_blocking=True)
        self._send_slots[slot][1].copy_(second, non_blocking=True)
        staged = torch.cuda.Event()
        staged.record(torch.cuda.current_stream(device=self.device))

        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(staged)
            self.comm.put_signal(
                self._send_slots[slot],
                self._next_rank,
                self._recv_window,
                peer_window_offset=slot * 2 * self._numel,
                stream=self._comm_stream.cuda_stream,
            )
        self._outstanding_sends += 1
        _log(
            self.rank,
            f"NCCL4PyGradRing.publish: step={step} slot={slot} next={self._next_rank}",
        )

    def initial_publish(self, dk: Tensor, dv: Tensor) -> None:
        self._publish_to_next(dk, dv, step=0)

    def republish(self, dk: Tensor, dv: Tensor, step: int) -> None:
        self._publish_to_next(dk, dv, step=step)

    def prepare_read_prev(self, step: int) -> torch.cuda.Event:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self.device))
        self._last_read_step = step
        return event

    def start_read_prev(
        self, step: int, _ready_event: torch.cuda.Event | None = None
    ) -> None:
        del _ready_event
        self._last_read_step = step

    def wait_read(self) -> tuple[Tensor, Tensor]:
        """Wait for the predecessor's dk / dv and return them.

        The returned tensors are this ring's staging buffers: later reads, and
        later backward calls (the ring is cached per shape), overwrite them.
        Copy them to keep them.

        Raises:
            RuntimeError: If called before :meth:`start_read_prev`.
        """
        if self._last_read_step is None:
            raise RuntimeError("wait_read called before start_read_prev.")
        _require_nccl4py()
        recv_idx = self._slot_for_step(self._last_read_step)
        current_stream = torch.cuda.current_stream(device=self.device)
        self.comm.wait_signal(
            nccl.WaitSignalDesc(self._prev_rank),
            stream=current_stream.cuda_stream,
        )
        self._staging_first[recv_idx].copy_(
            self._recv_slots[recv_idx, 0],
        )
        self._staging_second[recv_idx].copy_(
            self._recv_slots[recv_idx, 1],
        )
        return self._staging_first[recv_idx], self._staging_second[recv_idx]

    def ack_read_prev(self, _step: int | None = None) -> None:
        if _step is None:
            if self._last_read_step is None:
                raise RuntimeError("ack_read_prev called before start_read_prev.")
        current_stream = torch.cuda.current_stream(device=self.device)
        self._ctrl_comm.signal(self._prev_rank, stream=current_stream.cuda_stream)

    def finalize(self, _step: int) -> None:
        while self._outstanding_sends > 0:
            self._wait_next_done()

    def close(self) -> None:
        if self._closed:
            return
        for window in self._send_windows:
            window.close()
        self._recv_window.close()
        self._closed = True


def _get_or_create_grad_ring(
    grad_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream | None,
) -> NCCL4PyGradRing:
    key = (
        tuple(grad_shape),
        dtype,
        str(device),
        _group_key(process_group),
        dist.get_rank(process_group),
        "grad",
    )
    cached = _grad_cache.get(key)
    if cached is not None:
        return cached
    grad_ring = NCCL4PyGradRing(
        grad_shape=grad_shape,
        dtype=dtype,
        device=device,
        process_group=process_group,
        comm_stream=comm_stream,
    )
    _grad_cache[key] = grad_ring
    return grad_ring


def run_async_ring_kv_pipeline_nccl4py_direct(
    *,
    initial_key: Tensor,
    initial_value: Tensor,
    process_group: dist.ProcessGroup,
    step_callback: RingStepCallback,
    rank_plan: RankRingPlan | None = None,
    use_cuda_comm_stream: bool = True,
    comm_stream: torch.cuda.Stream | None = None,
) -> None:
    """Direct-send forward ring pipeline using NCCL4Py RMA windows."""

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before ring pipeline."
        )
    world_size = dist.get_world_size(process_group)
    if world_size <= 0:
        raise ValueError(f"Invalid process group world size: {world_size=}.")
    if world_size == 1:
        step_plan = rank_plan.steps[0] if rank_plan is not None else None
        step_callback(0, initial_key, initial_value, step_plan)
        return
    if rank_plan is not None and len(rank_plan.steps) != world_size:
        raise ValueError(
            f"rank_plan must have one step per rank: {len(rank_plan.steps)=}, "
            f"{world_size=}."
        )
    if initial_key.shape != initial_value.shape:
        raise ValueError(
            "NCCL4Py direct ring currently expects matching K/V shapes, got "
            f"{initial_key.shape=} {initial_value.shape=}."
        )
    if initial_key.dtype != initial_value.dtype:
        raise ValueError(
            "NCCL4Py direct ring currently expects matching K/V dtypes, got "
            f"{initial_key.dtype=} {initial_value.dtype=}."
        )
    if not initial_key.is_contiguous() or not initial_value.is_contiguous():
        raise ValueError("NCCL4Py direct ring expects contiguous K/V tensors.")

    if comm_stream is None:
        comm_stream = (
            torch.cuda.Stream(device=initial_key.device)
            if use_cuda_comm_stream and initial_key.is_cuda
            else None
        )

    direct = _get_or_create_direct(
        initial_key.shape,
        initial_key.dtype,
        initial_key.device,
        process_group,
        comm_stream,
    )

    with torch.autograd.profiler.record_function("ring_nccl4py_initial_publish"):
        direct.initial_publish(initial_key, initial_value)
    with torch.autograd.profiler.record_function("ring_nccl4py_prefetch/step=1"):
        direct.prepare_prefetch(1)
        direct.start_prefetch(1)

    step_plan = rank_plan.steps[0] if rank_plan is not None else None
    step_callback(0, initial_key, initial_value, step_plan)

    for step in range(1, world_size):
        with torch.autograd.profiler.record_function(
            f"ring_nccl4py_wait_prefetch/step={step}"
        ):
            direct.wait_prefetch()
        if step < world_size - 1:
            with torch.autograd.profiler.record_function(
                f"ring_nccl4py_prefetch/step={step + 1}"
            ):
                direct.prepare_prefetch(step + 1)
                direct.start_prefetch(step + 1)
        step_key, step_value = direct.staging(step)
        step_plan = rank_plan.steps[step] if rank_plan is not None else None
        step_callback(step, step_key, step_value, step_plan)

    with torch.autograd.profiler.record_function("ring_nccl4py_finalize"):
        direct.finalize()
