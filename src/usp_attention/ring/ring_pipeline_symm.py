# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Symmetric memory ring pipeline for KV exchange.

Drop-in replacement for :mod:`ring_pipeline` using
``torch.distributed._symmetric_memory`` for zero-copy peer KV reads
instead of NCCL P2P ops.  Each rank publishes its KV shard into a
shared buffer and reads remote shards directly via NVLink/NVSwitch,
synchronized with GPU-side semaphore signals.

Forward pipelines
    Three topologies are available, all drop-in replacements for
    :func:`~ring_pipeline.run_async_ring_kv_pipeline`:

    * :func:`run_async_ring_kv_pipeline_symm` -- every rank publishes its
      local KV once, then across W-1 loop steps each rank issues direct
      NVLink reads from each of the W-1 other peers.  Cross-call safety
      relies on a per-call ``hdl.barrier()`` plus a bounded multi-slot
      buffer cache to tolerate drift between stacked attention layers.

    * :func:`run_async_ring_kv_pipeline_symm_rotator` -- strict next/prev
      KV rotation with bidirectional ``READY`` / ``DONE`` signaling.  Each
      rank only ever talks to its direct predecessor and successor;
      channel count is constant regardless of W.  The ``DONE`` handshake
      closes the producer-overwrite-before-consumer-read hazard that a
      pure "ready" signal cannot, at the cost of one extra signal per
      step.  See :class:`SymmMemRingRotator` for the underlying object.

    * :func:`run_async_ring_kv_pipeline_symm_direct` -- direct peer reads
      with explicit per-peer ``READY`` / ``DONE`` handshakes instead of a
      collective ``hdl.barrier()``.  Each rank publishes its KV once per
      call, fires ``READY`` to every peer, and every consumer reads
      directly from the true owner of each step's shard.  Eliminates the
      per-step ``staging -> buf`` copy the rotator incurs, freeing
      comm-stream HBM bandwidth for the attention kernel.  See
      :class:`SymmMemRingDirect` for the underlying object.

Gradient rotation
    :class:`SymmMemGradRing` provides signal-based dk/dv accumulator
    rotation via symmetric memory with the same bidirectional READY/DONE
    handshake used by the rotator forward.  Each rank publishes its
    updated fp32 accumulator, fires ``READY`` to its successor, reads its
    predecessor's accumulator into local staging, and fires ``DONE`` back
    so the predecessor can safely overwrite its buffer on the next step.

    ``PendingGradExchange``, ``_launch_async_grad_rotation``, and
    ``_wait_pending_grad_exchange`` are re-exported from :mod:`ring_pipeline`
    for the NCCL fallback path.

Buffer allocation
    Shared buffers are cached by ``(shape, dtype, group, tag)`` so the
    expensive ``symm_mem.rendezvous`` cost is paid only once per cache
    entry. The all-to-all forward pipeline also supports a bounded
    multi-slot cache so callers can trade a small amount of extra VRAM
    for safer reuse when stacked attention blocks get ahead of slower
    peers; the rotator forward does not need slots because its per-call
    signal handshake is already sufficient for cross-call safety.

    A new buffer's signal pad is zeroed asynchronously on the allocating
    rank's GPU, behind whatever work is already queued there, while
    ``rendezvous`` lets a peer signal into it as soon as every rank has
    joined. :func:`_alloc_symm_kv_buf` therefore synchronizes the device
    between ``sym.empty`` and ``sym.rendezvous``; without that, a peer whose
    GPU is ahead can have its first signal erased by the late zeroing, and the
    matching wait never completes.

Debug logging
    Set the environment variable ``SYMM_RING_DEBUG=1`` to enable
    per-rank stage logging to stderr for deadlock diagnosis.  The
    setting also covers the rotator forward pipeline.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as sym

loguru = logging.getLogger(__name__)
from torch import Tensor
from torch.distributed._symmetric_memory import _SymmetricMemory

# Re-export types and grad rotation helpers so callers can import everything
# from this module.
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
from usp_attention.types import RingCombineMode as RingCombineMode

_SYMM_RING_DEBUG = os.environ.get("SYMM_RING_DEBUG", "0").lower() in {
    "1",
    "true",
    "yes",
}
_SYMM_RING_FWD_FINAL_BARRIER = os.environ.get(
    "SYMM_RING_FWD_FINAL_BARRIER",
    "0",
).lower() in {
    "1",
    "true",
    "yes",
}
_SYMM_RING_FWD_CACHE_SLOTS = max(
    2,
    int(os.environ.get("SYMM_RING_FWD_CACHE_SLOTS", "2")),
)


def _log(rank: int, msg: str) -> None:
    if _SYMM_RING_DEBUG:
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
        loguru.info(f"[symm_ring {rank_str} t={t:.6f}] {msg}")


def _peer_for_step(rank: int, step: int, world_size: int) -> int:
    """Return the rank whose KV is processed at the given ring step.

    Step 0 is the local rank; step *s* reads KV from rank
    ``(rank - s + world_size) % world_size``, matching the same KV-owner
    ordering as the NCCL rotation in :mod:`ring_pipeline`.
    """
    return (rank - step + world_size) % world_size


_CacheKey = tuple[torch.Size, torch.dtype, str, str]
_symm_cache: dict[_CacheKey, tuple[Tensor, _SymmetricMemory]] = {}
_FwdSlotKey = tuple[torch.Size, torch.dtype, str]
_symm_fwd_slot_cursor: dict[_FwdSlotKey, int] = {}
_symm_fwd_slots_ready: set[_FwdSlotKey] = set()


def _alloc_symm_kv_buf(
    kv_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    rank: int,
    tag: str = "default",
) -> tuple[Tensor, _SymmetricMemory]:
    """Get or create a shared ``(2, *kv_shape)`` buffer with rendezvous.

    The buffer and handle are cached by ``(shape, dtype, group_name, tag)``.
    The first call does the expensive ``sym.empty`` + ``sym.rendezvous``;
    subsequent calls with the same key return the cached objects instantly.

    Use distinct *tag* values (e.g. ``"fwd"`` / ``"bwd"``) to get
    separate buffers for forward and backward.

    A cache miss synchronizes the device before the rendezvous: ``sym.empty``
    zeroes the buffer's signal pad on this rank's GPU behind already queued
    work, and a peer may signal into the pad as soon as the rendezvous
    completes. Without the sync, a late zeroing erases that signal (in the
    first ring backward, the peer's barrier signal), a later wait consumes the
    next signal on the channel in its place, and one wait spins forever.
    """
    key: _CacheKey = (kv_shape, dtype, process_group.group_name, tag)
    cached = _symm_cache.get(key)
    if cached is not None:
        _log(rank, f"alloc_symm_kv_buf: cache hit for {kv_shape} tag={tag}")
        return cached

    _log(
        rank,
        f"alloc_symm_kv_buf: cache miss, allocating {kv_shape} dtype={dtype} tag={tag}",
    )
    buf = sym.empty((2, *kv_shape), dtype=dtype, device=device)
    # Let the signal-pad zeroing run before any peer can reach the pad.
    torch.cuda.synchronize(device)
    hdl = sym.rendezvous(buf, group=process_group)
    _log(rank, "alloc_symm_kv_buf: rendezvous complete")
    entry = (buf, hdl)
    _symm_cache[key] = entry
    return entry


def _lookup_cached_symm_kv_buf(
    kv_shape: torch.Size,
    dtype: torch.dtype,
    process_group: dist.ProcessGroup,
    rank: int,
    *,
    tag: str,
) -> tuple[Tensor, _SymmetricMemory]:
    """Return a cached symmetric KV buffer and fail loudly if missing."""
    key: _CacheKey = (kv_shape, dtype, process_group.group_name, tag)
    cached = _symm_cache.get(key)
    if cached is None:
        raise RuntimeError(
            "Expected a prewarmed symmetric KV buffer but cache entry was missing "
            f"for {kv_shape=} {dtype=} {process_group.group_name=} {tag=}."
        )
    _log(rank, f"alloc_symm_kv_buf: cache hit for {kv_shape} tag={tag}")
    return cached


def _fwd_slot_tags() -> tuple[str, ...]:
    """Return the forward slot tags in the deterministic prewarm order."""
    if _SYMM_RING_FWD_CACHE_SLOTS == 1:
        return ("fwd",)
    return tuple(
        f"fwd_slot_{slot_idx}" for slot_idx in range(_SYMM_RING_FWD_CACHE_SLOTS)
    )


def _ensure_symm_fwd_kv_bufs(
    kv_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    rank: int,
) -> tuple[str, ...]:
    """Prewarm every configured forward slot in one synchronized order.

    Creating a new symmetric buffer requires ``symm_mem.rendezvous``, which is
    collective across the process group. If we lazily create slot 2 only after
    some ranks have already advanced to later blocks, the faster ranks can hang
    waiting for stragglers to join the new rendezvous. Prewarming all forward
    slots on the first call avoids that mid-forward collective.
    """
    slot_key: _FwdSlotKey = (kv_shape, dtype, process_group.group_name)
    slot_tags = _fwd_slot_tags()
    if slot_key in _symm_fwd_slots_ready:
        return slot_tags

    _log(
        rank,
        f"alloc_symm_fwd_kv_buf: prewarming {len(slot_tags)} forward slot(s)",
    )
    for slot_idx, slot_tag in enumerate(slot_tags, start=1):
        _log(
            rank,
            "alloc_symm_fwd_kv_buf: "
            f"prewarm slot {slot_idx}/{len(slot_tags)} tag={slot_tag}",
        )
        _alloc_symm_kv_buf(
            kv_shape,
            dtype,
            device,
            process_group,
            rank,
            tag=slot_tag,
        )
    _symm_fwd_slots_ready.add(slot_key)
    return slot_tags


def _alloc_symm_fwd_kv_buf(
    kv_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    rank: int,
) -> tuple[Tensor, _SymmetricMemory]:
    """Get a forward shared buffer, optionally using a bounded slot cache.

    ``SYMM_RING_FWD_CACHE_SLOTS`` controls how many rendezvoused forward
    buffers are kept per ``(shape, dtype, group)``. All configured slots are
    prewarmed on first use in a deterministic order, and later calls only
    select among cached slots.
    """
    slot_key: _FwdSlotKey = (kv_shape, dtype, process_group.group_name)
    slot_tags = _ensure_symm_fwd_kv_bufs(
        kv_shape,
        dtype,
        device,
        process_group,
        rank,
    )
    slot_idx = _symm_fwd_slot_cursor.get(slot_key, 0)
    _symm_fwd_slot_cursor[slot_key] = (slot_idx + 1) % len(slot_tags)
    slot_tag = slot_tags[slot_idx]
    _log(
        rank,
        "alloc_symm_fwd_kv_buf: "
        f"using slot {slot_idx + 1}/{len(slot_tags)} tag={slot_tag}",
    )
    return _lookup_cached_symm_kv_buf(
        kv_shape,
        dtype,
        process_group,
        rank,
        tag=slot_tag,
    )


# ---------------------------------------------------------------------------
# SymmMemGradRing — backward dk/dv accumulator rotation via symmetric memory
# ---------------------------------------------------------------------------


class SymmMemGradRing:
    """Symmetric memory dk/dv gradient rotation with pairwise signaling.

    Uses ``put_signal`` / ``wait_signal`` for pairwise producer-consumer
    synchronization instead of a full group barrier at each step.  This
    avoids the O(W) serialization cost of barrier-per-step while still
    guaranteeing that a rank never reads stale data from its predecessor.

    Protocol per step (bundled convenience API):

    1. First iteration: producer calls :meth:`initial_publish` which writes
       dk/dv into its shared buffer and fires ``READY(step=0)`` to the
       successor.
    2. Subsequent iterations: producer calls :meth:`republish` which first
       waits for ``DONE(step-1)`` from its successor (confirming the
       successor's previous read has landed), writes dk/dv into ``buf``,
       and fires ``READY(step)``.
    3. Consumer calls :meth:`prepare_read_prev` on the current stream to wait
       for ``READY(step)`` from the predecessor and record an event.
    4. Consumer calls :meth:`start_read_prev` to make the comm stream wait on
       that event and read the predecessor buf into local staging.
    5. :meth:`wait_read` syncs the compute stream against the comm stream,
       then :meth:`ack_read_prev` fires ``DONE(step)`` back to the predecessor.
    6. At end of call, :meth:`finalize` drains the final ``DONE`` so signal
       counters return to zero between calls.

    Why the DONE handshake matters
        ``put_signal`` is stream-ordered, so ``READY`` guarantees the
        producer's write has landed before the consumer's read is allowed
        to begin.  It does **not** tell the producer when the consumer's
        NVLink DMA has physically completed.  Without ``DONE``, the next
        ``publish`` on the producer can overwrite ``buf`` while a slow
        consumer's DMA is still draining peer memory, silently corrupting
        the consumer's staging tensor.  Waiting for ``DONE`` before every
        overwrite closes that hazard explicitly.

    The predecessor is ``(rank - 1 + W) % W`` and the successor is
    ``(rank + 1) % W``, matching the NCCL grad rotation direction.

    Channels:

    * Channels 0, 1 -- ``READY`` signals indexed by ``step % 2``.
    * Channels 2, 3 -- ``DONE`` signals indexed by ``step % 2``.

    The even/odd lanes are defensive: a strict in-order protocol cannot
    alias signals on a single channel, but the extra lane mirrors the
    forward ring rotator and makes stray-signal bugs easier to spot.

    Stream placement
        Signal ops and local publishes run on the caller's current stream
        before the backward kernel starts. The comm stream waits on a recorded
        event and only performs the peer-copy work, so semaphore waits are not
        queued in front of overlappable DMA.

    Shared buffers are cached by ``(shape, dtype, group, "grad")`` so the
    expensive ``symm_mem.rendezvous`` cost is paid only once.
    """

    CH_READY_EVEN = 0
    CH_READY_ODD = 1
    CH_DONE_EVEN = 2
    CH_DONE_ODD = 3

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

        _log(self.rank, "SymmMemGradRing.__init__: start")

        self._buf, self._hdl = _alloc_symm_kv_buf(
            grad_shape,
            dtype,
            device,
            process_group,
            self.rank,
            tag="grad",
        )

        self._recv_dks = [
            torch.empty(grad_shape, dtype=dtype, device=device),
            torch.empty(grad_shape, dtype=dtype, device=device),
        ]
        self._recv_dvs = [
            torch.empty(grad_shape, dtype=dtype, device=device),
            torch.empty(grad_shape, dtype=dtype, device=device),
        ]
        self._next_buf_idx = 0
        self._last_read_idx = 0

        self._comm_stream = (
            comm_stream if comm_stream is not None else torch.cuda.Stream(device=device)
        )
        self._last_read_step: int | None = None

        # Initial barrier so all ranks' shared buffers are visible before any
        # signaling begins.
        self._hdl.barrier()
        _log(self.rank, "SymmMemGradRing.__init__: complete")

    @staticmethod
    def _ready_channel(step: int) -> int:
        return (
            SymmMemGradRing.CH_READY_EVEN
            if step % 2 == 0
            else SymmMemGradRing.CH_READY_ODD
        )

    @staticmethod
    def _done_channel(step: int) -> int:
        return (
            SymmMemGradRing.CH_DONE_EVEN
            if step % 2 == 0
            else SymmMemGradRing.CH_DONE_ODD
        )

    # ------------------------------------------------------------------
    # Low-level primitives
    # ------------------------------------------------------------------

    def wait_prev_signal(self, step: int) -> None:
        """Wait for the predecessor's ``READY(step)`` signal.

        Args:
            step: Ring loop step index.  Must match the ``step`` the
                predecessor used when it fired ``READY``.
        """
        channel = self._ready_channel(step)
        _log(
            self.rank,
            f"SymmMemGradRing.wait_prev_signal: "
            f"src={self._prev_rank} channel={channel} step={step}",
        )
        self._hdl.wait_signal(self._prev_rank, channel=channel)

    def _record_ready_event(self) -> torch.cuda.Event:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self.device))
        return event

    def prepare_read_prev(self, step: int) -> torch.cuda.Event:
        """Wait for predecessor ``READY(step)`` and return a comm-stream gate."""
        self.wait_prev_signal(step)
        return self._record_ready_event()

    def start_read_prev(self, step: int, ready_event: torch.cuda.Event) -> None:
        """Start async copy of predecessor's dk/dv on the comm stream.

        The matching ``READY`` wait already completed in
        :meth:`prepare_read_prev`. The comm stream waits on ``ready_event`` and
        then enqueues only the NVLink DMA copy.

        Args:
            step: Ring loop step index of the data being read (i.e., the
                step for which the predecessor fired ``READY``).

        Must be paired with a later :meth:`wait_read` before the
        resulting staging tensors are consumed.
        """
        peer_buf = self._hdl.get_buffer(
            self._prev_rank, self._buf.shape, self._buf.dtype
        )
        buf_idx = self._next_buf_idx
        self._next_buf_idx = 1 - self._next_buf_idx
        self._last_read_idx = buf_idx
        self._last_read_step = step

        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(ready_event)
            self._recv_dks[buf_idx].copy_(peer_buf[0])
            self._recv_dvs[buf_idx].copy_(peer_buf[1])

        _log(
            self.rank,
            f"SymmMemGradRing.start_read_prev: "
            f"peer={self._prev_rank} buf_idx={buf_idx} step={step}",
        )

    def wait_read(self) -> tuple[Tensor, Tensor]:
        """Block the compute stream until the pending read completes."""
        _log(self.rank, "SymmMemGradRing.wait_read: waiting on comm_stream")
        torch.cuda.current_stream(device=self.device).wait_stream(self._comm_stream)
        return (
            self._recv_dks[self._last_read_idx],
            self._recv_dvs[self._last_read_idx],
        )

    def ack_read_prev(self, step: int | None = None) -> None:
        """Signal that the predecessor read has completed."""
        if step is None:
            if self._last_read_step is None:
                raise RuntimeError("ack_read_prev called before start_read_prev.")
            step = self._last_read_step
        done_ch = self._done_channel(step)
        _log(
            self.rank,
            f"SymmMemGradRing.ack_read_prev: "
            f"dst={self._prev_rank} channel={done_ch} step={step}",
        )
        self._hdl.put_signal(self._prev_rank, channel=done_ch)

    # ------------------------------------------------------------------
    # Bundled convenience API
    # ------------------------------------------------------------------

    def initial_publish(self, dk: Tensor, dv: Tensor) -> None:
        """First publish of the call: write ``buf`` and fire ``READY(0)``.

        Runs on the caller's current stream so the ``READY`` signal is not
        queued behind comm-stream DMA.
        """
        ready_ch = self._ready_channel(0)
        _log(
            self.rank,
            f"SymmMemGradRing.initial_publish: copy dk/dv + put READY(0) ch={ready_ch}",
        )
        self._buf[0].copy_(dk)
        self._buf[1].copy_(dv)
        self._hdl.put_signal(self._next_rank, channel=ready_ch)

    def republish(self, dk: Tensor, dv: Tensor, step: int) -> None:
        """Subsequent publish: wait ``DONE(step-1)``, write ``buf``, fire ``READY(step)``.

        Runs on the caller's current stream: wait for the successor's prior
        read to complete, publish the updated accumulator, then fire READY.
        ``step`` must be >= 1.
        """
        if step < 1:
            raise ValueError(
                f"republish: step must be >= 1 (use initial_publish for step 0), got {step=}."
            )
        done_ch = self._done_channel(step - 1)
        ready_ch = self._ready_channel(step)
        _log(
            self.rank,
            f"SymmMemGradRing.republish(step={step}): "
            f"wait DONE({step - 1}) ch={done_ch}, copy dk/dv, "
            f"put READY({step}) ch={ready_ch}",
        )
        self._hdl.wait_signal(self._next_rank, channel=done_ch)
        self._buf[0].copy_(dk)
        self._buf[1].copy_(dv)
        self._hdl.put_signal(self._next_rank, channel=ready_ch)

    def finalize(self, step: int) -> None:
        """Drain the trailing ``DONE(step)`` signal at the end of the call.

        The final rotation's read fires a ``DONE`` that has no matching
        republish to wait on. Draining it on the current stream keeps signal
        counters balanced across calls (so no pre-seeding is required).
        """
        done_ch = self._done_channel(step)
        _log(
            self.rank,
            f"SymmMemGradRing.finalize(step={step}): wait DONE({step}) ch={done_ch}",
        )
        self._hdl.wait_signal(self._next_rank, channel=done_ch)


# ---------------------------------------------------------------------------
# Forward pipeline — signal-based double-buffered loop
# ---------------------------------------------------------------------------


def run_async_ring_kv_pipeline_symm(
    *,
    initial_key: Tensor,
    initial_value: Tensor,
    process_group: dist.ProcessGroup,
    step_callback: RingStepCallback,
    rank_plan: RankRingPlan | None = None,
    use_cuda_comm_stream: bool = True,
    comm_stream: torch.cuda.Stream | None = None,
) -> None:
    """Read KV from peers via symmetric memory and run per-step callback.

    Drop-in replacement for
    :func:`~ring_pipeline.run_async_ring_kv_pipeline`.

    Uses stream-wait synchronisation for comm/compute overlap: a
    dedicated comm stream pre-fetches the next step's KV while the
    compute stream runs the current step's callback.  This approach is
    stateless across calls (no GPU-side semaphores) so it is safe under
    ``do_bench`` or any repeated-invocation harness.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialised.
        ValueError: If the process group world size is invalid, or
            *rank_plan* step count does not match world size.
    """
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

    rank = dist.get_rank(process_group)
    device = initial_key.device

    _log(rank, f"pipeline_symm: start W={world_size} shape={initial_key.shape}")

    buf, hdl = _alloc_symm_fwd_kv_buf(
        initial_key.shape,
        initial_key.dtype,
        device,
        process_group,
        rank,
    )
    if comm_stream is None:
        comm_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    # Double-buffered read targets (regular GPU memory).
    read_ks = [torch.empty_like(initial_key), torch.empty_like(initial_key)]
    read_vs = [torch.empty_like(initial_value), torch.empty_like(initial_value)]

    _log(rank, "pipeline_symm: copy KV into shared buf")
    buf[0].copy_(initial_key)
    buf[1].copy_(initial_value)

    _log(rank, "pipeline_symm: barrier start")
    with torch.autograd.profiler.record_function("ring_symm_barrier"):
        hdl.barrier()
    _log(rank, "pipeline_symm: barrier done")

    # ---- step 0: local KV, overlap with pre-fetch for step 1 ----------
    if world_size > 1:
        peer = _peer_for_step(rank, 1, world_size)
        peer_buf = hdl.get_buffer(peer, buf.shape, buf.dtype)
        _log(rank, f"step 0: pre-fetch from peer={peer}")
        with torch.autograd.profiler.record_function(
            f"ring_symm_prefetch/step=1/peer={peer}"
        ):
            comm_stream.wait_stream(compute_stream)
            with torch.cuda.stream(comm_stream):
                read_ks[0].copy_(peer_buf[0])
                read_vs[0].copy_(peer_buf[1])

    step_plan = rank_plan.steps[0] if rank_plan is not None else None
    _log(rank, "step 0: callback (local KV)")
    step_callback(0, initial_key, initial_value, step_plan)

    # ---- steps 1 .. W-1 -----------------------------------------------
    for step in range(1, world_size):
        _log(rank, f"step {step}: wait for read")
        with torch.autograd.profiler.record_function(
            f"ring_symm_wait_prefetch/step={step}"
        ):
            compute_stream.wait_stream(comm_stream)

        read_idx = (step - 1) % 2
        k, v = read_ks[read_idx], read_vs[read_idx]

        # Pre-fetch next step's KV while current callback runs.
        if step < world_size - 1:
            next_read_idx = step % 2
            next_peer = _peer_for_step(rank, step + 1, world_size)
            next_peer_buf = hdl.get_buffer(next_peer, buf.shape, buf.dtype)
            _log(rank, f"step {step}: pre-fetch from peer={next_peer}")
            with torch.autograd.profiler.record_function(
                f"ring_symm_prefetch/step={step + 1}/peer={next_peer}"
            ):
                comm_stream.wait_stream(compute_stream)
                with torch.cuda.stream(comm_stream):
                    read_ks[next_read_idx].copy_(next_peer_buf[0])
                    read_vs[next_read_idx].copy_(next_peer_buf[1])

        step_plan = rank_plan.steps[step] if rank_plan is not None else None
        _log(rank, f"step {step}: callback")
        step_callback(step, k, v, step_plan)

    if _SYMM_RING_FWD_FINAL_BARRIER:
        _log(rank, "pipeline_symm: final barrier start")
        with torch.autograd.profiler.record_function("ring_symm_final_barrier"):
            hdl.barrier()
        _log(rank, "pipeline_symm: final barrier done")

    _log(rank, "pipeline_symm: complete")


# ---------------------------------------------------------------------------
# Rotator forward pipeline — strict next/prev KV rotation
# ---------------------------------------------------------------------------


class SymmMemRingRotator:
    """Strict next/prev KV rotation with bidirectional READY/DONE signals.

    Alternative forward transport that mirrors :class:`SymmMemGradRing`'s
    topology: every rank only ever talks to its direct predecessor and
    direct successor, and KV *contents* rotate around the ring so at step
    ``s`` each rank's shared buffer holds the KV that its successor needs
    to read at step ``s + 1``.

    One instance owns one shared symmetric-memory buffer of shape
    ``(2, *kv_shape)`` (slot 0 = K, slot 1 = V) plus a double-buffered
    local staging pair for incoming peer data.

    Per-call lifecycle (rank ``r``, ``W`` ranks total):

    1. :meth:`initial_publish` -- on the comm stream, write local KV
       into ``buf`` and fire ``READY(step=0)`` to ``r+1``.
    2. For ``step`` in ``[1, W-1]``:

        a. :meth:`start_prefetch` -- on the comm stream, wait for
           ``READY(step-1)`` from ``r-1``, copy ``r-1``'s buf into
           ``staging[(step-1) % 2]``, then fire ``DONE(step-1)`` back to
           ``r-1`` so it can safely overwrite its buf next step.
        b. :meth:`wait_prefetch` -- compute stream syncs with comm stream
           so the callback can read the just-fetched staging slot.
        c. :meth:`republish` -- on the comm stream, wait for
           ``DONE(step-1)`` from ``r+1`` (confirms its last read of our
           buf has completed), copy the freshly-received KV from staging
           into ``buf``, and fire ``READY(step)`` to ``r+1``.  Skipped at
           ``step == W-1`` since no successor step will read it.
    3. :meth:`finalize` -- remember that the final ``DONE(W-2)`` from
       ``r+1`` must be drained before the next call overwrites ``buf``.

    Stream placement
        Signal ops and local publishes run on the caller's current stream
        before each attention callback.  The comm stream waits on a recorded
        event and only performs the remote peer copy, so a semaphore wait is
        never queued ahead of an overlappable DMA.

    Signal accounting (per call, per direction):

    * Fires ``W-1`` ``READY`` to successor, waits ``W-1`` ``READY`` from
      predecessor.
    * Fires ``W-1`` ``DONE`` to predecessor, waits ``W-1`` ``DONE`` from
      successor.

    Counters therefore return to zero between calls, and no pre-seeding
    is required.

    Why the DONE handshake matters
        ``put_signal`` is stream-ordered, so a ``READY`` guarantees the
        producer's write has landed before the consumer's read is allowed
        to begin.  It does **not** tell the producer when the consumer's
        NVLink DMA has physically completed.  Without a matching
        ``DONE``, the next ``republish`` on the producer can overwrite
        ``buf`` while a slow consumer's copy is still draining peer
        memory, silently corrupting the consumer's staging tensor.
        Waiting for ``DONE`` before every overwrite makes this hazard
        structurally impossible.

    Channels:

    * Channels 0, 1 -- ``READY`` signals indexed by ``step % 2``.
    * Channels 2, 3 -- ``DONE`` signals indexed by ``step % 2``.
    """

    CH_READY_EVEN = 0
    CH_READY_ODD = 1
    CH_DONE_EVEN = 2
    CH_DONE_ODD = 3

    def __init__(
        self,
        kv_shape: torch.Size,
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

        _log(self.rank, "SymmMemRingRotator.__init__: start")

        self._buf, self._hdl = _alloc_symm_kv_buf(
            kv_shape,
            dtype,
            device,
            process_group,
            self.rank,
            tag="rotator",
        )

        self._staging_k = [
            torch.empty(kv_shape, dtype=dtype, device=device),
            torch.empty(kv_shape, dtype=dtype, device=device),
        ]
        self._staging_v = [
            torch.empty(kv_shape, dtype=dtype, device=device),
            torch.empty(kv_shape, dtype=dtype, device=device),
        ]

        self._comm_stream = (
            comm_stream if comm_stream is not None else torch.cuda.Stream(device=device)
        )
        self._pending_next_done_step: int | None = None

        # One-time rendezvous fence.  Not strictly required for correctness
        # (the signal handshake is the real cross-rank sync), but it's a
        # cheap "everyone is ready" checkpoint that matches how the grad
        # ring bootstraps itself.
        self._hdl.barrier()
        _log(self.rank, "SymmMemRingRotator.__init__: complete")

    @staticmethod
    def _ready_channel(step: int) -> int:
        return (
            SymmMemRingRotator.CH_READY_EVEN
            if step % 2 == 0
            else SymmMemRingRotator.CH_READY_ODD
        )

    @staticmethod
    def _done_channel(step: int) -> int:
        return (
            SymmMemRingRotator.CH_DONE_EVEN
            if step % 2 == 0
            else SymmMemRingRotator.CH_DONE_ODD
        )

    def staging(self, step: int) -> tuple[Tensor, Tensor]:
        """Return the staging ``(k, v)`` tensors holding step ``step``'s data.

        Step ``s`` data (for ``s >= 1``) is prefetched into staging slot
        ``(s - 1) % 2``.  Step 0 uses caller-owned local tensors directly.
        """
        if step <= 0 or step >= self.world_size:
            raise ValueError(
                f"staging: step must be in [1, {self.world_size}), got {step=}."
            )
        idx = (step - 1) % 2
        return self._staging_k[idx], self._staging_v[idx]

    def _record_ready_event(self) -> torch.cuda.Event:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self.device))
        return event

    def _drain_pending_next_done(self) -> None:
        if self._pending_next_done_step is None:
            return
        step = self._pending_next_done_step
        done_ch = self._done_channel(step)
        _log(
            self.rank,
            f"drain pending DONE(step={step}) from rank={self._next_rank} ch={done_ch}",
        )
        self._hdl.wait_signal(self._next_rank, channel=done_ch)
        self._pending_next_done_step = None

    def initial_publish(self, k: Tensor, v: Tensor) -> torch.cuda.Event:
        """Write local KV, exchange ``READY(0)``, and return a copy gate event.

        Signal work intentionally runs on the caller's current stream before
        attention starts.  The comm stream waits on the returned event and then
        performs only the peer DMA copy, avoiding a signal wait queued in front
        of overlappable communication.
        """
        ready_ch = self._ready_channel(0)
        _log(
            self.rank,
            f"initial_publish: drain prior DONE, copy local KV, "
            f"put READY(step=0) to rank={self._next_rank} ch={ready_ch}, "
            f"wait READY(step=0) from rank={self._prev_rank}",
        )
        self._drain_pending_next_done()
        self._buf[0].copy_(k)
        self._buf[1].copy_(v)
        self._hdl.put_signal(self._next_rank, channel=ready_ch)
        self._hdl.wait_signal(self._prev_rank, channel=ready_ch)
        return self._record_ready_event()

    def republish(self, step: int) -> torch.cuda.Event:
        """Overwrite ``buf``, exchange ``READY(step)``, and return a copy gate.

        The blocking signal ops run before the step's attention kernel.  The
        comm stream later waits on the returned event before copying the next
        predecessor buffer into staging.
        """
        if step <= 0 or step >= self.world_size:
            raise ValueError(
                f"republish: step must be in [1, {self.world_size}), got {step=}."
            )
        staging_k, staging_v = self.staging(step)
        done_ch = self._done_channel(step - 1)
        ready_ch = self._ready_channel(step)
        _log(
            self.rank,
            f"republish(step={step}): "
            f"wait DONE(step={step - 1}) ch={done_ch}, "
            f"copy staging->buf, put READY(step={step}) ch={ready_ch}, "
            f"wait READY(step={step}) from rank={self._prev_rank}",
        )
        self._hdl.wait_signal(self._next_rank, channel=done_ch)
        self._buf[0].copy_(staging_k)
        self._buf[1].copy_(staging_v)
        self._hdl.put_signal(self._next_rank, channel=ready_ch)
        self._hdl.wait_signal(self._prev_rank, channel=ready_ch)
        return self._record_ready_event()

    def start_prefetch(self, step: int, ready_event: torch.cuda.Event) -> None:
        """Start async read of predecessor's ``buf`` into staging for ``step``.

        The matching ``READY`` wait already completed on the caller's current
        stream.  The comm stream waits on ``ready_event`` and then enqueues only
        the peer-copy work, so signal waits cannot block DMA launch.

        Must be paired with a later :meth:`wait_prefetch` before the
        callback accesses the staging slot, followed by :meth:`ack_prefetch`.
        """
        if step <= 0 or step >= self.world_size:
            raise ValueError(
                f"start_prefetch: step must be in [1, {self.world_size}), got {step=}."
            )
        idx = (step - 1) % 2
        _log(
            self.rank,
            f"start_prefetch(step={step}): enqueue on comm_stream, "
            f"copy from rank={self._prev_rank} into staging[{idx}]",
        )

        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(ready_event)
            peer_buf = self._hdl.get_buffer(
                self._prev_rank, self._buf.shape, self._buf.dtype
            )
            self._staging_k[idx].copy_(peer_buf[0])
            self._staging_v[idx].copy_(peer_buf[1])
        _log(
            self.rank,
            f"start_prefetch(step={step}): DMA enqueued into staging[{idx}]",
        )

    def wait_prefetch(self) -> None:
        """Block the compute stream until the latest prefetch has landed."""
        _log(self.rank, "wait_prefetch: compute_stream.wait_stream(comm_stream)")
        torch.cuda.current_stream(device=self.device).wait_stream(self._comm_stream)

    def ack_prefetch(self, step: int) -> None:
        """Signal that this rank's predecessor read for ``step`` has completed."""
        if step <= 0 or step >= self.world_size:
            raise ValueError(
                f"ack_prefetch: step must be in [1, {self.world_size}), got {step=}."
            )
        done_ch = self._done_channel(step - 1)
        _log(
            self.rank,
            f"ack_prefetch(step={step}): put DONE(step={step - 1}) "
            f"to rank={self._prev_rank} ch={done_ch}",
        )
        self._hdl.put_signal(self._prev_rank, channel=done_ch)

    def finalize(self, last_step: int) -> None:
        """Defer draining the final ``DONE(last_step - 1)`` until next publish.

        The last iteration does not republish, but the successor still
        fires ``DONE(last_step - 1)`` after its final prefetch.  The next
        ``initial_publish`` drains that signal before overwriting ``buf``.
        """
        if last_step <= 0 or last_step >= self.world_size:
            raise ValueError(
                f"finalize: last_step must be in [1, {self.world_size}), "
                f"got {last_step=}."
            )
        _log(
            self.rank,
            f"finalize(last_step={last_step}): defer DONE(step={last_step - 1}) "
            f"from rank={self._next_rank}",
        )
        self._pending_next_done_step = last_step - 1


_RotatorKey = tuple[torch.Size, torch.dtype, str]
_rotator_cache: dict[_RotatorKey, SymmMemRingRotator] = {}


def _get_or_create_rotator(
    kv_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream | None,
) -> SymmMemRingRotator:
    """Return a cached rotator for the given geometry, constructing if needed.

    Caching avoids paying the rendezvous barrier on every pipeline call
    while still permitting multiple distinct (shape, dtype, group)
    combinations to coexist.
    """
    key: _RotatorKey = (kv_shape, dtype, process_group.group_name)
    cached = _rotator_cache.get(key)
    if cached is not None:
        return cached
    rotator = SymmMemRingRotator(
        kv_shape=kv_shape,
        dtype=dtype,
        device=device,
        process_group=process_group,
        comm_stream=comm_stream,
    )
    _rotator_cache[key] = rotator
    return rotator


def run_async_ring_kv_pipeline_symm_rotator(
    *,
    initial_key: Tensor,
    initial_value: Tensor,
    process_group: dist.ProcessGroup,
    step_callback: RingStepCallback,
    rank_plan: RankRingPlan | None = None,
    use_cuda_comm_stream: bool = True,
    comm_stream: torch.cuda.Stream | None = None,
) -> None:
    """Rotation-based forward ring pipeline using symmetric memory signals.

    Drop-in replacement (same signature) for
    :func:`run_async_ring_kv_pipeline_symm`.  The behavioural contract for
    ``step_callback`` is identical: it is invoked once per step with the KV
    that originated at rank ``(rank - step + W) % W``.  Only the
    underlying transport differs -- this variant rotates KV contents
    around the ring via strict next/prev signaling instead of issuing W-1
    direct reads from every peer.

    The ``use_cuda_comm_stream`` parameter is accepted for signature
    compatibility but unused: a dedicated comm stream is always required
    here to overlap the signal-gated peer read with the current step's
    callback.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialised.
        ValueError: If the process group world size is invalid or the
            ``rank_plan`` step count does not match world size.
    """
    del use_cuda_comm_stream

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

    rank = dist.get_rank(process_group)
    device = initial_key.device

    _log(rank, f"pipeline_rotator: start W={world_size} shape={initial_key.shape}")

    rotator = _get_or_create_rotator(
        initial_key.shape,
        initial_key.dtype,
        device,
        process_group,
        comm_stream,
    )

    # ---- step 0: initial publish on compute stream, launch prefetch for
    # step 1 concurrently, run callback on local KV. -------------------------
    with torch.autograd.profiler.record_function("ring_symm_rotator_initial_publish"):
        ready_event = rotator.initial_publish(initial_key, initial_value)
    with torch.autograd.profiler.record_function("ring_symm_rotator_prefetch/step=1"):
        rotator.start_prefetch(1, ready_event)

    step_plan = rank_plan.steps[0] if rank_plan is not None else None
    _log(rank, "step 0: callback (local KV)")
    step_callback(0, initial_key, initial_value, step_plan)

    # ---- steps 1 .. W-2: republish, fire READY, prefetch next, compute. ----
    # Rank r's buf at the end of step s holds the KV that r+1 will need to
    # read at step s+1.  We must wait for DONE(step-1) from r+1 before we
    # overwrite buf, then fire READY(step) afterwards so r+1's prefetch for
    # step+1 is unblocked.
    for step in range(1, world_size - 1):
        _log(rank, f"step {step}: wait prefetch for staging[(step-1)%2]")
        with torch.autograd.profiler.record_function(
            f"ring_symm_rotator_wait_prefetch/step={step}"
        ):
            rotator.wait_prefetch()
            rotator.ack_prefetch(step)

        _log(rank, f"step {step}: republish + READY(step={step})")
        with torch.autograd.profiler.record_function(
            f"ring_symm_rotator_republish/step={step}"
        ):
            ready_event = rotator.republish(step)

        _log(rank, f"step {step}: start prefetch for step {step + 1}")
        with torch.autograd.profiler.record_function(
            f"ring_symm_rotator_prefetch/step={step + 1}"
        ):
            rotator.start_prefetch(step + 1, ready_event)

        k_step, v_step = rotator.staging(step)
        step_plan = rank_plan.steps[step] if rank_plan is not None else None
        _log(rank, f"step {step}: callback")
        step_callback(step, k_step, v_step, step_plan)

    # ---- step W-1: no republish, no READY, no further prefetch.  We still
    # need to drain the final DONE(W-2) from r+1 so counters end balanced,
    # and we still need to wait for the last prefetch to land before calling
    # the callback. ---------------------------------------------------------
    last_step = world_size - 1
    _log(rank, f"step {last_step}: wait prefetch (final)")
    with torch.autograd.profiler.record_function(
        f"ring_symm_rotator_wait_prefetch/step={last_step}"
    ):
        rotator.wait_prefetch()
        rotator.ack_prefetch(last_step)

    k_step, v_step = rotator.staging(last_step)
    step_plan = rank_plan.steps[last_step] if rank_plan is not None else None
    _log(rank, f"step {last_step}: callback")
    step_callback(last_step, k_step, v_step, step_plan)

    _log(rank, f"step {last_step}: defer DONE(step={last_step - 1}) drain")
    with torch.autograd.profiler.record_function("ring_symm_rotator_finalize"):
        rotator.finalize(last_step)

    _log(rank, "pipeline_rotator: complete")


# ---------------------------------------------------------------------------
# Direct-read forward pipeline -- per-peer READY/DONE handshake
# ---------------------------------------------------------------------------


class SymmMemRingDirect:
    """Direct-read ring KV transport with per-peer READY/DONE signals.

    Third forward transport variant alongside
    :func:`run_async_ring_kv_pipeline_symm` (all-to-all with per-call
    ``hdl.barrier()``) and :class:`SymmMemRingRotator` (strict next/prev
    content rotation).

    Motivation
        The rotator performs two comm-stream HBM copies per ring step
        (``staging -> buf`` during ``republish`` and ``peer_buf ->
        staging`` during ``start_prefetch``).  Both copies compete with
        the attention kernel for HBM bandwidth, so the kernel never sees
        the full concurrency win of offloading signaling onto a dedicated
        comm stream.  This variant keeps the "publish once, read
        directly from the true owner" pattern of the all-to-all pipeline
        but replaces the ``hdl.barrier()`` with explicit per-peer
        ``READY`` / ``DONE`` handshakes, so cross-call buffer safety is
        structural rather than relying on a collective fence.

    Topology
        * Each rank publishes its local KV into its own shared buffer
          exactly once per call (no intermediate rotations).
        * At ring step ``s``, rank ``r`` reads directly from
          ``owner(s) = (r - s + W) mod W`` -- the rank that actually
          owns step ``s``'s KV shard.  NVLink traffic therefore spreads
          across all peer-to-peer links instead of concentrating on the
          ``r-1 <-> r`` link.
        * ``W-1`` peer reads per rank per call.

    Per-call lifecycle (rank ``r``, ``W`` ranks total):

    1. :meth:`initial_publish` -- on the caller's current stream, drain any
       prior peer ``DONE`` signals, write local KV into ``buf``, and fire
       ``READY(parity)`` to every peer.
    2. For ``step`` in ``[1, W-1]``:

        a. :meth:`prepare_prefetch` -- on the caller's current stream, wait
           for ``READY(parity)`` from ``owner(step)`` and record an event.
        b. :meth:`start_prefetch` -- on the comm stream, wait on that event
           and copy the owner's ``buf`` into ``staging[(step-1) % 2]``.
        c. :meth:`wait_prefetch` -- compute stream syncs with comm
           stream so the callback can read the just-fetched staging
           slot, then :meth:`ack_prefetch` fires ``DONE(parity)`` back.
    3. :meth:`finalize` -- defer the ``W-1`` peer-consumer
       ``DONE(parity)`` waits to the next call's :meth:`initial_publish`,
       and flip ``parity`` for the next call.

    Signal accounting (per call):

    * Fires ``W-1`` ``READY`` to distinct peers, waits ``W-1`` ``READY``
      from distinct owners.
    * Fires ``W-1`` ``DONE`` to distinct owners, waits ``W-1`` ``DONE``
      from distinct peer consumers.

    Counters therefore return to zero between calls, and no pre-seeding
    is required.

    Stream placement
        Signal ops and local publishes run on the caller's current stream
        before each attention callback.  The comm stream waits on the recorded
        event and only performs the peer copy, so semaphore waits cannot block
        overlappable DMA launch.

    Cross-call safety
        :meth:`finalize` records the parity whose peer-consumer ``DONE``
        signals are still outstanding.  The next :meth:`initial_publish`
        drains them before writing to ``buf``, so the publish cannot overwrite
        data while any peer consumer's prior DMA is still draining.

    Channels:

    * Channels 0, 1 -- ``READY`` signals indexed by call parity.
    * Channels 2, 3 -- ``DONE`` signals indexed by call parity.
    """

    CH_READY_EVEN = 0
    CH_READY_ODD = 1
    CH_DONE_EVEN = 2
    CH_DONE_ODD = 3

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
        self._peer_ranks = tuple(p for p in range(self.world_size) if p != self.rank)

        _log(self.rank, "SymmMemRingDirect.__init__: start")

        self._buf, self._hdl = _alloc_symm_kv_buf(
            kv_shape,
            dtype,
            device,
            process_group,
            self.rank,
            tag=tag,
        )

        self._staging_k = [
            torch.empty(kv_shape, dtype=dtype, device=device),
            torch.empty(kv_shape, dtype=dtype, device=device),
        ]
        self._staging_v = [
            torch.empty(kv_shape, dtype=dtype, device=device),
            torch.empty(kv_shape, dtype=dtype, device=device),
        ]

        self._comm_stream = (
            comm_stream if comm_stream is not None else torch.cuda.Stream(device=device)
        )

        # Per-call parity alternates 0/1; flipped in finalize.
        self._call_parity = 0
        self._pending_done_drain_parity: int | None = None

        # One-time rendezvous fence.  The real cross-rank sync is the
        # per-call signal handshake, but this establishes initial buffer
        # visibility and matches the bootstrapping style of the rotator
        # and grad ring.
        self._hdl.barrier()
        _log(self.rank, "SymmMemRingDirect.__init__: complete")

    @staticmethod
    def _ready_channel(parity: int) -> int:
        return (
            SymmMemRingDirect.CH_READY_EVEN
            if parity % 2 == 0
            else SymmMemRingDirect.CH_READY_ODD
        )

    @staticmethod
    def _done_channel(parity: int) -> int:
        return (
            SymmMemRingDirect.CH_DONE_EVEN
            if parity % 2 == 0
            else SymmMemRingDirect.CH_DONE_ODD
        )

    def staging(self, step: int) -> tuple[Tensor, Tensor]:
        """Return the staging ``(k, v)`` tensors holding step ``step``'s data.

        Step ``s`` data (for ``s >= 1``) is prefetched into staging slot
        ``(s - 1) % 2``.  Step 0 uses caller-owned local tensors
        directly.
        """
        if step <= 0 or step >= self.world_size:
            raise ValueError(
                f"staging: step must be in [1, {self.world_size}), got {step=}."
            )
        idx = (step - 1) % 2
        return self._staging_k[idx], self._staging_v[idx]

    def _record_ready_event(self) -> torch.cuda.Event:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self.device))
        return event

    def _drain_pending_peer_dones(self) -> None:
        if self._pending_done_drain_parity is None:
            return
        parity = self._pending_done_drain_parity
        done_ch = self._done_channel(parity)
        _log(
            self.rank,
            f"SymmMemRingDirect: drain {len(self._peer_ranks)} pending "
            f"DONE(parity={parity}) ch={done_ch}",
        )
        for peer in self._peer_ranks:
            self._hdl.wait_signal(peer, channel=done_ch)
        self._pending_done_drain_parity = None

    def initial_publish(self, k: Tensor, v: Tensor) -> None:
        """Write local KV into ``buf`` and fire ``READY`` to every peer.

        Runs on the caller's current stream.  Draining prior DONEs before the
        local copy makes buffer reuse safe without placing signal waits ahead
        of comm-stream peer DMA.
        """
        ready_ch = self._ready_channel(self._call_parity)
        _log(
            self.rank,
            f"SymmMemRingDirect.initial_publish: "
            f"copy local KV, put READY(parity={self._call_parity}) "
            f"ch={ready_ch} to {len(self._peer_ranks)} peers",
        )
        self._drain_pending_peer_dones()
        self._buf[0].copy_(k)
        self._buf[1].copy_(v)
        for peer in self._peer_ranks:
            self._hdl.put_signal(peer, channel=ready_ch)

    def prepare_prefetch(self, step: int) -> torch.cuda.Event:
        """Wait for ``owner(step)`` READY and return a comm-stream gate event."""
        if step <= 0 or step >= self.world_size:
            raise ValueError(
                f"prepare_prefetch: step must be in [1, {self.world_size}), got {step=}."
            )
        owner = _peer_for_step(self.rank, step, self.world_size)
        ready_ch = self._ready_channel(self._call_parity)
        _log(
            self.rank,
            f"SymmMemRingDirect.prepare_prefetch(step={step}): "
            f"owner={owner} wait READY(parity={self._call_parity}) ch={ready_ch}",
        )
        self._hdl.wait_signal(owner, channel=ready_ch)
        return self._record_ready_event()

    def start_prefetch(self, step: int, ready_event: torch.cuda.Event) -> None:
        """Start async read of ``owner(step)``'s ``buf`` into staging.

        The READY wait runs in :meth:`prepare_prefetch`.  This method only
        gates on the resulting event and enqueues the peer DMA copy on the comm
        stream.

        Must be paired with a later :meth:`wait_prefetch` before the
        callback accesses the staging slot, followed by :meth:`ack_prefetch`.
        """
        if step <= 0 or step >= self.world_size:
            raise ValueError(
                f"start_prefetch: step must be in [1, {self.world_size}), got {step=}."
            )
        idx = (step - 1) % 2
        owner = _peer_for_step(self.rank, step, self.world_size)
        _log(
            self.rank,
            f"SymmMemRingDirect.start_prefetch(step={step}): "
            f"owner={owner} enqueue copy on comm_stream",
        )

        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(ready_event)
            peer_buf = self._hdl.get_buffer(owner, self._buf.shape, self._buf.dtype)
            self._staging_k[idx].copy_(peer_buf[0])
            self._staging_v[idx].copy_(peer_buf[1])
        _log(
            self.rank,
            f"SymmMemRingDirect.start_prefetch(step={step}): "
            f"DMA enqueued into staging[{idx}]",
        )

    def wait_prefetch(self) -> None:
        """Block the compute stream until the latest prefetch has landed."""
        _log(
            self.rank,
            "SymmMemRingDirect.wait_prefetch: compute_stream.wait_stream(comm_stream)",
        )
        torch.cuda.current_stream(device=self.device).wait_stream(self._comm_stream)

    def ack_prefetch(self, step: int) -> None:
        """Signal that this rank's direct read for ``step`` has completed."""
        if step <= 0 or step >= self.world_size:
            raise ValueError(
                f"ack_prefetch: step must be in [1, {self.world_size}), got {step=}."
            )
        owner = _peer_for_step(self.rank, step, self.world_size)
        done_ch = self._done_channel(self._call_parity)
        _log(
            self.rank,
            f"SymmMemRingDirect.ack_prefetch(step={step}): "
            f"put DONE(parity={self._call_parity}) to rank={owner} ch={done_ch}",
        )
        self._hdl.put_signal(owner, channel=done_ch)

    def finalize(self) -> None:
        """Defer draining peer ``DONE(parity)`` signals and flip parity.

        Each of the ``W-1`` peer consumers fires exactly one ``DONE`` at
        the end of the step that reads from this rank.  The next call's
        :meth:`initial_publish` drains them before overwriting ``buf``.
        """
        _log(
            self.rank,
            f"SymmMemRingDirect.finalize: defer {len(self._peer_ranks)} "
            f"DONE(parity={self._call_parity}) waits",
        )
        self._pending_done_drain_parity = self._call_parity
        self._call_parity = 1 - self._call_parity


_DirectKey = tuple[torch.Size, torch.dtype, str, str]
_direct_cache: dict[_DirectKey, SymmMemRingDirect] = {}


def _get_or_create_direct(
    kv_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    process_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream | None,
    *,
    tag: str = "direct",
) -> SymmMemRingDirect:
    """Return a cached direct ring for the given geometry, constructing if needed.

    Caching avoids paying the rendezvous barrier on every pipeline call
    while still permitting multiple distinct ``(shape, dtype, group)``
    combinations to coexist.
    """
    key: _DirectKey = (kv_shape, dtype, process_group.group_name, tag)
    cached = _direct_cache.get(key)
    if cached is not None:
        return cached
    direct = SymmMemRingDirect(
        kv_shape=kv_shape,
        dtype=dtype,
        device=device,
        process_group=process_group,
        comm_stream=comm_stream,
        tag=tag,
    )
    _direct_cache[key] = direct
    return direct


def run_async_ring_kv_pipeline_symm_direct(
    *,
    initial_key: Tensor,
    initial_value: Tensor,
    process_group: dist.ProcessGroup,
    step_callback: RingStepCallback,
    rank_plan: RankRingPlan | None = None,
    use_cuda_comm_stream: bool = True,
    comm_stream: torch.cuda.Stream | None = None,
) -> None:
    """Direct-read forward ring pipeline using symmetric-memory signals.

    Drop-in replacement (same signature) for
    :func:`run_async_ring_kv_pipeline_symm` and
    :func:`run_async_ring_kv_pipeline_symm_rotator`.  The behavioural
    contract for ``step_callback`` is identical: it is invoked once per
    step with the KV that originated at rank
    ``(rank - step + W) mod W``.  Only the underlying transport differs
    -- this variant publishes local KV once per call then has every rank
    read directly from the owner of each step's shard, eliminating the
    per-step ``staging -> buf`` copy that the rotator pipeline incurs.

    The ``use_cuda_comm_stream`` parameter is accepted for signature
    compatibility but unused: a dedicated comm stream is always required
    here to overlap the signal-gated peer read with the current step's
    callback.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialised.
        ValueError: If the process group world size is invalid or the
            ``rank_plan`` step count does not match world size.
    """
    del use_cuda_comm_stream

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

    rank = dist.get_rank(process_group)
    device = initial_key.device

    _log(rank, f"pipeline_direct: start W={world_size} shape={initial_key.shape}")

    direct = _get_or_create_direct(
        initial_key.shape,
        initial_key.dtype,
        device,
        process_group,
        comm_stream,
    )

    # ---- step 0: publish local KV to all peers, launch prefetch for
    # step 1 concurrently, run callback on local KV. -------------------------
    with torch.autograd.profiler.record_function("ring_symm_direct_initial_publish"):
        direct.initial_publish(initial_key, initial_value)
    with torch.autograd.profiler.record_function("ring_symm_direct_prefetch/step=1"):
        ready_event = direct.prepare_prefetch(1)
        direct.start_prefetch(1, ready_event)

    step_plan = rank_plan.steps[0] if rank_plan is not None else None
    _log(rank, "step 0: callback (local KV)")
    step_callback(0, initial_key, initial_value, step_plan)

    # ---- steps 1 .. W-1: wait prior prefetch, issue next prefetch, compute.
    # No republish: buf already contains our final KV for the call, and every
    # peer reads directly from its step's owner. ----------------------------
    for step in range(1, world_size):
        _log(rank, f"step {step}: wait prefetch for staging[(step-1)%2]")
        with torch.autograd.profiler.record_function(
            f"ring_symm_direct_wait_prefetch/step={step}"
        ):
            direct.wait_prefetch()
            direct.ack_prefetch(step)

        if step < world_size - 1:
            _log(rank, f"step {step}: start prefetch for step {step + 1}")
            with torch.autograd.profiler.record_function(
                f"ring_symm_direct_prefetch/step={step + 1}"
            ):
                ready_event = direct.prepare_prefetch(step + 1)
                direct.start_prefetch(step + 1, ready_event)

        k_step, v_step = direct.staging(step)
        step_plan = rank_plan.steps[step] if rank_plan is not None else None
        _log(rank, f"step {step}: callback")
        step_callback(step, k_step, v_step, step_plan)

    # ---- defer W-1 DONE drains so the next call's publish is safe. ---------
    _log(rank, "pipeline_direct: finalize (defer W-1 DONE drains)")
    with torch.autograd.profiler.record_function("ring_symm_direct_finalize"):
        direct.finalize()

    _log(rank, "pipeline_direct: complete")
