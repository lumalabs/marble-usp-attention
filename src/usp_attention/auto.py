# SPDX-FileCopyrightText: 2026 Luma AI, Inc.
# SPDX-License-Identifier: MIT
"""Pick the sequence-parallel strategy per mask regime, off the training step.

The cost model (:mod:`usp_attention.strategy`) ranks every exact strategy for a
packed layout; its top three contain the measured fastest in every case we
measured, so :class:`AutoStrategy` does not trust its first pick blindly:

1. **Shortlist.** For each mask *regime* (a coarse signature of the layout:
   length, window, share of tokens per attention mode, document size) the top
   ``top_k`` strategies by predicted time become the candidates.
2. **Measure on real steps.** While a regime is unlocked, step ``s`` runs
   candidate ``s % top_k``. Every candidate is exact, so these are ordinary
   training steps; nothing is recomputed or thrown away.
3. **Lock.** Once each candidate has ``trial_steps`` timed steps (its first
   step compiles kernels and is not counted), the ranks
   agree on the slowest rank's median per candidate (one ``all_reduce``) and
   lock the fastest. The lock applies from ``lock_step + horizon`` on, so plans
   that data-loader workers built ahead of time stay valid; ``horizon`` must
   exceed the prefetch depth.

Planning stays off the critical path: :meth:`AutoStrategy.plan` runs the cost
model and builds the plan on CPU with coordinate-only groups, and
:class:`PlanPrefetcher` runs it for the next steps in data-loader worker
processes (or a thread) while the GPU runs the current one. The training step
only calls :meth:`AutoStrategy.attach`, which copies the pinned plan's tensors
to the GPU asynchronously and plugs in process groups created once, for every
split, by :meth:`AutoStrategy.create_groups`.
"""

from __future__ import annotations

import dataclasses
import importlib
import io
import math
import os
import pickle
import statistics
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import Tensor

from usp_attention.distributed import USPGroups
from usp_attention.fa4.flash4_attn import AttentionMode
from usp_attention.strategy import (
    Hardware,
    Strategy,
    candidates,
    estimate,
    features,
    mask_stats,
)
from usp_attention.usp import (
    USPConfig,
    USPPlan,
    build_sliding_window_plan,
    build_usp_plan,
    usp_attention,
)

Regime = tuple


@dataclass(kw_only=True, slots=True)
class PlannedBatch:
    """A batch's strategy and its CPU-built plans (picklable).

    Attributes:
        step: Training step this batch is planned for.
        regime: Mask regime signature of the layout.
        shortlist: The regime's candidates (identical on every rank).
        strategy: Strategy the plans implement.
        plan: Plan for global attention layers (CPU tensors, no process groups).
        sliding_plan: Plan for sliding-window layers sharded like ``plan``, or
            ``None`` without a window.
        layout: ``(document_lens, split_lens, attn_modes)`` for re-planning.
        plan_ms: CPU time :meth:`AutoStrategy.plan` took (selection + plans).
    """

    step: int
    regime: Regime
    shortlist: list[Strategy]
    strategy: Strategy
    plan: USPPlan
    sliding_plan: USPPlan | None
    layout: tuple[list[int], list[int], list[AttentionMode]]
    plan_ms: float = 0.0

    def __reduce__(self):
        # Plans reference FA4 mask predicates defined in closures; pickle those
        # by their module-level name (see _PlanPickler).
        buf = io.BytesIO()
        _PlanPickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(
            tuple(getattr(self, f.name) for f in dataclasses.fields(self))
        )
        return (_load_planned, (buf.getvalue(),))

    def pin_memory(self) -> PlannedBatch:
        """The batch with its plans in page-locked memory (``DataLoader`` hook)."""
        return dataclasses.replace(
            self, plan=pin_plan(self.plan), sliding_plan=pin_plan(self.sliding_plan)
        )


def _module_name(obj) -> str | None:
    """``"module:attr"`` of a module-level ``usp_attention`` object, if any."""
    for name, module in list(sys.modules.items()):
        if name.startswith("usp_attention") and module is not None:
            for attr, value in vars(module).items():
                if value is obj:
                    return f"{name}:{attr}"
    return None


def _resolve(name: str):
    module, attr = name.split(":")
    return getattr(importlib.import_module(module), attr)


class _PlanPickler(pickle.Pickler):
    """Pickles closure-defined kernel predicates (``mask_mod``) by reference.

    Plans hold module-level singletons such as
    ``usp_attention.ring.flash4_ring_attn.POSITION_RANGE_MASK_MOD``, created
    inside factory functions; plain pickle rejects them as local objects.
    """

    def reducer_override(self, obj):
        if callable(obj) and "<locals>" in getattr(obj, "__qualname__", ""):
            name = _module_name(obj)
            if name is None:
                raise pickle.PicklingError(f"plan holds a local callable {obj!r}")
            return (_resolve, (name,))
        return NotImplemented


def _load_planned(data: bytes) -> PlannedBatch:
    return PlannedBatch(
        **dict(
            zip(
                (f.name for f in dataclasses.fields(PlannedBatch)),
                pickle.loads(data),
                strict=True,
            )
        )
    )


def regime_signature(
    document_lens: list[int],
    split_lens: list[int],
    attn_modes: list[AttentionMode],
    *,
    sliding_window: int | None = None,
) -> Regime:
    """Coarse layout signature: layouts with the same one share a decision.

    Length and window exactly; the share of tokens in each attention mode to
    the nearest 10%; the token-weighted mean document length to the nearest
    power of two. O(number of spans), cheap enough to run every batch.
    """
    total = sum(document_lens)
    share = {m: 0 for m in ("causal", "full", "noise", "group")}
    for length, mode in zip(split_lens, attn_modes, strict=True):
        share[mode] += length
    mean_doc = sum(d * d for d in document_lens) / total
    return (
        total,
        sliding_window or 0,
        tuple(round(10 * share[m] / total) for m in share),
        round(math.log2(mean_doc)),
    )


def strategy_config(strategy: Strategy, base: USPConfig, *, hopper: bool) -> USPConfig:
    """The exact :class:`USPConfig` that runs ``strategy`` (base for the rest)."""
    if strategy.balance == "chunked":
        # The chunked ring with exact fp32 merging of its partials.
        return dataclasses.replace(
            base,
            kv_exchange="ring",
            balance="chunks",
            dense_ring=True,
            exact_dense_ring=True,
        )
    return dataclasses.replace(
        base,
        kv_exchange=strategy.kv_exchange,
        balance="chunks" if strategy.balance == "chunks" else "ptrr",
        dense_ring=False,
        halo_overlap=hopper and strategy.kv_exchange == "gather",
    )


def coordinates(strategy: Strategy, cp_rank: int) -> USPGroups:
    """This rank's coordinates under ``strategy``, without process groups.

    Ulysses ranks are consecutive (as in :meth:`USPGroups.create`).
    """
    u = strategy.ulysses_size
    return USPGroups(
        ring_size=strategy.ring_size,
        ulysses_size=u,
        ring_rank=cp_rank // u,
        ulysses_rank=cp_rank % u,
        cp_rank=cp_rank,
        ring_group=None,
        ulysses_group=None,
        cp_group=None,
    )


def plan_to(obj, device: torch.device):
    """Copy of a plan (nested dataclasses, lists, dicts) with tensors on ``device``.

    From pinned CPU memory (:func:`pin_plan`) the copies are asynchronous.
    """
    if isinstance(obj, Tensor):
        return obj.to(device, non_blocking=True)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: plan_to(getattr(obj, f.name), device)
                for f in dataclasses.fields(obj)
                if f.init
            },
        )
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(plan_to(x, device) for x in obj))  # NamedTuple
    if isinstance(obj, list | tuple):
        return type(obj)(plan_to(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: plan_to(v, device) for k, v in obj.items()}
    return obj


def pin_plan(obj):
    """The plan with CPU tensors in page-locked memory (async GPU copies)."""
    if not torch.cuda.is_available():
        return obj
    if isinstance(obj, Tensor):
        return obj.pin_memory() if obj.device.type == "cpu" else obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: pin_plan(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
                if f.init
            },
        )
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(pin_plan(x) for x in obj))
    if isinstance(obj, list | tuple):
        return type(obj)(pin_plan(x) for x in obj)
    if isinstance(obj, dict):
        return {k: pin_plan(v) for k, v in obj.items()}
    return obj


@dataclass(kw_only=True, slots=True)
class _Trial:
    shortlist: list[Strategy]
    times: dict[Strategy, list[float]] = field(default_factory=dict)


class AutoStrategy:
    """Chooses, plans and locks the fastest exact strategy per mask regime.

    Args:
        world: Context-parallel group size (ring x Ulysses GPUs).
        heads: Query heads.
        kv_heads: Key/value heads.
        head_dim: Head dimension.
        config: Execution options shared by all strategies (combine mode,
            transport, ...); the strategy fields are set per candidate.
        hardware: Cost-model constants.
        layers: ``(global_layers, sliding_layers)`` weights for ranking when
            the model mixes both (``sliding_window`` given to :meth:`plan`).
        top_k: Candidates measured per regime.
        trial_steps: Timed steps per candidate before locking (after one
            untimed step that compiles its kernels).
        horizon: Steps between a lock decision and its first use; must exceed
            how many batches the data loader plans ahead.
        shared: Lock table visible to data-loader workers (e.g. a
            ``multiprocessing.Manager().dict()``); a plain dict otherwise.
        hopper: Run gathers with halo overlap (SM90 only).
    """

    def __init__(
        self,
        *,
        world: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        config: USPConfig,
        hardware: Hardware | None = None,
        layers: tuple[int, int] = (1, 0),
        top_k: int = 3,
        trial_steps: int = 2,
        horizon: int = 64,
        shared: dict | None = None,
        hopper: bool = True,
    ) -> None:
        self.world, self.heads, self.kv_heads, self.head_dim = (
            world,
            heads,
            kv_heads,
            head_dim,
        )
        self.config = config
        self.hardware = hardware or Hardware()
        self.layers = layers
        self.top_k = top_k
        self.trial_steps = trial_steps
        self.horizon = horizon
        self.locks = shared if shared is not None else {}
        self.hopper = hopper
        self.groups: dict[tuple[int, int], USPGroups] = {}
        self._streams: dict[tuple[int, int], tuple] = {}
        self._shortlists: dict[Regime, list[Strategy]] = {}
        self._trials: dict[Regime, _Trial] = {}

    # -- process groups (main process, collective, once) ------------------
    def create_groups(self) -> None:
        """Create process groups for every (ring, Ulysses) split, collectively."""
        for s in candidates(self.world, self.heads, self.kv_heads):
            key = (s.ring_size, s.ulysses_size)
            if key not in self.groups:
                self.groups[key] = USPGroups.create(
                    ring_size=s.ring_size, ulysses_size=s.ulysses_size
                )

    # -- selection (CPU, any process) --------------------------------------
    def shortlist(
        self,
        document_lens: list[int],
        split_lens: list[int],
        attn_modes: list[AttentionMode],
        *,
        sliding_window: int | None = None,
    ) -> list[Strategy]:
        """The ``top_k`` strategies by predicted time (cached per regime)."""
        regime = regime_signature(
            document_lens, split_lens, attn_modes, sliding_window=sliding_window
        )
        if regime in self._shortlists:
            return self._shortlists[regime]
        from usp_attention.ring.dense_gather import dense_gather_supported

        n_global, n_sliding = self.layers if sliding_window else (1, 0)
        stats = mask_stats(document_lens, split_lens, attn_modes)
        windowed = (
            mask_stats(
                document_lens, split_lens, attn_modes, sliding_window=sliding_window
            )
            if n_sliding
            else None
        )
        chunked = sliding_window is None and dense_gather_supported(
            document_lens, split_lens, attn_modes
        )
        scored = []
        for s in candidates(
            self.world, self.heads, self.kv_heads, chunked_ring=chunked
        ):
            kernel = "dense" if s.balance == "chunked" else "sparse"
            cost = 0.0
            fits = True
            for weight, st in ((n_global, stats), (n_sliding, windowed)):
                if not weight:
                    continue
                e = estimate(
                    s,
                    features(
                        s,
                        st,
                        heads=self.heads,
                        kv_heads=self.kv_heads,
                        head_dim=self.head_dim,
                        gpus_per_node=self.hardware.gpus_per_node,
                    ),
                    self.hardware,
                    kernel=kernel,
                )
                cost += weight * e.total_ms
                fits &= e.fits
            scored.append((not fits, cost, s))
        scored.sort(key=lambda t: (t[0], t[1]))
        out = [s for _, _, s in scored[: self.top_k]]
        self._shortlists[regime] = out
        return out

    def decide(self, step: int, regime: Regime, shortlist: list[Strategy]) -> Strategy:
        """Strategy for ``step``: the lock if it applies, else the trial schedule."""
        lock = self.locks.get(regime)
        if lock is not None and step >= lock[1]:
            return lock[0]
        return shortlist[step % len(shortlist)]

    def plan(
        self,
        step: int,
        document_lens: list[int],
        split_lens: list[int],
        attn_modes: list[AttentionMode],
        *,
        cp_rank: int,
        sliding_window: int | None = None,
    ) -> PlannedBatch:
        """Choose this step's strategy and build its plans on CPU.

        Safe to call in a data-loader worker: no process groups, no GPU.
        """
        start = time.perf_counter()
        regime = regime_signature(
            document_lens, split_lens, attn_modes, sliding_window=sliding_window
        )
        shortlist = self.shortlist(
            document_lens, split_lens, attn_modes, sliding_window=sliding_window
        )
        strategy = self.decide(step, regime, shortlist)
        plan, sliding = self._build(
            strategy,
            (document_lens, split_lens, attn_modes),
            cp_rank,
            sliding_window,
            torch.device("cpu"),
        )
        return PlannedBatch(
            step=step,
            regime=regime,
            shortlist=shortlist,
            strategy=strategy,
            plan=plan,
            sliding_plan=sliding,
            layout=(list(document_lens), list(split_lens), list(attn_modes)),
            plan_ms=(time.perf_counter() - start) * 1e3,
        )

    def _build(self, strategy, layout, cp_rank, sliding_window, device):
        docs, splits, modes = layout
        config = strategy_config(strategy, self.config, hopper=self.hopper)
        groups = coordinates(strategy, cp_rank)
        plan = build_usp_plan(
            document_lens=docs,
            split_lens=splits,
            attn_modes=modes,
            groups=groups,
            config=config,
            device=device,
        )
        sliding = None
        if sliding_window:
            sliding = build_sliding_window_plan(
                plan,
                sliding_window=sliding_window,
                document_lens=docs,
                split_lens=splits,
                attn_modes=modes,
                device=device,
            )
        return plan, sliding

    # -- training step (main process) --------------------------------------
    def attach(
        self, batch: PlannedBatch, device: torch.device
    ) -> tuple[USPPlan, USPPlan | None]:
        """Plans on ``device`` with real process groups.

        Re-plans synchronously in the rare case the batch was planned before a
        lock this step must follow (data loader ahead of ``horizon``), so every
        rank always runs the same strategy.
        """
        self._shortlists.setdefault(batch.regime, batch.shortlist)
        expected = self.decide(batch.step, batch.regime, batch.shortlist)
        if expected != batch.strategy:
            plan, sliding = self._build(
                expected,
                batch.layout,
                dist.get_rank() % self.world,
                batch.regime[1] or None,
                torch.device("cpu"),
            )
            batch.strategy, batch.plan, batch.sliding_plan = expected, plan, sliding
        split = (batch.strategy.ring_size, batch.strategy.ulysses_size)
        if split not in self._streams:
            self._streams[split] = (torch.cuda.Stream(), torch.cuda.Stream())
        kv_stream, grad_stream = self._streams[split]

        def place(p: USPPlan) -> USPPlan:
            # Real groups, and one pair of comm streams per split for all steps.
            return dataclasses.replace(
                plan_to(p, device),
                groups=self.groups[split],
                _kv_comm_stream=kv_stream,
                _grad_comm_stream=grad_stream,
            )

        plan = place(batch.plan)
        sliding = place(batch.sliding_plan) if batch.sliding_plan is not None else None
        return plan, sliding

    def record(self, batch: PlannedBatch, step_ms: float) -> None:
        """Feed a step's time back; locks a regime once its trials are done.

        Must be called on every rank for every step (the lock decision is one
        ``all_reduce`` that all ranks reach at the same step).
        """
        if batch.regime in self.locks:
            return
        trial = self._trials.setdefault(batch.regime, _Trial(shortlist=batch.shortlist))
        trial.times.setdefault(batch.strategy, []).append(step_ms)
        # A candidate's first step compiles its kernels: it is not timed.
        needed = self.trial_steps + 1
        if any(len(trial.times.get(s, ())) < needed for s in trial.shortlist):
            return
        medians = torch.tensor(
            [statistics.median(trial.times[s][1:]) for s in trial.shortlist],
            dtype=torch.float64,
            device=self._reduce_device(),
        )
        if dist.is_initialized():
            dist.all_reduce(medians, op=dist.ReduceOp.MAX)
        best = trial.shortlist[int(torch.argmin(medians))]
        self.locks[batch.regime] = (best, batch.step + self.horizon)

    @staticmethod
    def _reduce_device() -> torch.device:
        if dist.is_initialized() and dist.get_backend() == "nccl":
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")


class _Steps(torch.utils.data.Dataset):
    """Step ``i`` of ``layouts`` planned by :meth:`AutoStrategy.plan` (in a worker)."""

    def __init__(self, auto, layouts, cp_rank, sliding_window):
        self.auto, self.layouts = auto, layouts
        self.cp_rank, self.sliding_window = cp_rank, sliding_window

    def __len__(self) -> int:
        return len(self.layouts)

    def __getitem__(self, i: int) -> PlannedBatch:
        step, docs, splits, modes = self.layouts[i]
        return self.auto.plan(
            step,
            docs,
            splits,
            modes,
            cp_rank=self.cp_rank,
            sliding_window=self.sliding_window,
        )


def _share_cores(workers: int) -> None:
    """Planner workers split the node's cores with the other ranks' planners."""
    local = int(os.environ.get("LOCAL_WORLD_SIZE", "1")) * max(1, workers)
    torch.set_num_threads(max(1, (os.cpu_count() or local) // local))


class PlanPrefetcher:
    """Plans the next steps in data-loader worker processes (or a thread).

    ``layouts`` is a sequence of ``(step, document_lens, split_lens,
    attn_modes)``; iterating yields :class:`PlannedBatch` es, pinned, built by
    :meth:`AutoStrategy.plan` up to ``depth`` steps ahead (``depth`` must be
    below the strategy's ``horizon``). Worker processes keep the planners'
    Python work off the training process (no GIL contention); they see lock
    decisions through a shared dict, set up here when ``auto.locks`` is a plain
    dict. ``processes=False`` plans in one background thread instead, which is
    enough when a plan is cheap next to a step.

    Args:
        auto: The strategy that plans every step.
        layouts: Upcoming steps' layouts.
        cp_rank: This rank's position in the CP group.
        sliding_window: Window of the model's sliding layers, if any.
        depth: Steps planned ahead.
        workers: Planner processes (or threads).
        processes: Plan in worker processes (forked) rather than threads.

    Raises:
        ValueError: If ``depth`` is not below ``auto.horizon`` (a lock could then
            apply to a step already planned with a trial strategy).
    """

    def __init__(
        self,
        auto: AutoStrategy,
        layouts,
        *,
        cp_rank: int,
        sliding_window: int | None = None,
        depth: int = 2,
        workers: int = 2,
        processes: bool = True,
    ) -> None:
        if depth >= auto.horizon:
            raise ValueError(
                f"depth {depth} must be below the lock horizon {auto.horizon}"
            )
        layouts = list(layouts)
        if processes:
            import multiprocessing

            if type(auto.locks) is dict:
                auto.locks = multiprocessing.Manager().dict(dict(auto.locks))
            loader = torch.utils.data.DataLoader(
                _Steps(auto, layouts, cp_rank, sliding_window),
                batch_size=None,
                num_workers=workers,
                prefetch_factor=max(1, depth // workers),
                pin_memory=torch.cuda.is_available(),
                multiprocessing_context="fork",
                worker_init_fn=lambda _: _share_cores(workers),
            )
            self._it = iter(loader)
        else:
            self._it = self._threaded(
                auto, layouts, cp_rank, sliding_window, depth, workers
            )

    @staticmethod
    def _threaded(auto, layouts, cp_rank, sliding_window, depth, workers):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="usp-plan"
        ) as pool:

            def submit(i):
                step, docs, splits, modes = layouts[i]
                return pool.submit(
                    lambda: auto.plan(
                        step,
                        docs,
                        splits,
                        modes,
                        cp_rank=cp_rank,
                        sliding_window=sliding_window,
                    ).pin_memory()
                )

            pending = [submit(i) for i in range(min(depth, len(layouts)))]
            for i in range(len(layouts)):
                batch = pending.pop(0).result()
                if i + depth < len(layouts):
                    pending.append(submit(i + depth))
                yield batch

    def __iter__(self):
        return self

    def __next__(self) -> PlannedBatch:
        return next(self._it)


def auto_attention(
    q: Tensor, k: Tensor, v: Tensor, plan: USPPlan, strategy: Strategy
) -> Tensor:
    """:func:`usp_attention` with the strategy's key/value head replication."""
    if strategy.replicate_kv > 1:
        k, v = (t.repeat_interleave(strategy.replicate_kv, dim=2) for t in (k, v))
    return usp_attention(q, k, v, plan)
