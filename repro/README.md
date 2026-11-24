# MARBLE reproduction guide

This guide covers rebuilding the paper's tables from the committed measurements,
planning and running fresh measurements, installing the baselines, and checking
numerical correctness.

Run commands from the repository root. GPU runs need Linux x86-64, Python 3.11, a
recent NVIDIA driver and the CUDA development tools. Use a fresh node-local
directory for caches and baseline environments.

## Which results back the paper

| Rows | Location |
|---|---|
| Main multi-node comparison (32–256 H100s, NCCL 2.29.7) | `benchmarks/results/rerun_oct28/mlsys` |
| Grid gap-fill (2026-10-30) | `benchmarks/results/rerun_oct30_gapfill` |
| MagiAttention reruns and strong-scaling fill-in | `benchmarks/results/rerun_oct28_scaling` |
| The same H100 measurements with NCCL 2.28.9 (superseded by the 2026-10-28 rerun) | `benchmarks/results/rerun_oct26` |

Each of these folders has a README with the measurement settings.
`benchmarks/MLSYS_RESULTS.md` is the full report generated from the first three
(60 cases). Older folders such as `benchmarks/results/mlsys` hold earlier
measurements with a different software stack. They are kept for reference, and
the paper's tables do not use them.

### Provenance of the committed rows

Rows from the reruns carry an environment stamp (`provenance`): torch, FA4, CuTe
DSL, NCCL, CUDA, driver, GPU models, Python and the command line. Earlier rows
were recorded before stamping existed and have no stamp. Their environment is
described in the README of their folder, and no stamp is reconstructed for
them. Rows from the development repository do not record a source revision. Rows
for FlashMask and MagiAttention do record the public baseline commit they ran
(`6a150fc` and `de13b2a`).

Earlier versions of `bench_attention.py` recorded the `exact_dense_ring`,
`gather_regather` and `gather_max_keys` arguments without passing them into
`USPConfig`, so those flags alone do not establish which execution path an old
row used. The driver now passes them through and records the full configuration
and the active generation-ring path.

### Layout packs

All 104 compressed layout packs and their generators are in
[`benchmarks/layouts`](../benchmarks/layouts/README.md), including exact public
shards and regeneration commands. A saved row naming `<name>.json#0` can use
`file:benchmarks/layouts/packs/<name>.json#0`; the loader resolves the `.gz`.

- The three small 32k BAGEL accuracy fixtures are committed as used and have no
  generation script.
- All seven long BAGEL packs were regenerated and match the committed JSON.
- The other generators need external public shards and tokenizers, and their
  Hugging Face references can change. For stable table inputs, use the committed
  packs.

The FlashMask launchers and the original environment pins are also committed; see
[their run guide](../benchmarks/baselines/flashmask_runs/README.md).

## Install MARBLE

```bash
uv sync --frozen
```

This installs the locked Linux stack: torch 2.11/cu129, FA4 4.0.0b34 and CuTe DSL
4.7. The earlier H100 measurements under `benchmarks/results/mlsys` used FA4 b21 and
CuTe DSL 4.6. Installing the current lockfile does not recreate that environment.

## Rebuild a table from the committed measurements

These commands use only Python's standard library and need no GPU. Every output
cell keeps a `file:line` reference to its source row. They rebuild the tables
from the saved measurements; they do not measure anything again. With no options
they read the rows listed above.

| Table | Command |
|---|---|
| Main multi-node comparison and headline | `python repro/tables.py scale` |
| Exchange × token-order ablation | `python repro/tables.py ablation` |
| BAGEL long video and dense ring | `python repro/tables.py long-video` |
| B300 real-pack comparison | `python repro/tables.py b300` |
| Full-model forward and backward | `python repro/tables.py end-to-end` |

For rows you collected yourself with the same schema, pass
`--results <directory>` (and `--extra <files or globs>` for more files). The
complete report is regenerated with:

```bash
python benchmarks/report_tables.py > benchmarks/MLSYS_RESULTS.md
```

## Plans for fresh runs of the saved configurations

Each command below prints a JSON plan with the following for every row of the table:

- the committed input pack or built-in synthetic layout;
- the driver command and its options;
- the GPU count;
- the source row.

Plans include every candidate configuration the table considered, including
tuning repeats. They describe new measurements with the current code and default
timing settings; they do not reconstruct an earlier environment. Options a row
did not record use the current driver defaults. A recorded dense-ring activation
or explicit exact-dense request turns on `--dense-ring on`. Built-in layout names
such as `mixed`, `vlm` and `video` go to the driver directly; file-backed layouts
need their committed pack.

| Table | Plan command |
|---|---|
| Main comparison | `python repro/replay.py scale` |
| Exchange × token-order ablation | `python repro/replay.py ablation` |
| BAGEL long video | `python repro/replay.py long-video` |
| B300 comparison | `python repro/replay.py b300` |
| Full-model forward and backward | `python repro/replay.py end-to-end` |

To execute a plan, choose one system and GPU count per allocation. For example, on
each of four 8-GPU H100 nodes, with `NODE_RANK=0..3` and the same `MASTER_ADDR`:

```bash
python repro/replay.py ablation --system ours --world 32 --execute --out artifacts/new-ablation
```

Use the MARBLE environment for `--system ours`, the isolated MagiAttention
environment for `magi`, and FlashMask's for `flashmask`, one at a time. The runner:

- checks that each subprocess succeeds and produces result rows;
- saves per-node logs;
- appends the newly stamped rows to files named after the original report files.

Use a fresh output directory per allocation, then rebuild the tables with
`repro/tables.py ... --results <directory>`. FlashMask balanced, hybrid and
overlap rows use the committed attention-only FlashMask driver; model rows use the
full-model adapter. B300 overlap needs the full FlashMask and NCCL setup described
below.

## Install a baseline in isolation

Each recipe refuses an existing target directory and leaves the MARBLE environment
alone. Append `--dry-run` to preview. CUDA builds need a matching `nvcc`, CMake, a
compiler and architecture support. All four recipes pass install, import,
exact-pin and package-consistency checks on H100; B300 has not been validated yet.
These setup checks do not certify benchmark timings.

```bash
python repro/setup_baseline.py flashmask --gpu h100 --root /tmp/marble-baselines
python repro/setup_baseline.py magi --gpu h100 --root /tmp/marble-baselines
python repro/setup_baseline.py te --gpu h100 --root /tmp/marble-baselines
python repro/setup_baseline.py yunchang --gpu h100 --root /tmp/marble-baselines
```

Use `--gpu b300` for FlashMask, MagiAttention and TransformerEngine. The yunchang
baseline uses Hopper FA3 and is rejected on B300.

| Baseline | Pins and implementation |
|---|---|
| FlashMask | `6a150fc`, torch `2.11.0+cu129`, `nvidia-cutlass==4.2.0.0`, `nvidia-cutlass-dsl[cu13]==4.5.2`; torch frontend, `core+cpb` |
| MagiAttention | `de13b2a`; its pinned requirements, including CuTe DSL `4.4.2` and quack `0.4.1`; FFA on H100, vendored FFA_FA4 on B300 |
| TransformerEngine | `2.8.0`; `NVTE_CUDA_ARCHS=90` on H100, `103` on B300 |
| yunchang | `0.6.4`, FA3 `3.0.0`; the benchmark driver patches its causal-flag bug |

Each recipe writes `setup.json`, `resolved-requirements.txt` and `activate.sh` into
its target, recording the resolved checkout and package closure; keep them with
new measurements. Notes on specific recipes:

- CUDA Python `12.9.4` and bindings `12.9.7` are pinned, because otherwise CuTe's
  resolver can upgrade the bindings to CUDA 13, which conflicts with torch cu129.
- Top-level pins are fixed. The remaining transitive packages resolve on first
  setup and are recorded.
- FlashMask's optional overlap bridge needs a separate NCCL >= 2.30.5 installation
  and is not built by the all-gather recipe.
- MagiAttention compiles for CUDA 12 to match torch cu129. It needs Python >= 3.12
  (`typing.override`), so its recipe uses 3.12.14 and installs the undeclared
  `debugpy==1.8.20` dependency. Bulk FFA prebuild is disabled; kernels compile
  during warm-up, before timing.
- TransformerEngine builds against torch's bundled NCCL headers and libraries. Its
  recipe uses Python 3.10.19 to match the cp310 tag inside the pinned TE 2.8 wheel
  (whose filename says py3-none). The TE benchmark does not need MARBLE installed.
- FA3 installation takes public dependencies missing from torch's wheel index from
  PyPI.

Activate a baseline with `source /tmp/marble-baselines/<baseline>-<gpu>/activate.sh`.
Keep `FLASHMASK_DIR` and `MAGI_DIR` exported so that result stamps include the
baseline's Git revision and dirty flag. Install MARBLE with `--no-deps` only when a
driver needs it, so MARBLE's FA4 and CuTe dependencies do not replace a baseline's
pins. In a baseline environment, run commands with its `python`, not `uv run`.

## Collect new synthetic measurements

MARBLE examples (eight GPUs):

```bash
uv run --frozen python benchmarks/sweep.py ablation --out artifacts/ablation.jsonl
uv run --frozen python benchmarks/sweep.py heads8 --out artifacts/heads8.jsonl
uv run --frozen python benchmarks/sweep.py model8 --out artifacts/model8.jsonl
```

Baseline sweeps, after activating the matching environment:

```bash
python benchmarks/sweep.py te8 --out artifacts/te8.jsonl
python benchmarks/sweep.py yunchang8 --out artifacts/yunchang8.jsonl
python benchmarks/sweep.py magi8 --out artifacts/magi8.jsonl
python benchmarks/baselines/sweep_magi.py --out artifacts/magi-tuned
```

Run MagiAttention's `--check` option for a single correctness check before timing.
FlashMask's attention-only driver is `benchmarks/baselines/flashmask_cp_bench.py`;
the full-model driver accepts `--attn flashmask`. The original H100 argument lists
and B300 commands are in `benchmarks/baselines/flashmask_runs/`.

### FlashMask environment used for the measurements

`benchmarks/baselines/flashmask_runs/setup_flashmask.sh` recreates the environment
the FlashMask rows were measured in:

- Python 3.12.14, numpy 2.5.4, setuptools 81.0.0 and TVM FFI 0.1.14.post1;
- torch 2.11.0+cu129, CUTLASS 4.2.0.0 and CuTe DSL 4.5.2.

Run it in a fresh directory:

```bash
mkdir /tmp/marble-flashmask
bash benchmarks/baselines/flashmask_runs/setup_flashmask.sh h100 /tmp/marble-flashmask
```

For B300, pass `b300`. It builds every component with NCCL cu13 2.32.3, which the
overlap path needs; follow the printed library-path instructions and the B300
launcher. The isolated `repro/setup_baseline.py` recipe above checks the core pins
on a fresh Python 3.11 stack and builds balanced context parallelism only. It does
not reproduce the additional version choices of this script.

## Numerical correctness

```bash
uv run --frozen pytest -m 'not benchmark' -q
uv run --frozen python repro/check_exactness.py --gpu h100 --out artifacts/h100-exactness
uv run --frozen python repro/check_exactness.py --gpu b300 --out artifacts/b300-exactness
```

Each distributed command needs eight visible GPUs (or `--nproc 4`). The matrix
covers:

- native causal and full masks;
- packed documents, mixed masks and grouped masks;
- Ulysses, ring and hybrid GQA;
- head dimensions 128 and 256, and batch size two;
- gather, the exact generation ring and sliding windows.

The runner checks the output and every Q/K/V gradient against fp32. Mean-relative
errors must stay within 1.05× the single-GPU FA4 error plus `1e-6`, under an
absolute ceiling of `1e-2`. A missing reference, NaN or Inf, a subprocess failure
or a timeout fails the gate. The GPU-specific interval dense-ring options are
opt-in, and this gate does not label that path exact.

The committed [H100 validation rows](validation/h100.jsonl) contain all 14 passing
cases (42 rows) with their software and hardware metadata. B300 validation is
pending.

Known limits of the gate:

- **Full multi-head attention.** The strict ring and gather gate uses GQA, because
  FA4 keeps fp32 dK/dV partials for GQA but rounds MHA dK/dV at each step. An H100
  eight-rank causal MHA check measured a dK mean-relative error of 0.00232727,
  against 0.00166687 for single-GPU FA4 (1.40×). That fails the 1.05× budget, so
  MHA ring and gather are not certified exact. Ulysses-only cases keep MHA
  coverage.
- **Head dimension 256.** On H100 it is checked through packed causal gather with
  dense kernels, as used in the paper; a plain full document keeps the native ring
  even when gather is requested. The online ring at head dimension 256 produced an
  invalid CUDA launch on the locked FA4 and CuTe stack and is not certified. B300
  head dimension 256 uses the fp32 ring path.

Use `--case <name>` to rerun one case.

The gate also exposed a dense-gather bug on noise-only spans: shared-prefix and
span calls were marked disjoint, which overwrote partial outputs. The merge
decision is fixed and covered by a regression test. Older rows do not record
enough of the plan to tell whether they hit this condition; rerun affected
dense-gather measurements with the current code.

## Dense-ring configuration

Four process-wide environment variables were replaced by plan-scoped `USPConfig`
fields:

| Old variable | Configuration field | Default |
|---|---|---|
| `USP_DENSE_RING_CHUNKS` | `dense_ring_chunks` | `None` (automatic) |
| `USP_DENSE_RING_REFINE` | `dense_ring_refine` | `True` |
| `USP_DENSE_RING_INPLACE` | `dense_ring_inplace` | `False` |
| `USP_DENSE_RING_DKV_OVERLAP` | `dense_ring_dkv_overlap` | `False` |

The old variables no longer affect execution. Benchmark flags with the
corresponding names set the fields directly; `--no-dense-ring-refine` turns
refinement off. The in-place and overlapping dK/dV options are experimental
interval-ring options. The backward pass reads the choice saved in its plan, even
if the environment changes later.

## Stamping new measurements

New benchmark and correctness rows get a `provenance` stamp containing:

- the Git revision and dirty flag of your checkout, counting untracked files;
- torch, FA4, CuTe DSL and NCCL (distribution and loaded runtime);
- the CUDA build version and `nvcc`;
- the driver, and the visible GPU models, compute capabilities and memory;
- Python, the command, and the relevant execution variables.

Launchers that run without `.git` can pass the revision in `MARBLE_GIT_COMMIT` (a
full SHA, optionally followed by `-dirty`). Unavailable fields are null with a
reason; optional baseline packages may be absent. The measuring worker makes the
stamp, and the sweep collectors keep it. The CUDA version is torch's build
version. The stamp does not collect GPU serial numbers, Git remotes, user names
or credentials.
