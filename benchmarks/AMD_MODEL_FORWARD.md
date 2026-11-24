# AMD full-decoder forward benchmark

`bench_amd_model_forward.py` reuses the article's `MoTConfig` and decoder layers
from `bench_model.py`: 28 layers, hidden width 3584, intermediate width 18944,
28 query heads, four KV heads, and head dimension 128. Each layer has separate
understanding and generation experts, each with Qwen2.5-7B dimensions. The
“7B” label describes an expert's dimensions; both experts' weights are stored.
Weights and hidden inputs are random, with generated-image tokens routed to
the generation expert. This measures execution, not model quality.

The timed forward includes every decoder layer: FSDP2 weight gathers, expert
routing, norms, Q/K/V and output projections, RoPE, distributed AMD sparse
attention, residuals, and SwiGLU MLPs. Strided QKV projection views are
packed into contiguous attention inputs inside the timed region for every
configuration, as required by ring P2P communication. Parameters are sharded across all ranks,
and each layer reshards after forward. The run uses `torch.no_grad()` and
`model.eval()`, with no activation checkpointing, backward, loss or optimizer.
Embeddings, modality encoders and output heads are outside this decoder-stack
benchmark, as in the original model benchmark. This is a full packed-sequence
forward, not a KV-cached autoregressive generation benchmark.

Plans, padding, routing indices and RoPE tables are built once outside timing.
Padding is an isolated causal document; its compute is timed, while throughput
counts useful tokens only. The default mask is the article's synthetic `mixed`
layout. Full-model inputs are generated per rank; different configurations do
not constitute a numerical parity comparison of the decoder.

Before timing, every configuration passes the existing distributed 6144-token,
all-row attention check against an FP32 reference and a finite-output check
through the requested decoder depth. These checks do not establish full-model
FP32 parity.

Each iteration records host wall time with GPU synchronization and GPU-event
time, then takes the maximum across ranks for each metric. The primary result
is median wall time, including Python dispatch and communication completion.
Peak allocated GPU memory is also the maximum across ranks. JSONL rows include
all samples, hardware/software, source/binary fingerprints and layout metadata
hashes. Timings are forward-only and cannot be compared directly with the
article's forward-plus-backward seconds per step or training TFLOP/s.

Run from `projects/usp_attention` in the ROCm environment with the AMD extension
built (see the [AMD backend README](../src/usp_attention/online_flash_amd/README.md)):

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=8 benchmarks/bench_amd_model_forward.py \
  --seq-len 131072 --config usp --kv-exchange gather \
  --output benchmarks/results/amd_model_forward_128k.jsonl
```

Repeat with `--config ring_contiguous` and `--config ring_ptrr` for the two ring
baselines. Both use ring KV rotation and unfurled query heads. USP defaults to
R2/U4 on eight GPUs, full KV gather and seven query heads folded into each
query sequence. All configurations default to `rtna` rounding. `--kv-exchange
ring --no-pack-mqa` selects the ordinary USP ring path. Gather's additional KV
and plan memory is included in peak allocation. The benchmark uses RCCL for
communication, including FSDP gathers.

For a quick smoke run, add `--seq-len 6144 --layers 1 --warmup 1 --trials 1
--iterations 2`. The default depth is all 28 layers, with three warmups and
three trials of ten iterations. Results append to the output file; use a new
file when changing the experiment.

## Context-length results

Full 28-layer decoder forward on eight MI325X/gfx942 GPUs (one node),
with PyTorch 2.11.0+rocm7.1, HIP 7.1.52802, BF16, synthetic `mixed` layout,
seed 0 and `rtna`. All configurations use two warmups and three trials of three
timed forwards (`--warmup 2 --trials 3 --iterations 3`), plus a separate
finite-output validation forward before warmup. Each latency is the median of
nine per-iteration maximum-rank wall times. The 128k results are the previous
validation run; 256k, 512k and 1M were measured in the subsequent sweep. All
12 records have identical benchmark/model/backend source and extension hashes.

| Context | Naive ring | Ring + PTRR | USP R2/U4, KV gather | USP vs PTRR latency | Naive ring / USP |
|---|---:|---:|---:|---:|---:|
| 128k | 1.224 s | 0.834 s | 0.812 s | 2.6% lower | 1.51× |
| 256k | 3.314 s | 2.034 s | 2.043 s | 0.5% higher | 1.62× |
| 512k | 10.381 s | 6.000 s | 5.903 s | 1.6% lower | 1.76× |
| 1M | 37.097 s | 20.094 s | 19.380 s | 3.6% lower | 1.91× |

Peak **allocated** memory in GiB (maximum across ranks; allocator reserves
and other processes are excluded):

| Context | Useful / padded tokens | Naive ring | Ring + PTRR | USP |
|---|---:|---:|---:|---:|
| 128k | 131,072 / 132,096 | 7.67 | 7.20 | 7.24 |
| 256k | 262,144 / 264,192 | 10.47 | 9.24 | 9.37 |
| 512k | 524,288 / 525,312 | 16.02 | 13.65 | 14.13 |
| 1M | 1,048,576 / 1,050,624 | 27.22 | 22.45 | 24.28 |

All four lengths and all three configurations completed without OOM. Every
attention FP32-reference gate passed (relative L2 approximately 0.0022), and
every requested full-depth decoder output was finite. The full decoder itself
was not compared against an FP32 model. A one-layer 6144-token USP smoke run
also passed before the original full-depth validation. These are forward-only
measurements on one node and one synthetic layout, not training-step timings
or pretrained-model quality results.

### Variation across trials

Ranges below are the minimum and maximum of the three trial medians, in
seconds. These describe variation within each run, not confidence intervals
from repeated independent sweeps.

| Context | Naive ring | Ring + PTRR | USP |
|---|---:|---:|---:|
| 128k | 1.2229–1.2249 | 0.8335–0.8351 | 0.8069–0.8120 |
| 256k | 3.3090–3.3150 | 2.0337–2.0349 | 2.0427–2.0434 |
| 512k | 10.3774–10.4015 | 6.0001–6.0039 | 5.8982–5.9037 |
| 1M | 37.0942–37.0982 | 20.0868–20.0967 | 19.3793–19.3814 |

### Raw results and reproduction

- [128k records](results/amd_model_forward_128k.jsonl)
- [256k records](results/amd_model_forward_256k.jsonl)
- [512k records](results/amd_model_forward_512k.jsonl)
- [1m records](results/amd_model_forward_1m.jsonl)

For each configuration and length, run the command above with
`--warmup 2 --trials 3 --iterations 3`. Select `--config ring_contiguous`,
`ring_ptrr`, or `usp`, and `--seq-len 131072`, `262144`, `524288`, or `1048576`.
Use a separate output file per context length and append the three configurations.
Plan setup, synthetic input construction and metadata preparation remain outside
the timed region. No environment tuning was added for this sweep.

Provenance shared by all 12 records:

- Base commit: `0484b8dde3005343ea503f3c276df9a13db1181f` (working tree contains uncommitted optimization and benchmark changes).
- Benchmark/model/AMD source SHA256: `429022de33d905ad5a34f2949bf870744558ac4b5b5a2cb8666615e3bf7a66cb`.
- AMD extension SHA256: `12ce1adfb932473ddea05393c1412bee39757d2dda82c8a25702e552a613e47b`.
