# 8x B300 results, October 7 2026

One node, 8x NVIDIA B300 SXM6 (sm_103), driver 595.71.05, NVLink 5.

Environment differs from the lockfile in one pin: `nvidia-cutlass-dsl==4.7.0`
instead of 4.6.0 (FA4 b21 unchanged). With cutlass-dsl 4.6.x the SM100
FlashAttention-4 backward kernel deadlocks on Blackwell (an mbarrier wait that
never completes); 4.7.0 and 4.8.0 fix it. NCCL is PyTorch's bundled 2.28.9.

Every distributed run unsets the image's `LD_PRELOAD` of `/opt/nccl` (an NCCL
build without sm_103 code, which fails every collective with "named symbol not
found"). The online combine kernel is SM90 only, so ring configurations run
with `--combine python` (labels keep their H100 names).

- `parity.jsonl`: `tests/distributed/check_usp.py --no-fa4-reference` (fp32
  reference only) on the native causal / full layouts, flash and python
  combines, GQA, batch 2 and bf16 gradients.
- `parity_sparse.jsonl`: `check_usp.py` on the mixed and docs layouts with
  256x128 block sparsity (what FA4 needs on SM100), against fp32 and against
  single-GPU FA4.
- `gqa8.jsonl`, `heads8.jsonl`, `ablation.jsonl`: real head layouts and the
  32-head ablation. `*_ulysses4_dp2` rows run two sequences per step (Ulysses
  is capped at 4 GPUs by 4 KV heads), so their time covers two sequences.
- `torchcp8_match.jsonl`: our configurations with `--combine python`. For one
  plain document, ring configurations take the native striped path whatever
  their load-balance flag, so `ring8_python_contig` and `ring8_online_ptrr`
  run the same code here.
- `torchcp8.jsonl`: PyTorch context-parallel SDPA ring (cuDNN backend).
- `kernel_fa4_cudnn.jsonl`: single-GPU FA4 vs cuDNN SDPA, 4 heads x 128.
- `fp32_parity.jsonl`, `fp32_combine.jsonl`: `--combine fp32`, the Blackwell
  stand-in for the SM90 online kernel. Each ring step runs FA4 split-KV with
  two splits and keeps the fp32 per-split partials (captured before FA4's
  bf16 combine); steps merge in fp32. Accuracy matches single-GPU FA4 (output
  2.1e-3 vs 2.2e-3 mean relative error against fp32, where the python combine
  is 2.7-3.0e-3 and flash 2.5-3.8e-3) for at most 3% more step time than
  the python merge; `oct14/combine_default_check.log` measures it 4% slower
  than the flash combine on causal 128k (76.3 vs 73.2 ms).
- `te_checks.jsonl`, `te8.jsonl`: TransformerEngine 2.8.0 built for sm_103
  (`NVTE_CUDA_ARCHS=103`), in its own venv with the same torch. All nine
  forward plus backward checks pass.
- `torchcp8_flex.jsonl`: PyTorch flex-attention all-gather context parallel.
- `magi_checks.jsonl`, `magi8.jsonl`: MagiAttention at commit de13b2a with
  `MAGI_ATTENTION_KERNEL_BACKEND=fa4` (its FFA_FA4 backend), installed with
  its `scripts/install_flash_attn_cute.sh sm100` from its vendored FA4 fork.
  That fork does not import with cutlass-dsl 4.7 (`cute.core.ThrMma` moved),
  so Magi's venv keeps cutlass-dsl 4.4.2 and quack-kernels 0.4.1, the versions
  its B300 CI uses; its backward does not hang with 4.4.2. All four checks pass.
  On B300 Magi's FA4 backend is close to ours: 6% faster on mixed 512k
  (353.3 vs 377.0 ms for our Ulysses-8), and 2-26% slower elsewhere.

## MagiAttention tuning sweep

`benchmarks/baselines/sweep_magi.py` gives MagiAttention the same best-of-a-sweep
treatment as the other libraries on the cells where our lead is at most ~15%:
chunk size (512, 1024, 2048) x overlap degree (0, 1, 2, 4, dynamic), then its
`RANGE_MERGE`, `BWD_HIDE_TAIL_REDUCE`, `FLATTEN_HEAD_GROUPS` and (GQA only)
`CATGQA` switches, the three fastest re-timed at full length, and the winner
gated on the same correctness check. Outputs: `magi8_sweep.jsonl` (every timed
setting), `magi_checks_tuned.jsonl`, and the winners in `magi8_tuned.jsonl`
and `magi8_models_tuned.jsonl`, which sit next to the default-config files under
the same labels (take the fastest row per cell). Run in Magi's venv on one
8x B300 node from the project root:

    python benchmarks/baselines/sweep_magi.py --out benchmarks/results/b300

