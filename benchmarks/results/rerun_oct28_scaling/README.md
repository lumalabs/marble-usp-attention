# Strong-scaling fill-in on the 2026-10-28 measurement settings

Rows that complete the fixed-length strong-scaling chains for every system, measured on the released code with
the container image's NCCL 2.29.7 preloaded (`LD_PRELOAD=/opt/nccl/build/lib/libnccl.so`; `NCCL_DEBUG=VERSION`
confirmed 2.29.7 in the canary log) and `--halo-overlap` on every gather configuration. H100 nodes of 8 GPUs,
2026-10-28, one job per `benchmarks/sweep.py` sweep (`rerun_*`), 139 rows.

| file | sweep | GPUs | what |
|---|---|---|---|
| `rerun_fill128.jsonl` | fill128 | 128 | MARBLE 2M, Gemma 3 / Molmo2 mixed+video, R16U8 gather + overlap |
| `rerun_fill128_fm.jsonl`, `rerun_fill128cp_fm.jsonl` | fill128_fm, fill128cp_fm | 128 | FM+U (U8) and FM CP on the same 2M layouts |
| `rerun_magi32.jsonl`, `rerun_magi64.jsonl`, `rerun_magi128_2m.jsonl`, `rerun_magi128.jsonl` | magi32/64/128_2m/128 | 32/64/128 | MagiAttention 2M (32/64/128) and 4M (64/128), 8 layouts, chunk 4096 and 8192 |
| `rerun_fill32.jsonl`, `rerun_fill32_fm.jsonl` | fill32, fill32_fm | 32 | 4M at 32 GPUs: MARBLE (gather + overlap), FM+U (U8/U16), FM CP; BAGEL 8M FM rows |
| `rerun_fill32_bagel.jsonl`, `rerun_fill64.jsonl` | fill32 (bagel), fill64 | 32/64 | BAGEL video 8M at 32, 4M at 64 (chunked ring) |
| `rerun_oom32_bagel.jsonl`, `rerun_oom64_bagel.jsonl` | oom32, oom64 | 32/64 | BAGEL video ladder 1M-16M, chunked ring |
| `rerun_oom32_fm_bagel.jsonl`, `rerun_oom64_fm_bagel.jsonl` | oom32_fm, oom64_fm | 32/64 | FM CP and FM+U (U4) on the same ladder (16M: out of memory) |
| `rerun_canary.jsonl` | - | 8 | mixed 128k R2U4 gather + overlap, 49.3 ms |

Not retained: the NCCL banner of the sweep runs (`sweep.py` captures driver output only on error).

Rescoring these rows with the 32/64-GPU rows of `../rerun_oct28/mlsys/` gives, per doubling at a fixed sequence
length on the layouts all three systems have, 2M 64->128 GPUs (Gemma 3 and Molmo2, mixed and video): MARBLE 89%,
FM+U 46%, FM CP 49% parallel efficiency.

Independent spot check (2026-10-29, a fresh set of 16 nodes, same settings and the arguments of the 10-28 64-GPU rows
with ring 16): `spotcheck.jsonl`. At 2M tokens on 128 GPUs, MARBLE Molmo2 mixed 25.5 ms (25.0 above), FM+U Gemma 3
mixed 172.5 ms (177.9), FM+U Molmo2 mixed 100.8 ms (103.1), all within 3%. MARBLE Gemma 3 mixed hit a
first-collective NCCL timeout twice on that node set.

Confirmation (2026-10-30): two further relaunches of the `fill128` sweep failed with an NCCL remote error
(`ncclRemoteError: remote process exited or there was a network error`). The failing allocations shared one node
whose NCCL init could not bind NVLS multicast memory (`CUDA error 802 'system not yet initialized'`, Fabric Manager /
NVSwitch); it was taken out of service. Rerun with `NCCL_NVLS_ENABLE=0` (sweep `confirm_fill128_nonvls`):
`confirm_nonvls.jsonl`, Gemma 3 mixed 27.4 ms (27.4 above), Gemma 3 video 36.5 (35.3), Molmo2 mixed 25.2 (25.0),
Molmo2 video 39.3 (39.5), all within 3.5%. The paper uses the `rerun_*` rows (supplement Table S9).
