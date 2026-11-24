# Packed layouts used by the benchmarks

Every 32-128 GPU result in `results/mlsys/` and every `file:` layout in the B300
runs reads one of the packs in `packs/`. A pack is gzipped JSON
`{"layouts": [{"document_lens", "split_lens", "attn_modes"}, ...]}`; results use
index `#0`. Result rows name the file as it sat on the cluster
(`layouts/<name>.json` or `layouts/<name>.json`);
pass `file:benchmarks/layouts/packs/<name>.json.gz#0` (or the `.json` path, which
finds the `.gz` next to it) to rerun a row:

```bash
torchrun --nproc-per-node 8 benchmarks/bench_attention.py --ring 2 --ulysses 4 \
    --layout file:benchmarks/layouts/packs/molmo2_video_1024k.json.gz#0 \
    --seq-len 1048576 --heads 32 --kv-heads 8 --head-dim 128
```

## What is in `packs/`

| Packs | Source | Generator |
|---|---|---|
| `molmo2_*`, `qwen3vl_*`, `qwen38_*`, `gemma3_*`, `gemma3ps_*` (`video`, `interleaved`, `mixed`; 128k-4M) | Molmo2-Cap captioning videos and OBELICS web documents, tokenized with each model's own vision and text rules (`token_rules.md`) | `stats.py` then `pack.py` |
| `bagel_mix_*`, `bagel_video_*` (1M-16M) | BAGEL's token setup (SigLIP2 understanding, FLUX VAE generation; noise then clean frames) | `bagel.py` |
| `bagel_mix_32k`, `bagel_video_32k`, `bagel_video_32k_f1k` | small BAGEL-style packs for the accuracy checks | not scripted; committed as used |
| `synth_<name>_<len>` | `usp_attention.layouts.build_layout("<name>", len, seed=0)` | the library |

`SUMMARY.md` lists per-pack statistics (documents per sequence, vision-token
share, mask density); `raw_stats.json` summarizes the document statistics.

## Regenerating

The packs are deterministic given the inputs; regenerating should reproduce the
committed files.

1. Document statistics (needs about 3.5 GB of public parquet shards):

   ```bash
   mkdir -p dl out && cd dl
   for i in 0 1 2 3 4 5 6; do curl -sL -o molmo2cap_000$i.parquet "https://huggingface.co/datasets/allenai/Molmo2-Cap/resolve/refs%2Fconvert%2Fparquet/default/train/000$i.parquet"; done
   for i in 0000 0175 0350 0525 0700 0875 1050 1400; do curl -sL -o obelics_$i.parquet "https://huggingface.co/datasets/HuggingFaceM4/OBELICS/resolve/refs%2Fconvert%2Fparquet/default/train/$i.parquet"; done
   curl -sL -o qwen3_tokenizer.json https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json
   cd .. && uv run --with pyarrow --with tokenizers python -I stats.py dl out
   ```

2. Model packs: `uv run -q --with numpy --with tokenizers python -I pack.py`
   (options `--models`, `--mixes`, `--seq-lens`; the token rules and their
   sources, the model repos' processor configs, are in `token_rules.md`).
3. BAGEL packs: `python bagel.py layouts`.
4. `check_mask.py` checks a pack's mask against the reference predicate.

Then `gzip -9 -n` each JSON into `packs/`.
