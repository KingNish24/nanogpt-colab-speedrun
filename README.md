# nanogpt-colab-speedrun

A single-GPU (Google Colab T4, 16 GB) port of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
record #360 (track 1, `track_1_short/`, ANVIL2): train the same architecture from scratch on FineWeb
toward **≤3.5 canonical-masked validation cross-entropy** — upstream reaches 3.28 on 8×H100; this
port relaxes the target, shrinks the data and the run, and cuts everything that does not fit one
16 GB Turing GPU.

## What's kept, what's cut

Kept (the algorithm): Muon/ANVIL twin-rail optimizer, five-stage schedule with YaRN window warmup,
value embeddings, multi-token prediction, prefix-token loss, sampled softmax, smear/XSA/gates/
paired heads/MUDD, mixed-width attention, tail weight averaging, canonical validation mask,
Triton kernels, the data pipeline.

Cut (the hardware):

| upstream (#360, 8×H100) | this port (1×T4) |
|---|---|
| bf16 | **fp16**, static `LOSS_SCALE=1024` + isfinite guard (skip step, halve scale) — no GradScaler |
| FP8 head/MLP/QKV, fused CUDA CE kernel | fp16 GEMMs, pure-PyTorch chunked softcapped CE (`losses.py`) |
| FlashAttention-3 (sm90) | `F.scaled_dot_product_attention` + additive fp16 bias, per segment |
| `torch.distributed` (8 ranks) | one process, `world_size=1`; every reduce is a copy |
| CUDA graphs + `torch.compile` | fully eager |
| n-gram/bigram table (16.2 GB shard) | deleted (upstream's own baseline predates it) |

Consequences of the cuts, in `train_gpt.py`:
- each stage batch is split into 16,384-token microbatches whose gradients accumulate into one
  optimizer step under the static loss scale;
- the sampled-softmax candidate set is built per microbatch (upload → gather → forward → backward
  → read-done), not per rank-step;
- validation runs in 65,536-token chunks so the CE slabs and SDPA bias caches fit 16 GB, with
  attention segments capped at `VIRTUAL_SEQ_CAP=2560` (the unaligned val stream has no cap, and an
  uncapped segment would need an unbounded `[L, L]` bias);
- rotary factor tables cover 65,536 positions (eval chunks) but rebuild only the first 16,384 rows
  on a window change, completed before each validation.

## Run it (Colab)

Open `port_t4.ipynb` in a Colab session with a T4 GPU (Runtime → Change runtime type → T4) and run
the cells in order: environment assert → uv venv (system-site-packages, reusing Colab's CUDA torch)
→ clone → data download → 2-minute smoke run → full run. Every cell is plain `!bash` / `!python`,
so the same commands work in any Linux shell.

Locally (any Linux box with an NVIDIA GPU + the torch/triton that matches it):

```bash
git clone <this repo> && cd nanogpt-colab-speedrun
uv venv --system-site-packages .venv && uv pip install --python .venv/bin/python -r requirements.txt
python data/cached_fineweb10B.py 4   # 4 × 100M-token train shards + the val shard (~800 MB)
python train_gpt.py
```

`NUM_SCHEDULED_ITERATIONS` overrides the step count (the notebook uses a small value for the smoke
run); `TRAIN_SEED` fixes the init; `DATA_PATH` relocates the data directory.

## Data and run

- train: 4 shards of the pre-tokenized `kjj0/fineweb10B-gpt2` set = **400M tokens** (~800 MB); the
  default schedule consumes ~287M of them, so the run never exhausts the shards;
- validation: `fineweb_val_000000.bin`, the fixed **1,048,576 tokens**, evaluated every 150 steps
  and at the end (final = tail-averaged weights + canonical mask + extended 20-block window);
- schedule: 1,080 + 52 growth + 20 extension steps ≈ 1,100 optimizer steps ≈ 287M tokens, same
  stage table as record #360;
- expected wall time on a T4: on the order of a few hours (the notebook prints `step_avg` from the
  first log line onward); logs land in `logs/<run_id>.txt` and start with the full source.

## Layout

`train_gpt.py` is the outline of a run (setup, model, warmup, timed loop, final validation);
`track_1_short/` holds the model, optimizer, data and schedules; `track_1_short/perf/` holds the
kernels and systems tricks — skippable when reading for the algorithm. Every run log embeds the
full source.

## Provenance

Derived from [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
(record #360, `track_1_short/`) by porting it to one Turing GPU; MIT license retained (see
`LICENSE`). Upstream contributors are credited in the upstream README.
