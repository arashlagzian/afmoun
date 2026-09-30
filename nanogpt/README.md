# NanoGPT AF-Muon Experiments

Self-contained NanoGPT experiments for tied token embeddings. Everything in this
folder is isolated from the main release code so optimizer/model variants can be
changed without touching the paper runs.

The defaults follow the NanoGPT settings reported in the SCION paper table and
the cited Muon/modded-nanogpt record family:

| Hyperparameter | Default |
| --- | --- |
| layers | 12 |
| head dim | 128 |
| heads | 6 |
| width | 768 |
| activation | ScaledReLU^2 for all paper arms |
| vocabulary size | 50304 |
| dataset | FineWeb token memmap |
| batch size | 512 sequences |
| block size | 1024 |
| iterations | 5100 |
| warmdown | 0% for the matched paper protocol |
| warmup | 0 |
| gradient clipping | norm 1.0 for the matched paper protocol |
| matrix radius knobs | rho_hidden=50, rho_output=3000 |
| Muon stepsize multiplier | 1.0 for the matched paper protocol |
| matrix weight decay | 0.1, matched across all arms |
| auxiliary weight decay | configurable per arm |

Important dtype convention: parameters and optimizer states stay FP32. The
default `--autocast bf16` uses BF16 autocast only for forward/backward compute
and needs no loss scaler. If `--autocast fp16` is selected, the trainer enables
a CUDA GradScaler before the optimizer step.

The matched paper protocol uses `--activation-mode all_scaled`, so Hybrid Muon,
SCION-style Sign, and AF-Muon all use the same `ScaledReLU^2` architecture.
The implementation still exposes `scion_table` and `all_relu` for architecture
sensitivity checks, but those modes are not the matched protocol reported in
the main paper.

For our controlled comparison, `--matrix-weight-decay 0.1` is applied to the
matrix branch of every arm. SCION's constrained formulation is not literally
AdamW-style weight decay, so this is a matched-protocol choice in our optimizer
wrapper.
For the T5-small-matched paper protocol, launch Hybrid Muon with auxiliary
AdamW weight decay 0.01 and launch SCION-style Sign / AF-Muon with auxiliary
weight decay 0.0. The launcher supports this directly with
`--hybrid-aux-weight-decay 0.01 --sign-aux-weight-decay 0.0
--afmoun-aux-weight-decay 0.0`.

Example three-arm run:

```bash
cd <repo>

AFMOUN_WORKER_PYTHON=<python> \
<python> nanogpt/run_nanogpt.py \
  --gpus 0 1 2 \
  --data-dir <fineweb_token_cache> \
  --run-root runs/nanogpt_scion_recipe_seed43 \
  --seeds 43 \
  --arms "muon,sign,afmoun" \
  --autocast bf16 \
  --activation-mode all_scaled \
  --warmdown-frac 0 \
  --hybrid-aux-weight-decay 0.01 \
  --sign-aux-weight-decay 0.0 \
  --afmoun-aux-weight-decay 0.0
```

The default `--multiplier-mode muon_only` keeps the hidden-matrix Muon learning
rate at the matched value while varying only the Muon-specific multiplier when
requested. The alternative `--multiplier-mode all` is retained for sensitivity
checks; the choice is saved in each run config.

CPU-only configuration check before using GPUs:

```bash
<python> nanogpt/train_nanogpt.py \
  --data-dir /tmp/unused \
  --output-dir /tmp/unused \
  --arm afmoun \
  --check-config
```

Summarize:

```bash
<python> nanogpt/summarize_nanogpt.py \
  --run-roots runs/nanogpt_scion_recipe_seed43 \
  --csv-out outputs/nanogpt_scion_recipe_seed43.csv
```
