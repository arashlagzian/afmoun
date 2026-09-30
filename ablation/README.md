# AF-Muon Ablation Recipes

This directory contains the released ablation launchers, workers, summaries, and
plotting helpers used for the NanoGPT appendix controls. All commands assume they
are run from the repository root and that `<python>` points to the environment
with the dependencies in `requirements.txt`.

Datasets are not included. Replace `<fineweb_token_cache>` with a local FineWeb
token cache in the layout described by the repository-level README.

## NanoGPT Factorial Tied-Table / Vector Control

This is the factorial ablation for the NanoGPT tied vocabulary table and the
remaining vector-like parameters. The four arms are:

| Arm | Tied vocabulary table | Vector-like parameters |
| --- | --- | --- |
| `fact_adamw_adamw` | AdamW-style auxiliary update | AdamW-style auxiliary update |
| `fact_c3_adamw` | finite-cap AF-Muon tied LMO | AdamW-style auxiliary update |
| `fact_adamw_rms` | AdamW-style auxiliary update | RMS-normalized update |
| `fact_c3_rms` | finite-cap AF-Muon tied LMO | RMS-normalized update |

The implementation is in `nanogpt/train_nanogpt.py`; the standard NanoGPT
launcher exposes the exact arm mapping.

```bash
AFMOUN_WORKER_PYTHON=<python> \
<python> nanogpt/run_nanogpt.py \
  --gpus 0 1 2 3 \
  --data-dir <fineweb_token_cache> \
  --run-root runs/nanogpt_factorial_tied_ablate_3seeds \
  --seeds 43,44,45 \
  --arms fact_adamw_adamw,fact_c3_adamw,fact_adamw_rms,fact_c3_rms \
  --iterations 2000
```

Summarize:

```bash
<python> nanogpt/summarize_nanogpt_paper.py \
  --run-root runs/nanogpt_factorial_tied_ablate_3seeds \
  --csv-prefix outputs/nanogpt_factorial_tied_ablate
```

## Cap / Scale Diagnostics

The cap/scale sweep uses the dedicated diagnostic worker
`ablation/train_nanogpt_vocab_effect.py`, which logs the finite-cap geometry
statistics: capped-coordinate fraction, clipped-row fraction, row-RMS tail
energy, cosine to Sign, and objective ratio.

Default launcher settings reproduce the three-seed, step-750 cap/scale grid.

```bash
AFMOUN_WORKER_PYTHON=<python> \
<python> ablation/run_nanogpt_cap_scale_sensitivity.py \
  --gpus 0 1 2 3 \
  --data-dir <fineweb_token_cache>
```

Summarize the exact comparison step:

```bash
<python> ablation/summarize_nanogpt_cap_scale_multiseed.py \
  --run-root runs/nanogpt_afmoun_cap_scale_sensitivity_3seeds_step750_matchbatch524k \
  --eval-step 750 \
  --expected-seeds 43,44,45
```

## Auxiliary / Fallback LR Sensitivity

The auxiliary/fallback LR sweep keeps the hidden-matrix Muon learning rate fixed
at `0.02` and varies the vector/fallback learning rate.

```bash
AFMOUN_WORKER_PYTHON=<python> \
<python> ablation/run_nanogpt_auxlr_sensitivity.py \
  --gpus 0 1 2 3 \
  --data-dir <fineweb_token_cache>
```

Summarize:

```bash
<python> ablation/summarize_nanogpt_auxlr_sensitivity.py \
  --run-root runs/nanogpt_auxlr_sensitivity_seed43_500m_matchbatch524k
```

## Batch-Size Sensitivity

The paper comparison uses a common token point. The summary requires an exact
match by default when `--target-tokens` is supplied.

```bash
AFMOUN_WORKER_PYTHON=<python> \
<python> ablation/run_nanogpt_batch_sensitivity.py \
  --gpus 0 1 2 3 \
  --data-dir <fineweb_token_cache>

<python> ablation/summarize_nanogpt_batch_sensitivity.py \
  --run-root runs/nanogpt_batch_sensitivity_seed43_500m \
  --target-tokens 458752000
```

## Tied-Table Gradient Source

This ablation replaces the tied-table gradient before the standard global
gradient-clipping operation. Non-tied parameters keep their usual optimizer
rules.

```bash
AFMOUN_WORKER_PYTHON=<python> \
<python> ablation/run_nanogpt_tied_source_ablation.py \
  --gpus 0 1 2 \
  --data-dir <fineweb_token_cache>

<python> ablation/summarize_nanogpt_tied_source_ablation.py \
  --run-root runs/nanogpt_tied_source_ablation_seed43_500m_matchbatch524k
```

## T5 Sentinel Factorial

`ablation/train_tied_factorial.py` is a separate T5-style encoder-decoder
sentinel retained for provenance. It is not the NanoGPT factorial table in the
paper.
