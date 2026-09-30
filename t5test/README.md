# Shared-Vocabulary Encoder-Decoder Experiments

This folder contains encoder-decoder tests where one vocabulary table is shared
across three roles:

- encoder input lookup,
- decoder input lookup,
- output classifier / LM head.

Both experiments use contiguous FineWeb token blocks. The encoder receives a
source prefix and the decoder predicts the following target segment. This keeps
the data path simple while exercising the three-way shared table.

The two settings answer complementary questions:

- **Small T5-style:** controlled encoder-decoder test with a tied-versus-untied
  comparison.
- **T5Gemma2-style:** transfer test for the same optimizer recipe in a larger,
  different encoder-decoder configuration.

Default optimizer protocol:

- all arms: hidden matrix WD = 0.1
- Hybrid Muon: shared vocabulary table and vector aux group use AdamW with WD = 0.01
- SCION-style Sign and AF-Muon: tied table and vector roles use WD = 0
- all arms: constant LR, FP32 parameters/state, optional BF16 autocast

## Small T5-Style Shared Vocabulary

The small setting is a controlled diagnostic with 4 encoder layers, 4 decoder
layers, hidden size 512, 8 attention heads, and a 49,152-row shared vocabulary
table. The selected fully shared paper setting uses
`rho_output / rho_hidden = 1`.

Recommended dry run for the paper-scale small shared-vocabulary setting:

```bash
cd <repo>

AFMOUN_WORKER_PYTHON=<python> \
<python> t5test/run_three_way_shared_vocab.py \
  --gpus 5 6 7 \
  --data-dir <fineweb_token_cache> \
  --run-root runs/t5test_small_5xparams_3seeds_rho1 \
  --seeds 43 44 45 \
  --arms muon,sign,afmoun \
  --vocab-size 49152 \
  --train-tokens 275000000 \
  --eval-tokens 1000000 \
  --micro-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --rho-output-values 50 \
  --dry-run
```

If the dry run is clean, remove `--dry-run`.

## T5Gemma2-Style Shared Vocabulary

The larger setting uses the random-init `google/t5gemma-2-270m-270m`
configuration through Hugging Face `transformers`. It uses a 262k-row shared
vocabulary table, source length 256, target length 256, and 750M predicted
decoder target tokens. The matched large-batch protocol uses 524,288 target
tokens per update (`micro_batch_size=16`, `gradient_accumulation_steps=128`).

This experiment requires a `transformers` version that supports T5Gemma2. The
paper runs used a source build reporting `transformers 5.18.0.dev0`. If the
installed release does not recognize `model_type=t5gemma2`, install a recent
source version of `transformers` in a separate environment. The run is from
random initialization; pretrained weights are not loaded. Each run writes a
`config.json` with the requested vocabulary size and the resolved input/output
embedding shapes from the instantiated model.

Recommended dry run:

```bash
cd <repo>

AFMOUN_WORKER_PYTHON=<python> \
<python> t5test/run_t5gemma2_style.py \
  --gpus 0 1 2 \
  --data-dir <fineweb_token_cache> \
  --run-root runs/t5gemma2_style_270m270m_750m \
  --seeds 43 44 45 \
  --arms muon,sign,afmoun \
  --train-tokens 750000000 \
  --max-steps 1431 \
  --micro-batch-size 16 \
  --gradient-accumulation-steps 128 \
  --eval-every-steps 250 \
  --log-every-steps 25 \
  --checkpoint-every-steps 1431 \
  --local-files-only \
  --skip-initial-eval \
  --dry-run
```

If the dry run is clean, remove `--dry-run`. Omit `--local-files-only` for the
first run if the T5Gemma2 config has not already been cached locally.
