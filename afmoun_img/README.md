# AF-Muon Image Experiments

This folder contains the ImageGPT-style ImageNet-32 experiment used in the
paper. The setting uses a deterministic RGB554 tokenizer and intentionally ties
the input color-token table with the output classifier table.

## Setting

- Dataset: `benjamin-paine/imagenet-1k-32x32`
- Tokenizer: deterministic RGB554
- Color vocabulary: `16384`
- Full vocabulary: `16385` rows including one SOS row
- SOS token: `16384`, used only as an input token
- Sequence length: `1024` image tokens
- Architecture: 8 layers, width 512, 8 heads, head dimension 64, MLP hidden size 2048
- Parameters: 33,563,648
- Train budget: 2,500 updates, 1,310,720,000 image tokens
- Seeds: 43, 44, 45
- Optimizers: Hybrid Muon, SCION-style Sign, AF-Muon
- Batch: `batch_size=512`, `micro_batch_size=256`, 2 accumulation steps
- Evaluation: 16 micro-batches per evaluation, 4,194,304 image tokens total
- Precision: FP32 trainable parameters and optimizer state with BF16 autocast

## Files

- `scripts/prepare_imagenet32_rgb554_tokens.py`: builds the local RGB554 token cache.
- `train_tied_imagegpt.py`: trains one optimizer arm.
- `run_tied_imagegpt.py`: launches the paired optimizer arms across seeds.
- `summarize_tied_imagegpt.py`: summarizes completed runs.
- `make_completion_grid.py`: generates fixed top-half completion grids from a checkpoint.

The raw dataset and generated token cache are not included in this release.
