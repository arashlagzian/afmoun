# AF-Muon

This release contains code for AF-Muon, an AdamW-free Muon-family optimizer for
models with tied vocabulary/token tables. AF-Muon keeps Muon's spectral update
for hidden matrices, assigns the aliased embedding/output table a finite-cap
support-aware LMO, and uses RMS-normalized updates for remaining vector-like
auxiliary parameters.

The paper evaluates AF-Muon in nine tied-token settings:

1. NanoGPT-style decoder-only language modeling on FineWeb.
2. SmolLM2-135M decoder-only language modeling on FineWeb.
3. Qwen2.5-0.5B decoder-only language modeling on FineWeb.
4. Llama-3.2-1B decoder-only language modeling on FineWeb.
5. T5-style fully shared encoder-decoder vocabulary modeling on FineWeb.
6. T5Gemma2-style shared-vocabulary encoder-decoder modeling on FineWeb.
7. Sparse NanoGPT-MoE with a tied input/output vocabulary table.
8. ImageGPT-style ImageNet-32 modeling with a tied RGB554 color-token table.
9. Protein language modeling on UniRef50 / ProtGPT2-BPE.

Datasets, token caches, pretrained model weights, and machine-specific paths are
not included. To reproduce the experiments, prepare local token caches from the
public datasets and provide local Hugging Face-style config/tokenizer folders.
For from-scratch runs, the model config and tokenizer define architecture and
vocabulary metadata; pretrained weights are not required unless a script
explicitly asks for them.

## Optimizers

Core optimizer implementations:

- `optimizers/muon.py`: Hybrid Muon baseline. Hidden matrices use Muon; tied
  embeddings, heads, normalization parameters, and biases use AdamW-style
  auxiliary updates.
- `optimizers/scion.py`: SCION-style tied Sign endpoint inside the same
  Muon-family training framework, aliased to the V2 Sign endpoint used in the
  paper.
- `optimizers/afmoun_v2.py`: the released paper implementation of AF-Muon,
  including the chunked finite-cap tied-table clipped-row LMO. Public imports
  such as `build_afmoun` and `build_scion_sign` are aliases to this V2
  implementation.
- `optimizers/afmoun.py`: legacy V1 reference code kept unexported for
  provenance; it is not used by the paper experiments.

The NanoGPT experiments use isolated optimizer copies under
`nanogpt/optimizers/` for the paper ablations; their public imports are also
aliased to the V2 implementation.

## Default Optimizer Protocol

Unless stated otherwise, the matched paper protocol uses:

```text
hidden matrix Muon learning rate = 0.02
remaining 1D/vector learning rate = 3e-4
momentum = 0.95
matrix weight decay = 0.1
gradient clipping norm = 1.0
trainable parameters and optimizer state = FP32
forward/backward autocast = BF16
rho_hidden = 50
rho_output = 3000
rho_output / rho_hidden = 60
```

The tied vocabulary table is its own parameter group. It does not use the
ordinary `3e-4` vector learning rate. For SCION-style Sign and AF-Muon, the tied
table uses the tied-table LMO step size

```text
eta_tie = hidden_matrix_lr * (rho_output / rho_hidden) * tied_scale / d_model
```

AF-Muon tied-table default:

```text
tied_cap = 3
tied_scale = 0.5
eta_tie = 0.02 * 60 * 0.5 / d_model = 0.6 / d_model
```

Thus the three optimizer groups are separated: hidden 2D matrices use Muon with
learning rate `0.02`; the tied vocabulary table uses the Sign or AF-Muon LMO
with the effective `eta_tie` above; remaining one-dimensional/vector parameters
use learning rate `3e-4`.

For example, with `d_model=2048`,

```text
eta_tie = 0.02 * 60 * 0.5 / 2048 = 0.00029296875
```

Hybrid Muon uses auxiliary AdamW weight decay `0.01` where the matched protocol
calls for the original Hybrid treatment. SCION-style Sign and AF-Muon use zero
auxiliary/tied weight decay in the controlled LMO comparisons.

## Data and Model Recipe

The experiments use three public data sources, converted into local token/cache
files before training:

- **FineWeb text.** Decoder-only, T5-style, T5Gemma2-style, and MoE experiments use FineWeb text
  streams. The cache is tokenized with the tokenizer for the corresponding
  model family: SmolLM2 for NanoGPT/SmolLM2/T5/T5Gemma2/MoE-style settings,
  Qwen2.5 for Qwen2.5-0.5B, and Llama for Llama-3.2-1B. The release expects prebuilt
  contiguous token memmaps rather than raw documents.
- **ImageNet-32 images.** The ImageGPT-style experiment uses the
  `benjamin-paine/imagenet-1k-32x32` image dataset. Images are converted into
  sequences of 1024 discrete RGB554 color tokens from a 16k color vocabulary,
  with one additional SOS row used only on the input side.
- **UniRef50 proteins.** The protein experiment uses UniRef50 2021_04 with the
  ProtGPT2 BPE tokenizer. Amino-acid sequences are converted into BPE token
  memmaps for causal language modeling with a tied token embedding / LM head.
  The paper cache uses the tokenized sequences as provided and does not append
  an additional EOS token.

The raw datasets are not redistributed here. Users should download the public
datasets, tokenize them with the recipe above, and store the resulting memmaps
using the layout below.

Generic token-cache layout:

```text
data/<cache_name>/
  metadata.json
  train_tokens_uint16.bin or train_tokens_uint32.bin
  eval_tokens_uint16.bin or eval_tokens_uint32.bin
```

`metadata.json` records the dtype, block size, train path, and evaluation path.
Paths may be absolute or relative to the metadata file.

Generic model/config layout:

```text
models/
  <model_name>/
    config.json
    tokenizer.json
    tokenizer_config.json
    ...
```

See `data/README.md` and `models/README.md` for the minimal expected layouts.

## Nine Paper Settings

### 1. NanoGPT / FineWeb

Code: `nanogpt/`

Purpose: controlled decoder-only tied-embedding language modeling. The tied
table receives sparse input-lookup gradients and dense output-classifier
gradients.

```text
architecture: 12 layers, width 768, 6 heads, head_dim 128, MLP hidden 3072
parameters: 123,587,328
vocabulary size: 50304
context length: 1024
seeds: 43, 44, 45
optimizers: Hybrid Muon, SCION-style Sign, AF-Muon
train budget: 2.67B tokens; 2,673,868,800 actual tokens
eval tokens: 5M
batching: batch_size=512, micro_batch_size=16, gradient_accumulation_steps=32
effective batch: 524,288 tokens/update
steps: 5,100
eval_every_steps: 250
log_every_steps: 50
diagnostics: every 250 steps
precision: FP32 trainable parameters and optimizer state with BF16 autocast
gradient clipping: norm 1.0
```

Related ablation recipes released in `ablation/` cover the NanoGPT factorial
tied-table/vector update control, batch-size sensitivity, auxiliary/fallback LR
sensitivity, cap/scale diagnostics, and tied-table gradient-source ablations.

### 2. SmolLM2-135M / FineWeb

Code:

- `scripts/run_01_paper_smollm2_135m_v2.py`
- `scripts/train_01_paper_smollm2_135m_v2.py`

```text
architecture: SmolLM2-135M config
data: FineWeb token cache using the SmolLM2 tokenizer
parameters: 134,515,008
train budget: 2.5B requested tokens; 2,499,805,184 actual tokens
eval tokens: 5M
seeds: 43, 44, 45
optimizers: Hybrid Muon, SCION-style Sign, AF-Muon
batching: micro_batch_size=32, gradient_accumulation_steps=16, block_size=1024
effective batch: 524,288 tokens/update
steps: 4,768
eval_every_steps: 250
log_every_steps: 50
diagnostics: every 250 steps
precision: FP32 trainable parameters and optimizer state with BF16 autocast
gradient checkpointing: enabled
```

### 3. Qwen2.5-0.5B / FineWeb

Code:

- `scripts/run_01_paper_qwen25_nb90_v2.py`
- `scripts/train_01_paper_qwen25_nb90_v2.py`

```text
architecture: Qwen2.5-0.5B config
data: FineWeb sample-10BT token cache using the Qwen2.5 tokenizer
train budget: 1B requested tokens; 999,817,216 actual tokens
long-horizon check: seed 43 trained to 6.4B requested tokens; 6,399,983,616 actual tokens
long-horizon reported validation endpoint: last scheduled eval at 6.29B tokens
eval tokens: 5M
seeds: 43, 44, 45
optimizers: Hybrid Muon, SCION-style Sign, AF-Muon
batching: micro_batch_size=8, gradient_accumulation_steps=64, block_size=1024
effective batch: 524,288 tokens/update
steps: 1,907
eval_every_steps: 250
log_every_steps: 50
diagnostics: every 250 steps
precision: FP32 trainable parameters and optimizer state with BF16 autocast
checkpointing: disabled for this matched comparison
long-horizon checkpointing: rolling full checkpoint every 250 steps
```

### 4. Llama-3.2-1B / FineWeb

Code:

- `scripts/run_01_paper_llama32_nb98_v2.py`
- `scripts/train_01_paper_llama32_nb98_v2.py`

```text
architecture: Llama-3.2-1B config
parameters: 1,235,814,400
main comparison: three paired seeds, 2B requested tokens
main comparison actual tokens: 1,999,470,592
paper table note: seed 43 uses the long-horizon run's last pre-2B eval at 1.966B; seeds 44 and 45 use 2B short-horizon runs
long-horizon check: seed 43 continued to 9,983,508,480 tokens
eval tokens: 1M
seeds: 43, 44, 45
optimizers: Hybrid Muon, SCION-style Sign, AF-Muon
batching: micro_batch_size=32, gradient_accumulation_steps=16, block_size=1024
effective batch: 524,288 tokens/update
short-horizon steps: 3,814
long-horizon steps: 19,042
eval_every_steps: 250
log_every_steps: 50
diagnostics: every 250 steps
precision: FP32 trainable parameters and optimizer state with BF16 autocast
gradient checkpointing: enabled
```

### 5. T5-Style Fully Shared Encoder-Decoder / FineWeb

Code: `t5test/`

Purpose: three-way tying. One vocabulary table serves encoder input lookup,
decoder input lookup, and output classification.

```text
architecture: small T5-style encoder-decoder, dim 512, 8 heads
layers: 4 encoder layers, 4 decoder layers
feed-forward multiplier: 4
topology: fully shared encoder embedding, decoder embedding, and output head
vocabulary size: 49152
sequence layout: source_len=256, target_len=256
train budget: 275M predicted decoder target tokens
raw source-plus-target tokens: 550,002,688
eval tokens: 1M
seeds: 43, 44, 45
tied parameters: 54,561,792
untied Hybrid Muon control parameters: 104,893,440
optimizers: tied Hybrid Muon, SCION-style Sign, AF-Muon, untied Hybrid Muon control
batching: micro_batch_size=16, gradient_accumulation_steps=1
target tokens/update: 4,096
raw source-plus-target tokens/update: 8,192
steps: 67,139
eval_every_steps: 5,000
log_every_steps: 500
diagnostics: every 5,000 steps
selected tied-table ratio: rho_output / rho_hidden = 1
precision: FP32 trainable parameters and optimizer state with BF16 autocast
gradient clipping: norm 1.0
```

### 6. T5Gemma2-Style Shared Encoder-Decoder / FineWeb

Code: `t5test/`

Purpose: larger encoder-decoder transfer test for three-way shared vocabulary
optimization. The model is initialized from a Hugging Face T5Gemma2-style config
and trained from scratch on the same contiguous FineWeb token-cache interface.

```text
architecture: random-init google/t5gemma-2-270m-270m configuration
topology: shared encoder input, decoder input, and output vocabulary table
vocabulary size: 262k-row shared table
sequence layout: source_len=256, target_len=256
train budget: 750M predicted decoder target tokens
actual target tokens: 750,256,128
raw source-plus-target tokens: 1,500,512,256
eval tokens: 1M
seeds: 43, 44, 45
parameters: 786,029,296
tied vocabulary parameters: 167,772,160
optimizers: Hybrid Muon, SCION-style Sign, AF-Muon
batching: micro_batch_size=16, gradient_accumulation_steps=128
target tokens/update: 524,288
raw source-plus-target tokens/update: 1,048,576
steps: 1,431
eval_every_steps: 250
log_every_steps: 25
checkpointing: final checkpoint only
rho_output / rho_hidden = 60
precision: FP32 trainable parameters and optimizer state with BF16 autocast
gradient checkpointing: enabled
gradient clipping: norm 1.0
```

This script requires a `transformers` version with T5Gemma2 support. The paper
runs used a source build reporting `transformers 5.18.0.dev0`. The experiment is
random-init; pretrained weights are not loaded. Each run writes a `config.json`
with the requested vocabulary size and the resolved input/output embedding
shapes from the instantiated model.

### 7. Sparse NanoGPT-MoE / FineWeb

Code: `moe/`

This setting replaces each dense NanoGPT MLP with a top-1 router over four
expert MLPs while keeping the input/output vocabulary table tied.

```text
architecture: NanoGPT-style decoder, 12 layers, width 768, 6 heads, head_dim 128
parameters: 293,493,504
vocabulary size: 50304
context length: 1024
MoE: 4 experts, top-1 routing, expert MLP hidden size 3072
router auxiliary loss coefficient: 0.01
router z-loss coefficient: 0.0
train budget: 2,673,868,800 tokens
eval tokens: 5M
seeds: 43, 44, 45
optimizers: Hybrid Muon, SCION-style Sign, AF-Muon
batching: batch_size=512, micro_batch_size=16, gradient_accumulation_steps=32
effective batch: 524,288 tokens/update
steps: 5,100
eval_every_steps: 250
log_every_steps: 50
precision: FP32 trainable parameters and optimizer state with BF16 autocast
gradient clipping: norm 1.0
diagnostics: router entropy, expert fractions, load coefficient of variation
```

### 8. ImageGPT-Style ImageNet-32

Code: `afmoun_img/`

This setting uses a deterministic RGB554 tokenizer for ImageNet-32 and
intentionally ties the input color-token table with the output classifier table.

```text
dataset: benjamin-paine/imagenet-1k-32x32
tokenizer: deterministic RGB554
color vocabulary: 16384
full vocabulary: 16385 rows including SOS
SOS row: 16384, input-only
sequence length: 1024 image tokens
architecture: 8 layers, width 512, 8 heads, head_dim 64, MLP hidden 2048
parameters: 33,563,648
train budget: 1,310,720,000 image tokens
eval tokens: 4,194,304 per evaluation
seeds: 43, 44, 45
optimizers: Hybrid Muon, SCION-style Sign, AF-Muon
batching: batch_size=512, micro_batch_size=256, gradient_accumulation_steps=2
effective batch: 524,288 tokens/update
steps: 2,500
eval_every_steps: 250
log_every_steps: 50
diagnostics: every 250 steps
precision: FP32 trainable parameters and optimizer state with BF16 autocast
gradient clipping: norm 1.0
metrics: validation loss, validation bits/dim, top-half completion loss
```

### 9. Protein LM / UniRef50 + ProtGPT2-BPE

Code:

- `scripts/prepare_protgpt2_uniref50_cache.py`
- `scripts/train_protein_causal_lm.py`
- `scripts/run_protein_uniref50_three_arms.py`
- `scripts/run_protein_adamw_lr_sweep_queue.py`

```text
dataset: UniRef50 2021_04
tokenizer: ProtGPT2 BPE
splits: train for training, validation for evaluation
cache preparation: pass --train-split train --eval-split validation
architecture: randomly initialized 363M-parameter RoPE decoder
hidden size: 960
intermediate size: 2560
layers: 32
attention heads: 15 query heads, 5 key/value heads
vocabulary size: 50257
context length: 512
training budget: 500M requested tokens; 499,646,464 actual tokens
main comparison point: 458.8M training tokens, three paired seeds
eval tokens: 5M
seeds: 43, 44, 45
optimizers: tuned AdamW, Hybrid Muon, SCION-style Sign, AF-Muon
batching: micro_batch_size=32, gradient_accumulation_steps=32
effective batch: 524,288 tokens/update
steps: 953
eval_every_steps: 125
log_every_steps: 50
precision: FP32 trainable parameters and optimizer state with BF16 autocast
gradient checkpointing: disabled
gradient clipping: norm 1.0
AdamW pilot: lr in {1e-4, 3e-4, 6e-4, 1e-3}; selected lr=6e-4
AdamW beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1
```

## Notes and Limitations

- This package assumes tied input embedding / LM-head for the headline tied-table
  method.
- Untied output heads, learned absolute positional embeddings, and unrelated
  dense parameters are not treated as tied-table AF-Muon blocks.
- The tied-table implementation chunks complete rows and never splits a row;
  this preserves exact row RMS and clipping thresholds relative to the full-table
  computation.
- SCION-style Sign here is a controlled Sign endpoint inside the matched
  Muon-family training stack, not an exhaustive reproduction of every SCION
  configuration.
- The main comparisons use literature-motivated defaults plus targeted
  sensitivity diagnostics; they are not intended as compute-optimal tuning
  studies for every baseline on every model.
