# Tied-Embedding NanoGPT-MoE Probe

This folder is self-contained for the MoE ablation. It does not modify the
existing NanoGPT, Llama, Qwen, or SmolLM2 implementations.

## Question

Does AF-Muon's tied-table advantage survive when the transformer uses sparse
expert routing inside the FFN blocks?

The tied vocabulary table is still shared between sparse input lookup gradients
and dense output-head gradients. The MoE change only replaces the dense FFN
with routed expert FFNs.

## Paper Setting

Three seeds, 2.67B tokens, top-1 routing, 4 experts, same NanoGPT large-batch recipe:

- FP32 trainable parameters and optimizer state
- BF16 autocast
- 524k tokens/update: `batch_size=512`, `micro_batch_size=16`
- 50k SmolLM2/FineWeb tokenized data
- rho ratio 60: `rho_hidden=50`, `rho_output=3000`
- matrix weight decay 0.1
- Hybrid Muon auxiliary/tied AdamW weight decay 0.01
- SCION-style Sign and AF-Muon auxiliary/tied weight decay 0
- AF-Muon tied table: `c=3`, `s=0.5`

## Check Config

```bash
cd <repo>

<python> moe/run_moe_nanogpt.py \
  --gpus 0 1 2 \
  --data-dir <fineweb_token_cache> \
  --run-root runs/moe_nanogpt_50k_top1_e4_routeraux0p01_3seeds_2p67b \
  --seeds 43,44,45 \
  --arms muon,sign,afmoun \
  --check-config
```

Adjust `PY` and `DATA` for the local environment.

## Run

```bash
cd <repo>

mkdir -p logs/moe_nanogpt_50k_top1_e4_routeraux0p01_3seeds_2p67b

AFMOUN_WORKER_PYTHON=<python> \
nohup <python> \
  moe/run_moe_nanogpt.py \
  --gpus 0 1 2 \
  --data-dir <fineweb_token_cache> \
  --run-root runs/moe_nanogpt_50k_top1_e4_routeraux0p01_3seeds_2p67b \
  --seeds 43,44,45 \
  --arms muon,sign,afmoun \
  --autocast bf16 \
  --layers 12 \
  --heads 6 \
  --head-dim 128 \
  --mlp-hidden 3072 \
  --vocab-size 50304 \
  --block-size 1024 \
  --iterations 5100 \
  --batch-size 512 \
  --micro-batch-size 16 \
  --eval-tokens 5000000 \
  --eval-every-steps 250 \
  --log-every-steps 50 \
  --diag-every-steps 250 \
  --checkpoint-every-steps 0 \
  --full-checkpoint-every-steps 0 \
  --num-experts 4 \
  --top-k 1 \
  --router-z-loss-coef 0.0 \
  --router-aux-loss-coef 0.01 \
  --muon-lr 0.02 \
  --vector-lr 0.0003 \
  --momentum 0.95 \
  --matrix-weight-decay 0.1 \
  --hybrid-aux-weight-decay 0.01 \
  --aux-weight-decay 0.0 \
  --rho-hidden 50 \
  --rho-output 3000 \
  --tied-cap 3 \
  --tied-scale 0.5 \
  --chunk-rows 2048 \
  --max-grad-norm 1.0 \
  --poll-seconds 30 \
  > logs/moe_nanogpt_50k_top1_e4_routeraux0p01_3seeds_2p67b/launcher.log 2>&1 &
```

## Summarize

```bash
cd <repo>

<python> moe/summarize_moe_nanogpt.py \
  --run-root runs/moe_nanogpt_50k_top1_e4_routeraux0p01_3seeds_2p67b
```

## Diagnostics

The trainer logs:

- train/eval loss and perplexity
- tied-table RMS and max
- optimizer state dtypes
- MoE router entropy
- router max/min expert fraction
- router load coefficient of variation
- unused expert count
