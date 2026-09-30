from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizers import build_afmoun_v2, build_hybrid_muon, build_scion_sign_v2


class TokenBlockDataset(Dataset):
    def __init__(self, path: Path, *, block_size: int, dtype: str, max_blocks: int | None = None):
        np_dtype = np.uint32 if dtype == "uint32" else np.uint16
        self.tokens = np.memmap(path, dtype=np_dtype, mode="r")
        self.block_size = int(block_size)
        self.num_blocks = max(0, (len(self.tokens) - 1) // self.block_size)
        if max_blocks is not None:
            self.num_blocks = min(self.num_blocks, int(max_blocks))

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int):
        start = int(idx) * self.block_size
        arr = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(arr[:-1].copy())


def resolve_data_path(data_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [data_dir / p]
    for parent in data_dir.parents:
        candidates.append(parent / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def cosine_schedule(step: int, total: int, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    if total <= warmup:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup) / float(total - warmup)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(opt, *, muon_lr: float, vector_lr: float, adamw_lr: float, scale: float) -> None:
    for group in opt.param_groups:
        role = group.get("role")
        if role in ("matrix", "tied"):
            group["lr"] = muon_lr * scale
        elif role in ("vector", "aux"):
            group["lr"] = vector_lr * scale
        elif role == "adamw":
            group["lr"] = adamw_lr * scale


@torch.no_grad()
def evaluate(model, loader, *, device: str, use_bf16: bool, max_batches: int):
    model.eval()
    losses = []
    tokens = 0
    for i, x in enumerate(loader):
        if i >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.startswith("cuda") and use_bf16)):
            out = model(input_ids=x, labels=x)
        losses.append(float(out.loss.detach().float().cpu()))
        tokens += int(x.numel())
    model.train()
    loss = float(np.mean(losses)) if losses else float("nan")
    return loss, float(math.exp(min(20.0, loss))) if math.isfinite(loss) else float("nan"), tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ProtGPT2-BPE UniRef50 RoPE causal LM benchmark.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="runs/protein_uniref50")
    p.add_argument("--optimizer", choices=["adamw", "muon", "sign", "afmoun"], default="afmoun")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=43)

    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--train-tokens", type=int, default=500_000_000)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--max-steps", type=int, default=953)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--eval-every-steps", type=int, default=125)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")

    p.add_argument("--hidden-size", type=int, default=960)
    p.add_argument("--intermediate-size", type=int, default=2560)
    p.add_argument("--num-hidden-layers", type=int, default=32)
    p.add_argument("--num-attention-heads", type=int, default=15)
    p.add_argument("--num-key-value-heads", type=int, default=5)
    p.add_argument("--rms-norm-eps", type=float, default=1e-5)
    p.add_argument("--rope-theta", type=float, default=100000.0)

    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--adamw-lr", type=float, default=6e-4)
    p.add_argument("--adamw-beta1", type=float, default=0.9)
    p.add_argument("--adamw-beta2", type=float, default=0.95)
    p.add_argument("--adamw-eps", type=float, default=1e-8)
    p.add_argument("--adamw-weight-decay", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--aux-weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--tied-cap", type=float, default=3.0)
    p.add_argument("--tied-scale", type=float, default=0.5)
    p.add_argument("--sign-scale", type=float, default=1.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--chunk-rows", type=int, default=2048)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda"):
        cuda_index = 0 if args.device == "cuda" else torch.device(args.device).index
        torch.cuda.set_device(cuda_index)

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    train_path = resolve_data_path(data_dir, meta["train_path"])
    eval_path = resolve_data_path(data_dir, meta["eval_path"])
    dtype = meta.get("dtype", "uint32")
    vocab_size = int(meta["vocab_size"])
    eos_token_id = meta.get("eos_token_id", None)

    tokens_per_step = int(args.block_size) * int(args.micro_batch_size) * int(args.gradient_accumulation_steps)
    max_steps_by_tokens = max(1, int(args.train_tokens) // tokens_per_step)
    args.max_steps = min(int(args.max_steps), max_steps_by_tokens)
    eval_batches = max(1, int(args.eval_tokens) // (int(args.block_size) * int(args.micro_batch_size)))

    train_ds = TokenBlockDataset(train_path, block_size=args.block_size, dtype=dtype, max_blocks=max_steps_by_tokens * args.gradient_accumulation_steps * args.micro_batch_size)
    eval_ds = TokenBlockDataset(eval_path, block_size=args.block_size, dtype=dtype, max_blocks=eval_batches * args.micro_batch_size)
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True, generator=gen, num_workers=0, pin_memory=args.device.startswith("cuda"), drop_last=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.micro_batch_size, shuffle=False, num_workers=0, pin_memory=args.device.startswith("cuda"), drop_last=False)

    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.block_size,
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.rope_theta,
        tie_word_embeddings=True,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=None,
        use_cache=False,
    )
    model = LlamaForCausalLM(config)
    model.config.use_cache = False
    model.to(args.device)
    model.train()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    opt_kwargs = dict(
        muon_lr=args.muon_lr,
        vector_lr=args.vector_lr,
        momentum=args.momentum,
        matrix_weight_decay=args.matrix_weight_decay,
        rho_hidden=args.rho_hidden,
        rho_output=args.rho_output,
        chunk_rows=args.chunk_rows,
        ns_steps=5,
    )
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(
            [{"params": list(model.parameters()), "role": "adamw"}],
            lr=args.adamw_lr,
            betas=(args.adamw_beta1, args.adamw_beta2),
            eps=args.adamw_eps,
            weight_decay=args.adamw_weight_decay,
        )
    elif args.optimizer == "muon":
        opt = build_hybrid_muon(
            model,
            aux_lr=args.vector_lr,
            muon_lr=args.muon_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            aux_weight_decay=args.aux_weight_decay,
            ns_steps=5,
        )
    elif args.optimizer == "sign":
        opt = build_scion_sign_v2(model, tied_scale=args.sign_scale, **opt_kwargs)
    else:
        opt = build_afmoun_v2(model, tied_cap=args.tied_cap, tied_scale=args.tied_scale, **opt_kwargs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    config_payload = vars(args) | {
        "data_metadata": meta,
        "param_count": sum(p.numel() for p in model.parameters()),
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "master_params": "fp32",
        "autocast_only": "bf16" if args.bf16 and args.device.startswith("cuda") else None,
        "tokens_per_step": tokens_per_step,
        "eval_batches": eval_batches,
        "model_family": "llama_rope_protgpt2_bpe",
        "tie_word_embeddings": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    train_iter = iter(train_loader)
    start = time.time()
    for step in range(1, int(args.max_steps) + 1):
        lr_scale = cosine_schedule(step - 1, args.max_steps, args.warmup_steps) if args.lr_schedule == "cosine" else 1.0
        set_optimizer_lr(opt, muon_lr=args.muon_lr, vector_lr=args.vector_lr, adamw_lr=args.adamw_lr, scale=lr_scale)
        opt.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(int(args.gradient_accumulation_steps)):
            try:
                x = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x = next(train_iter)
            x = x.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(args.device.startswith("cuda") and args.bf16)):
                out = model(input_ids=x, labels=x)
                loss = out.loss / float(args.gradient_accumulation_steps)
            loss.backward()
            running_loss += float(loss.detach().float().cpu()) * float(args.gradient_accumulation_steps)

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        tokens_seen = step * tokens_per_step

        if step == 1 or step % int(args.log_every_steps) == 0:
            row = {"phase": "train", "step": step, "tokens_seen": tokens_seen, "loss": running_loss / float(args.gradient_accumulation_steps), "seconds": time.time() - start}
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(row, flush=True)
        if step == 1 or step % int(args.eval_every_steps) == 0:
            ev_loss, ev_ppl, ev_tokens = evaluate(model, eval_loader, device=args.device, use_bf16=args.bf16, max_batches=eval_batches)
            row = {"phase": "eval", "step": step, "tokens_seen": tokens_seen, "eval_loss": ev_loss, "eval_ppl": ev_ppl, "eval_tokens": ev_tokens, "seconds": time.time() - start}
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
