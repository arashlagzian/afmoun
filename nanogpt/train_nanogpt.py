from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanogpt.optimizers import (
    build_afmoun_v2,
    build_hybrid_muon,
    build_nanogpt_factorial_muon,
    build_scion_sign_v2,
)


class TokenBlockDataset(Dataset):
    def __init__(self, path: Path, *, block_size: int, num_blocks: int, dtype: str):
        self.path = Path(path)
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.dtype = np.dtype(dtype)
        self.tokens = np.memmap(self.path, dtype=self.dtype, mode="r")
        if self.tokens.shape[0] < self.num_blocks * self.block_size + 1:
            raise ValueError(f"{path} does not contain enough tokens for {num_blocks} blocks")

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = int(idx) * self.block_size
        arr = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(arr)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, block_size: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        pos = torch.arange(block_size).float()
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("rope_cos", emb.cos(), persistent=False)
        self.register_buffer("rope_sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:t].to(q.dtype), self.rope_sin[:t].to(q.dtype))
        k = apply_rope(k, self.rope_cos[:t].to(k.dtype), self.rope_sin[:t].to(k.dtype))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, scaled_relu_sq: bool):
        super().__init__()
        self.fc = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim, bias=False)
        self.scaled_relu_sq = bool(scaled_relu_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(x)).square()
        if self.scaled_relu_sq:
            x = x * math.sqrt(2.0)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_hidden: int, block_size: int, scaled_relu_sq: bool):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, block_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_hidden, scaled_relu_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, *, vocab_size: int, layers: int, heads: int, head_dim: int, mlp_hidden: int, block_size: int, scaled_relu_sq: bool):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(heads) * int(head_dim)
        self.block_size = int(block_size)
        self.wte = nn.Embedding(vocab_size, self.dim)
        self.blocks = nn.ModuleList(
            [Block(self.dim, heads, mlp_hidden, block_size, scaled_relu_sq) for _ in range(layers)]
        )
        self.norm = RMSNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.wte

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.wte(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}


def find_data_file(data_dir: Path, names: Iterable[str]) -> Path:
    for name in names:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {list(names)} found under {data_dir}")


def load_metadata(data_dir: Path) -> dict:
    for name in ("metadata.json", "meta.json", "data_metadata.json"):
        p = data_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def make_loader(path: Path, *, block_size: int, blocks: int, dtype: str, micro_batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    ds = TokenBlockDataset(path, block_size=block_size, num_blocks=blocks, dtype=dtype)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(ds, batch_size=micro_batch_size, shuffle=shuffle, generator=gen, num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=True)


def optimizer_state_dtypes(opt) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                out.setdefault(k, set()).add(str(v.dtype))
    return {k: sorted(v) for k, v in out.items()}


@torch.no_grad()
def weight_stats(model: NanoGPT) -> dict:
    tied = model.get_input_embeddings().weight.detach().float()
    matrix_vals = []
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight is not model.get_input_embeddings().weight:
            matrix_vals.append(module.weight.detach().float().square().mean())
    matrix_rms = torch.stack(matrix_vals).mean().sqrt().item() if matrix_vals else float("nan")
    return {
        "tied_rms": tied.square().mean().sqrt().item(),
        "tied_max_abs": tied.abs().amax().item(),
        "matrix_rms_mean": matrix_rms,
    }


def schedule_mult(step: int, max_steps: int, warmdown_frac: float) -> float:
    warmdown_steps = int(round(max_steps * warmdown_frac))
    decay_start = max_steps - warmdown_steps
    if warmdown_steps <= 0 or step < decay_start:
        return 1.0
    return max(0.0, (max_steps - step) / max(1, warmdown_steps))


def set_lrs(opt, base_lrs: dict[str, float], mult: float) -> None:
    for group in opt.param_groups:
        role = group.get("role")
        if role in {"vector", "vector_adamw", "vector_rms"}:
            group["lr"] = base_lrs["vector"] * mult
        elif role in {"aux", "tied_adamw"}:
            group["lr"] = base_lrs["aux"] * mult
        else:
            group["lr"] = base_lrs["matrix"] * mult


def autocast_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


@torch.no_grad()
def evaluate(model, loader, *, device, autocast: str, max_batches: int) -> tuple[float, float, int]:
    model.eval()
    losses = []
    tokens = 0
    enabled = autocast != "none" and device.type == "cuda"
    dtype = autocast_dtype(autocast)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            loss = model(x, y)["loss"]
        losses.append(float(loss.detach().float().item()))
        tokens += int(y.numel())
    model.train()
    loss = float(sum(losses) / max(1, len(losses)))
    return loss, math.exp(min(20.0, loss)), tokens


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NanoGPT tied-table AF-Muon experiment.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--arm",
        choices=[
            "muon",
            "sign",
            "afmoun",
            "fact_adamw_adamw",
            "fact_c3_adamw",
            "fact_adamw_rms",
            "fact_c3_rms",
        ],
        required=True,
    )
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--autocast", choices=["fp16", "bf16", "none"], default="bf16")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--mlp-hidden", type=int, default=3072)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=5100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--diag-every-steps", type=int, default=250)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--full-checkpoint-every-steps", type=int, default=0)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-lr-multiplier", type=float, default=1.0)
    p.add_argument("--multiplier-mode", choices=["all", "muon_only"], default="muon_only")
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--aux-weight-decay", type=float, default=0.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--tied-cap", type=float, default=3.0)
    p.add_argument("--tied-scale", type=float, default=0.5)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--warmdown-frac", type=float, default=0.0)
    p.add_argument("--activation-mode", choices=["all_scaled", "scion_table", "all_relu"], default="all_scaled")
    p.add_argument("--scaled-relu-sq", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--check-config", action="store_true", help="Build the model/optimizer, print config, and exit before data loading.")
    p.add_argument("--benchmark-memory", action="store_true", help="Reset CUDA peak stats before training, report exact peak memory, and skip final checkpoint.")
    return p.parse_args()


FACTORIAL_ARMS = {
    "fact_adamw_adamw": {"tied_rule": "adamw", "vector_rule": "adamw", "label": "AdamW tied / AdamW 1D"},
    "fact_c3_adamw": {"tied_rule": "afmoun", "vector_rule": "adamw", "label": "AF-Muon tied / AdamW 1D"},
    "fact_adamw_rms": {"tied_rule": "adamw", "vector_rule": "rms", "label": "AdamW tied / RMS-LMO 1D"},
    "fact_c3_rms": {"tied_rule": "afmoun", "vector_rule": "rms", "label": "AF-Muon tied / RMS-LMO 1D"},
}


def is_factorial_arm(arm: str) -> bool:
    return arm in FACTORIAL_ARMS


def effective_scaled_relu_sq(args: argparse.Namespace) -> bool:
    if args.activation_mode == "all_scaled":
        return True
    if args.activation_mode == "all_relu":
        return False
    return args.arm != "muon"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.check_config:
        scaled_relu_sq = effective_scaled_relu_sq(args)
        model = NanoGPT(
            vocab_size=args.vocab_size,
            layers=args.layers,
            heads=args.heads,
            head_dim=args.head_dim,
            mlp_hidden=args.mlp_hidden,
            block_size=args.block_size,
            scaled_relu_sq=scaled_relu_sq,
        ).float()
        param_count = sum(p.numel() for p in model.parameters())
        tied_ok = model.get_input_embeddings().weight is model.get_output_embeddings().weight
        print(json.dumps({
            "ok": True,
            "arm": args.arm,
            "master_params": "fp32",
            "autocast_only": args.autocast,
            "grad_scaler_enabled": bool(args.autocast == "fp16" and torch.cuda.is_available()),
            "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "tied_embedding_head_identity": tied_ok,
            "param_count": param_count,
            "tokens_per_step": int(args.batch_size) * int(args.block_size),
            "gradient_accumulation_steps": max(1, int(args.batch_size) // int(args.micro_batch_size)),
            "multiplier_mode": args.multiplier_mode,
            "activation_mode": args.activation_mode,
            "scaled_relu_sq_effective": scaled_relu_sq,
            "matrix_lr_after_multiplier": float(args.muon_lr)
            * (
                float(args.muon_lr_multiplier)
                if args.multiplier_mode == "all" or args.arm == "muon"
                else 1.0
            ),
            "vector_lr": float(args.vector_lr),
            "matrix_weight_decay": float(args.matrix_weight_decay),
            "aux_weight_decay": float(args.aux_weight_decay),
            "rho_hidden": float(args.rho_hidden),
            "rho_output": float(args.rho_output),
            "factorial": FACTORIAL_ARMS.get(args.arm),
            "tied_cap": None if args.arm in {"muon", "fact_adamw_adamw", "fact_adamw_rms"} else (1.0 if args.arm == "sign" else float(args.tied_cap)),
            "tied_scale": None if args.arm in {"muon", "fact_adamw_adamw", "fact_adamw_rms"} else (1.0 if args.arm == "sign" else float(args.tied_scale)),
        }, indent=2))
        return

    data_dir = Path(args.data_dir)
    meta = load_metadata(data_dir)
    token_dtype = meta.get("dtype", "uint16")
    train_path = find_data_file(data_dir, ["train_tokens_uint16.bin", "train_tokens_uint32.bin", "train.bin"])
    eval_path = find_data_file(data_dir, ["eval_tokens_uint16.bin", "eval_tokens_uint32.bin", "val_tokens_uint16.bin", "val.bin"])
    if "uint32" in train_path.name:
        token_dtype = "uint32"

    grad_accum = max(1, int(args.batch_size) // int(args.micro_batch_size))
    tokens_per_step = int(args.batch_size) * int(args.block_size)
    train_blocks = min(int(meta.get("train_blocks", 10**18)), int(args.iterations) * int(args.batch_size))
    if train_blocks == 10**18:
        train_blocks = int(args.iterations) * int(args.batch_size)
    eval_blocks = max(1, math.ceil(int(args.eval_tokens) / int(args.block_size)))
    if "eval_blocks" in meta:
        eval_blocks = min(eval_blocks, int(meta["eval_blocks"]))

    train_loader = make_loader(train_path, block_size=args.block_size, blocks=train_blocks, dtype=token_dtype, micro_batch_size=args.micro_batch_size, seed=args.seed, shuffle=True)
    eval_loader = make_loader(eval_path, block_size=args.block_size, blocks=eval_blocks, dtype=token_dtype, micro_batch_size=args.micro_batch_size, seed=args.seed, shuffle=False)

    scaled_relu_sq = effective_scaled_relu_sq(args)
    model = NanoGPT(
        vocab_size=args.vocab_size,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        mlp_hidden=args.mlp_hidden,
        block_size=args.block_size,
        scaled_relu_sq=scaled_relu_sq,
    ).float().to(device)
    model.train()
    lr_multiplier = float(args.muon_lr_multiplier) if args.multiplier_mode == "all" or args.arm == "muon" else 1.0
    base_matrix_lr = float(args.muon_lr) * lr_multiplier
    if args.arm == "muon":
        opt = build_hybrid_muon(
            model,
            muon_lr=base_matrix_lr,
            aux_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            aux_weight_decay=args.aux_weight_decay,
        )
        settings = {"optimizer": "hybrid_muon", "tied_cap": None, "tied_scale": None}
    elif args.arm == "sign":
        opt = build_scion_sign_v2(
            model,
            muon_lr=base_matrix_lr,
            vector_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            rho_hidden=args.rho_hidden,
            rho_output=args.rho_output,
            chunk_rows=args.chunk_rows,
        )
        settings = {"optimizer": "scion_sign_v2", "tied_cap": 1.0, "tied_scale": 1.0}
    elif args.arm == "afmoun":
        opt = build_afmoun_v2(
            model,
            muon_lr=base_matrix_lr,
            vector_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            tied_cap=args.tied_cap,
            tied_scale=args.tied_scale,
            rho_hidden=args.rho_hidden,
            rho_output=args.rho_output,
            chunk_rows=args.chunk_rows,
        )
        settings = {"optimizer": "afmoun_v2", "tied_cap": args.tied_cap, "tied_scale": args.tied_scale}
    elif is_factorial_arm(args.arm):
        rules = FACTORIAL_ARMS[args.arm]
        opt = build_nanogpt_factorial_muon(
            model,
            tied_rule=rules["tied_rule"],
            vector_rule=rules["vector_rule"],
            muon_lr=base_matrix_lr,
            vector_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            tied_weight_decay=args.aux_weight_decay if rules["tied_rule"] == "adamw" else 0.0,
            vector_weight_decay=args.aux_weight_decay if rules["vector_rule"] == "adamw" else 0.0,
            tied_cap=args.tied_cap,
            tied_scale=args.tied_scale,
            rho_hidden=args.rho_hidden,
            rho_output=args.rho_output,
            chunk_rows=args.chunk_rows,
        )
        settings = {
            "optimizer": "nanogpt_factorial_muon",
            "factorial_label": rules["label"],
            "tied_rule": rules["tied_rule"],
            "vector_rule": rules["vector_rule"],
            "tied_cap": args.tied_cap if rules["tied_rule"] == "afmoun" else None,
            "tied_scale": args.tied_scale if rules["tied_rule"] == "afmoun" else None,
        }
    else:
        raise ValueError(f"unsupported arm: {args.arm}")

    out_dir = Path(args.output_dir)
    ck_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / "metrics.jsonl"
    if metrics.exists():
        metrics.unlink()
    config_payload = vars(args) | {
        "arm_settings": settings,
        "data_metadata": meta,
        "train_path_resolved": str(train_path),
        "eval_path_resolved": str(eval_path),
        "param_count": sum(p.numel() for p in model.parameters()),
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "tokens_per_step": tokens_per_step,
        "gradient_accumulation_steps": grad_accum,
        "master_params": "fp32",
        "autocast_only": args.autocast,
        "grad_scaler_enabled": bool(args.autocast == "fp16" and device.type == "cuda"),
        "activation_mode": args.activation_mode,
        "scaled_relu_sq_effective": scaled_relu_sq,
        "nanogpt_recipe_note": "Matched paper protocol: 12 layers, head_dim 128, vocab 50304, batch 512, block 1024, iterations 5100, no warmdown, gradient clipping at norm 1.0.",
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    ev_loss, ev_ppl, ev_tokens = evaluate(model, eval_loader, device=device, autocast=args.autocast, max_batches=len(eval_loader))
    write_jsonl(metrics, {"phase": "eval", "step": 0, "tokens_seen": 0, "eval_loss": ev_loss, "eval_ppl": ev_ppl, "eval_tokens": ev_tokens})

    running_loss = 0.0
    running_count = 0
    step = 0
    micro_step = 0
    tokens_seen = 0
    started = time.time()
    enabled = args.autocast != "none" and device.type == "cuda"
    amp_dtype = autocast_dtype(args.autocast)
    scaler = make_grad_scaler(args.autocast == "fp16" and device.type == "cuda")
    data_iter = iter(train_loader)
    base_lrs = {"matrix": base_matrix_lr, "vector": float(args.vector_lr), "aux": float(args.vector_lr)}

    if args.benchmark_memory and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    while step < int(args.iterations):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            micro_step += 1
            batch = batch.to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
                loss = model(x, y)["loss"] / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.detach().float().item()) * grad_accum
            running_count += 1

        step += 1
        lr_mult = schedule_mult(step, int(args.iterations), float(args.warmdown_frac))
        set_lrs(opt, base_lrs, lr_mult)
        grad_norm = None
        if scaler is not None:
            scaler.unscale_(opt)
        if float(args.max_grad_norm) > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm)).item())
        else:
            grads = [p.grad.detach().float().square().sum() for p in model.parameters() if p.grad is not None]
            grad_norm = float(torch.stack(grads).sum().sqrt().item()) if grads else None
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        tokens_seen = step * tokens_per_step
        seconds = time.time() - started

        if step == 1 or step % int(args.log_every_steps) == 0:
            avg_loss = running_loss / max(1, running_count)
            row = {
                "phase": "train",
                "step": step,
                "micro_step": micro_step,
                "tokens_seen": tokens_seen,
                "loss": avg_loss,
                "ppl": math.exp(min(20.0, avg_loss)),
                "lr_mult": lr_mult,
                "matrix_lr": opt.param_groups[0]["lr"],
                "vector_lr": next((g["lr"] for g in opt.param_groups if g.get("role") in {"vector", "aux", "vector_adamw", "vector_rms"}), None),
                "grad_norm_preclip": grad_norm,
                "seconds": seconds,
            }
            if step == 1:
                row["optimizer_state_dtypes"] = optimizer_state_dtypes(opt)
                row["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
                row["grad_scaler_enabled"] = scaler is not None
            write_jsonl(metrics, row)
            running_loss = 0.0
            running_count = 0

        if step == 1 or step % int(args.diag_every_steps) == 0:
            write_jsonl(metrics, {"phase": "diagnostic", "step": step, "tokens_seen": tokens_seen, **weight_stats(model), "optimizer_state_dtypes": optimizer_state_dtypes(opt), "seconds": seconds})

        if step % int(args.eval_every_steps) == 0:
            ev_loss, ev_ppl, ev_tokens = evaluate(model, eval_loader, device=device, autocast=args.autocast, max_batches=len(eval_loader))
            write_jsonl(metrics, {"phase": "eval", "step": step, "tokens_seen": tokens_seen, "eval_loss": ev_loss, "eval_ppl": ev_ppl, "eval_tokens": ev_tokens, "seconds": time.time() - started})

        if int(args.checkpoint_every_steps) > 0 and step % int(args.checkpoint_every_steps) == 0:
            path = ck_dir / f"model_step{step:06d}_tokens{tokens_seen}.pt"
            torch.save({"model_state_dict": model.state_dict(), "step": step, "tokens_seen": tokens_seen, "config": config_payload}, path)
            write_jsonl(metrics, {"phase": "checkpoint", "kind": "model", "step": step, "tokens_seen": tokens_seen, "path": str(path), "seconds": time.time() - started})

        if int(args.full_checkpoint_every_steps) > 0 and step % int(args.full_checkpoint_every_steps) == 0:
            path = ck_dir / "latest_full.pt"
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "step": step, "tokens_seen": tokens_seen, "config": config_payload}, path)
            write_jsonl(metrics, {"phase": "checkpoint", "kind": "full_latest", "step": step, "tokens_seen": tokens_seen, "path": str(path), "seconds": time.time() - started})

    if args.benchmark_memory and device.type == "cuda":
        write_jsonl(metrics, {
            "phase": "memory_benchmark",
            "step": step,
            "tokens_seen": tokens_seen,
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
            "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
            "seconds": time.time() - started,
        })

    if not args.benchmark_memory:
        path = ck_dir / "latest_full.pt"
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "step": step, "tokens_seen": tokens_seen, "config": config_payload}, path)
        write_jsonl(metrics, {"phase": "checkpoint", "kind": "full_latest", "step": step, "tokens_seen": tokens_seen, "path": str(path), "seconds": time.time() - started})


if __name__ == "__main__":
    main()
