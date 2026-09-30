from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizers import build_afmoun_v2, build_hybrid_muon, build_scion_sign_v2


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


class DenseMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, scaled_relu_sq: bool):
        super().__init__()
        self.fc = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim, bias=False)
        self.scaled_relu_sq = bool(scaled_relu_sq)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = F.relu(self.fc(x)).square()
        if self.scaled_relu_sq:
            h = h * math.sqrt(2.0)
        return self.proj(h), {}


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, scaled_relu_sq: bool):
        super().__init__()
        self.fc = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim, bias=False)
        self.scaled_relu_sq = bool(scaled_relu_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc(x)).square()
        if self.scaled_relu_sq:
            h = h * math.sqrt(2.0)
        return self.proj(h)


class TopKMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        num_experts: int,
        top_k: int,
        scaled_relu_sq: bool,
        router_z_loss_coef: float,
        router_aux_loss_coef: float,
    ):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_z_loss_coef = float(router_z_loss_coef)
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertMLP(dim, hidden_dim, scaled_relu_sq) for _ in range(num_experts)]
        )
        self.last_router_stats: dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        original_shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        logits = self.router(flat)
        probs = F.softmax(logits.float(), dim=-1).to(flat.dtype)
        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = top_idx == expert_id
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            expert_in = flat.index_select(0, token_idx)
            expert_out = expert(expert_in)
            weights = top_vals[token_idx, slot_idx].unsqueeze(-1).to(expert_out.dtype)
            out.index_add_(0, token_idx, expert_out * weights)

        with torch.no_grad():
            hard = F.one_hot(top_idx[:, 0], num_classes=self.num_experts).float()
            counts = hard.sum(dim=0)
            frac = counts / counts.sum().clamp_min(1.0)
            entropy = -(frac.clamp_min(1e-12) * frac.clamp_min(1e-12).log()).sum() / math.log(self.num_experts)
            self.last_router_stats = {
                "router_entropy": float(entropy.item()),
                "router_max_frac": float(frac.max().item()),
                "router_min_frac": float(frac.min().item()),
                "router_cv": float((frac.std(unbiased=False) / frac.mean().clamp_min(1e-12)).item()),
                "router_unused_experts": int((counts == 0).sum().item()),
            }

        aux: dict[str, torch.Tensor] = {}
        if self.router_aux_loss_coef > 0:
            importance = probs.float().mean(dim=0)
            load = F.one_hot(top_idx[:, 0], num_classes=self.num_experts).float().mean(dim=0)
            aux["router_aux_loss"] = self.router_aux_loss_coef * self.num_experts * torch.sum(importance * load)
        if self.router_z_loss_coef > 0:
            aux["router_z_loss"] = self.router_z_loss_coef * torch.logsumexp(logits.float(), dim=-1).square().mean()
        return out.reshape(original_shape), aux


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_hidden: int,
        block_size: int,
        scaled_relu_sq: bool,
        *,
        use_moe: bool,
        num_experts: int,
        top_k: int,
        router_z_loss_coef: float,
        router_aux_loss_coef: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, block_size)
        self.norm2 = RMSNorm(dim)
        if use_moe:
            self.mlp = TopKMoE(
                dim,
                mlp_hidden,
                num_experts=num_experts,
                top_k=top_k,
                scaled_relu_sq=scaled_relu_sq,
                router_z_loss_coef=router_z_loss_coef,
                router_aux_loss_coef=router_aux_loss_coef,
            )
        else:
            self.mlp = DenseMLP(dim, mlp_hidden, scaled_relu_sq)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = x + self.attn(self.norm1(x))
        y, aux = self.mlp(self.norm2(x))
        x = x + y
        return x, aux


class MoENanoGPT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        layers: int,
        heads: int,
        head_dim: int,
        mlp_hidden: int,
        block_size: int,
        scaled_relu_sq: bool,
        use_moe: bool,
        num_experts: int,
        top_k: int,
        router_z_loss_coef: float,
        router_aux_loss_coef: float,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(heads) * int(head_dim)
        self.block_size = int(block_size)
        self.use_moe = bool(use_moe)
        self.wte = nn.Embedding(vocab_size, self.dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    self.dim,
                    heads,
                    mlp_hidden,
                    block_size,
                    scaled_relu_sq,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                    router_z_loss_coef=router_z_loss_coef,
                    router_aux_loss_coef=router_aux_loss_coef,
                )
                for _ in range(layers)
            ]
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

    def router_stats(self) -> dict[str, float]:
        stats = []
        for block in self.blocks:
            mlp = block.mlp
            if isinstance(mlp, TopKMoE) and mlp.last_router_stats:
                stats.append(mlp.last_router_stats)
        if not stats:
            return {}
        keys = stats[0].keys()
        return {f"moe_{k}_mean": float(sum(float(s[k]) for s in stats) / len(stats)) for k in keys}

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.wte(input_ids)
        aux_losses: list[torch.Tensor] = []
        for block in self.blocks:
            x, aux = block(x)
            aux_losses.extend(aux.values())
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        lm_loss = None
        aux_loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            aux_loss = sum(aux_losses) if aux_losses else logits.new_tensor(0.0)
            loss = lm_loss + aux_loss
        return {"logits": logits, "loss": loss, "lm_loss": lm_loss, "aux_loss": aux_loss}


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
def weight_stats(model: MoENanoGPT) -> dict:
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


def autocast_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


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
            loss = model(x, y)["lm_loss"]
        losses.append(float(loss.detach().float().item()))
        tokens += int(y.numel())
    model.train()
    loss = float(sum(losses) / max(1, len(losses)))
    return loss, math.exp(min(20.0, loss)), tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tied-embedding NanoGPT-MoE AF-Muon experiment.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=["muon", "sign", "afmoun"], required=True)
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
    p.add_argument("--moe", action="store_true")
    p.add_argument("--num-experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--router-z-loss-coef", type=float, default=0.0)
    p.add_argument("--router-aux-loss-coef", type=float, default=0.01)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--aux-weight-decay", type=float, default=0.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--tied-cap", type=float, default=3.0)
    p.add_argument("--tied-scale", type=float, default=0.5)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--scaled-relu-sq", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--check-config", action="store_true")
    return p.parse_args()


def build_optimizer(args: argparse.Namespace, model: nn.Module):
    if args.arm == "muon":
        return build_hybrid_muon(
            model,
            muon_lr=args.muon_lr,
            aux_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            aux_weight_decay=args.aux_weight_decay,
        ), {"optimizer": "hybrid_muon", "tied_cap": None, "tied_scale": None}
    if args.arm == "sign":
        return build_scion_sign_v2(
            model,
            muon_lr=args.muon_lr,
            vector_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            rho_hidden=args.rho_hidden,
            rho_output=args.rho_output,
            chunk_rows=args.chunk_rows,
        ), {"optimizer": "scion_sign_v2", "tied_cap": 1.0, "tied_scale": 1.0}
    return build_afmoun_v2(
        model,
        muon_lr=args.muon_lr,
        vector_lr=args.vector_lr,
        momentum=args.momentum,
        matrix_weight_decay=args.matrix_weight_decay,
        tied_cap=args.tied_cap,
        tied_scale=args.tied_scale,
        rho_hidden=args.rho_hidden,
        rho_output=args.rho_output,
        chunk_rows=args.chunk_rows,
    ), {"optimizer": "afmoun_v2", "tied_cap": args.tied_cap, "tied_scale": args.tied_scale}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = MoENanoGPT(
        vocab_size=args.vocab_size,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        mlp_hidden=args.mlp_hidden,
        block_size=args.block_size,
        scaled_relu_sq=args.scaled_relu_sq,
        use_moe=args.moe,
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_z_loss_coef=args.router_z_loss_coef,
        router_aux_loss_coef=args.router_aux_loss_coef,
    ).float()
    param_count = sum(p.numel() for p in model.parameters())
    tied_ok = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    opt, settings = build_optimizer(args, model)

    if args.check_config:
        print(json.dumps({
            "ok": True,
            "arm": args.arm,
            "moe": bool(args.moe),
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "param_count": param_count,
            "tied_embedding_head_identity": tied_ok,
            "tokens_per_step": int(args.batch_size) * int(args.block_size),
            "gradient_accumulation_steps": max(1, int(args.batch_size) // int(args.micro_batch_size)),
            "master_params": "fp32",
            "autocast_only": args.autocast,
            "optimizer_state_dtypes_initial": optimizer_state_dtypes(opt),
            "arm_settings": settings,
        }, indent=2))
        return

    model.to(device)
    model.train()
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
        "param_count": param_count,
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "tokens_per_step": tokens_per_step,
        "gradient_accumulation_steps": grad_accum,
        "master_params": "fp32",
        "autocast_only": args.autocast,
        "tied_embedding_head_identity": tied_ok,
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    ev_loss, ev_ppl, ev_tokens = evaluate(model, eval_loader, device=device, autocast=args.autocast, max_batches=len(eval_loader))
    write_jsonl(metrics, {"phase": "eval", "step": 0, "tokens_seen": 0, "eval_loss": ev_loss, "eval_ppl": ev_ppl, "eval_tokens": ev_tokens})

    data_iter = iter(train_loader)
    enabled = args.autocast != "none" and device.type == "cuda"
    amp_dtype = autocast_dtype(args.autocast)
    running_loss = 0.0
    running_lm_loss = 0.0
    running_aux_loss = 0.0
    running_count = 0
    started = time.time()

    for step in range(1, int(args.iterations) + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = batch.to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
                out = model(x, y)
                loss = out["loss"] / grad_accum
            loss.backward()
            running_loss += float(out["loss"].detach().float().item())
            running_lm_loss += float(out["lm_loss"].detach().float().item())
            running_aux_loss += float(out["aux_loss"].detach().float().item())
            running_count += 1

        grad_norm = None
        if float(args.max_grad_norm) > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm)).item())
        opt.step()
        tokens_seen = step * tokens_per_step
        seconds = time.time() - started

        if step == 1 or step % int(args.log_every_steps) == 0:
            avg_loss = running_loss / max(1, running_count)
            avg_lm = running_lm_loss / max(1, running_count)
            avg_aux = running_aux_loss / max(1, running_count)
            row = {
                "phase": "train",
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": avg_loss,
                "lm_loss": avg_lm,
                "aux_loss": avg_aux,
                "ppl": math.exp(min(20.0, avg_lm)),
                "muon_lr": args.muon_lr,
                "vector_lr": args.vector_lr,
                "grad_norm_preclip": grad_norm,
                "seconds": seconds,
                **model.router_stats(),
            }
            if step == 1:
                row["optimizer_state_dtypes"] = optimizer_state_dtypes(opt)
                row["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
            write_jsonl(metrics, row)
            running_loss = running_lm_loss = running_aux_loss = 0.0
            running_count = 0

        if step == 1 or step % int(args.diag_every_steps) == 0:
            write_jsonl(metrics, {
                "phase": "diagnostic",
                "step": step,
                "tokens_seen": tokens_seen,
                **weight_stats(model),
                **model.router_stats(),
                "optimizer_state_dtypes": optimizer_state_dtypes(opt),
                "seconds": seconds,
            })

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


if __name__ == "__main__":
    main()
