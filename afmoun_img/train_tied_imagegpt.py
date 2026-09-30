from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizers import build_afmoun_v2, build_hybrid_muon, build_scion_sign_v2
from optimizers.afmoun_v2 import _finite_cap_oracle_fp32, _row_rms_oracle_fp32


class ImageTokenDataset(Dataset):
    def __init__(self, path: Path, *, images: int, seq_len: int = 1024):
        self.path = Path(path)
        self.images = int(images)
        self.seq_len = int(seq_len)
        required = self.images * self.seq_len
        available = self.path.stat().st_size // 2
        if available < required:
            raise ValueError(f"{self.path} has {available:,} uint16 tokens, need {required:,}")
        self.tokens = torch.from_file(str(self.path), shared=False, size=required, dtype=torch.int16)

    def __len__(self) -> int:
        return self.images

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = int(idx) * self.seq_len
        return self.tokens[start : start + self.seq_len].to(torch.long)


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
    return (x * cos[None, None, :, :]) + (rotate_half(x) * sin[None, None, :, :])


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
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.relu(self.fc(x)).square() * math.sqrt(2.0))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_hidden: int, block_size: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, block_size)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TiedImageGPT(nn.Module):
    def __init__(
        self,
        *,
        color_vocab_size: int,
        sos_token: int,
        layers: int,
        heads: int,
        head_dim: int,
        mlp_hidden: int,
        block_size: int,
    ):
        super().__init__()
        self.color_vocab_size = int(color_vocab_size)
        self.sos_token = int(sos_token)
        self.vocab_size = self.color_vocab_size + 1
        self.dim = int(heads) * int(head_dim)
        self.block_size = int(block_size)
        self.wte = nn.Embedding(self.vocab_size, self.dim)
        self.blocks = nn.ModuleList([Block(self.dim, heads, mlp_hidden, block_size) for _ in range(layers)])
        self.norm = RMSNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.wte

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, image_tokens: torch.Tensor):
        b, t = image_tokens.shape
        if t != self.block_size:
            raise ValueError(f"expected block size {self.block_size}, got {t}")
        sos = torch.full((b, 1), self.sos_token, dtype=image_tokens.dtype, device=image_tokens.device)
        x_ids = torch.cat([sos, image_tokens[:, :-1]], dim=1)
        x = self.wte(x_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = F.linear(x, self.wte.weight[: self.color_vocab_size])
        loss = F.cross_entropy(logits.reshape(-1, self.color_vocab_size), image_tokens.reshape(-1))
        return {"logits": logits, "loss": loss}


def read_metadata(data_dir: Path) -> dict:
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))


def make_loader(path: Path, *, images: int, micro_batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    ds = ImageTokenDataset(path, images=images)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=int(micro_batch_size),
        shuffle=shuffle,
        generator=gen,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def autocast_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def optimizer_state_dtypes(opt) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                out.setdefault(k, set()).add(str(v.dtype))
    return {k: sorted(v) for k, v in out.items()}


@torch.no_grad()
def lmo_alignment_stats(model: TiedImageGPT, opt, *, cap: float, chunk_rows: int) -> dict:
    tied = model.get_input_embeddings().weight
    state = opt.state.get(tied, {})
    buf = state.get("momentum_buffer")
    if buf is None:
        buf = state.get("exp_avg")
    if buf is None or not torch.is_tensor(buf):
        return {}
    d = int(buf.shape[1])
    total = capped = capped_rows = active_rows = 0
    rms_energy = rms_tail_energy = 0.0
    dot_af_sign = norm_af = norm_sign = obj_af = obj_sign = 0.0
    row_norms = []
    row_norm_target = math.sqrt(float(d))
    for start in range(0, int(buf.shape[0]), int(chunk_rows)):
        raw = buf[start : start + int(chunk_rows)].detach().float()
        active = raw.abs().amax(dim=1) > 0
        if not bool(active.any().item()):
            continue
        raw = raw[active]
        active_rows += int(raw.shape[0])
        row_norms.append(raw.norm(dim=1).cpu())
        abs_raw = raw.abs()
        row_rms = raw.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-30)
        row_rms_endpoint = raw / row_rms
        if float(cap) == 1.0:
            af = raw.sign()
        elif math.isinf(float(cap)) or float(cap) >= row_norm_target:
            af = _row_rms_oracle_fp32(raw)
        else:
            af = _finite_cap_oracle_fp32(
                raw,
                cap=float(cap),
                bisection_steps=32,
                max_bracket_steps=128,
            )
        sign = torch.sign(raw)
        total += int(raw.numel())
        cap_value = float(cap)
        if math.isinf(cap_value) or cap_value >= row_norm_target:
            capped_mask = torch.zeros_like(abs_raw, dtype=torch.bool)
        else:
            capped_mask = af.abs() >= (cap_value - 1e-6)
        capped += int(capped_mask.sum().item())
        capped_rows += int(capped_mask.any(dim=1).sum().item())
        tail_mask = row_rms_endpoint.abs() > cap_value
        rms_sq = row_rms_endpoint.square()
        rms_energy += float(rms_sq.sum(dtype=torch.float64).item())
        rms_tail_energy += float(rms_sq[tail_mask].sum(dtype=torch.float64).item()) if bool(tail_mask.any().item()) else 0.0
        dot_af_sign += float((af * sign).sum(dtype=torch.float64).item())
        norm_af += float(af.square().sum(dtype=torch.float64).item())
        norm_sign += float(sign.square().sum(dtype=torch.float64).item())
        obj_af += float((af * raw).sum(dtype=torch.float64).item())
        obj_sign += float((sign * raw).sum(dtype=torch.float64).item())
    if active_rows == 0:
        return {}
    rn = torch.cat(row_norms).float()
    cos = dot_af_sign / max(1e-30, math.sqrt(norm_af) * math.sqrt(norm_sign))
    return {
        "tied_momentum_row_norm_mean": float(rn.mean().item()),
        "tied_momentum_row_norm_std_over_mean": float((rn.std(unbiased=False) / rn.mean().clamp_min(1e-30)).item()),
        "coords_capped_frac": capped / max(1, total),
        "rows_clipped_frac": capped_rows / max(1, active_rows),
        "raw_tail_energy_frac": rms_tail_energy / max(rms_energy, 1e-30),
        "cos_af_sign": cos,
        "objective_af_over_sign": obj_af / obj_sign if abs(obj_sign) > 1e-30 else None,
    }


@torch.no_grad()
def weight_stats(model: TiedImageGPT, opt, *, cap: float, chunk_rows: int) -> dict:
    tied = model.get_input_embeddings().weight.detach().float()
    color = tied[: model.color_vocab_size]
    sos = tied[model.sos_token]
    matrix_vals = []
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight is not model.get_input_embeddings().weight:
            matrix_vals.append(module.weight.detach().float().square().mean())
    matrix_rms = torch.stack(matrix_vals).mean().sqrt().item() if matrix_vals else float("nan")
    return {
        "tied_rms": tied.square().mean().sqrt().item(),
        "tied_max_abs": tied.abs().amax().item(),
        "color_tied_rms": color.square().mean().sqrt().item(),
        "color_tied_max_abs": color.abs().amax().item(),
        "sos_row_rms": sos.square().mean().sqrt().item(),
        "sos_row_max_abs": sos.abs().amax().item(),
        "matrix_rms_mean": matrix_rms,
        **lmo_alignment_stats(model, opt, cap=cap, chunk_rows=chunk_rows),
    }


@torch.no_grad()
def evaluate(model, loader, *, device, autocast: str, max_batches: int) -> dict:
    model.eval()
    losses = []
    tokens = 0
    enabled = autocast != "none" and device.type == "cuda"
    dtype = autocast_dtype(autocast)
    for i, batch in enumerate(loader):
        if i >= int(max_batches):
            break
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            loss = model(batch)["loss"]
        losses.append(float(loss.detach().float().item()))
        tokens += int(batch.numel())
    model.train()
    loss = float(sum(losses) / max(1, len(losses)))
    return {
        "eval_loss": loss,
        "eval_ppl": math.exp(min(20.0, loss)),
        "eval_bits_per_token": loss / math.log(2.0),
        "eval_bits_per_dim": loss / (3.0 * math.log(2.0)),
        "eval_tokens": tokens,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tied ImageGPT-style ImageNet-32 RGB554 experiment.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=["muon", "sign", "afmoun"], required=True)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--autocast", choices=["fp16", "bf16", "none"], default="bf16")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--mlp-hidden", type=int, default=2048)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=256)
    p.add_argument("--iterations", type=int, default=2500)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--diag-every-steps", type=int, default=250)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--full-checkpoint-every-steps", type=int, default=2500)
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


def diagnostic_cap(args: argparse.Namespace, settings: dict) -> float:
    cap = settings.get("tied_cap")
    if cap is None:
        cap = args.tied_cap
    return float(cap)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    data_dir = Path(args.data_dir)
    meta = read_metadata(data_dir)
    color_vocab = int(meta.get("color_vocab_size", 16_384))
    sos_token = int(meta.get("sos_token", color_vocab))
    seq_len = int(meta.get("sequence_length", args.block_size))
    if seq_len != int(args.block_size):
        raise ValueError(f"metadata sequence_length={seq_len} but block_size={args.block_size}")

    model = TiedImageGPT(
        color_vocab_size=color_vocab,
        sos_token=sos_token,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        mlp_hidden=args.mlp_hidden,
        block_size=args.block_size,
    ).float()
    param_count = sum(p.numel() for p in model.parameters())
    tied_ok = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    opt, settings = build_optimizer(args, model)
    if int(args.batch_size) % int(args.micro_batch_size) != 0:
        raise ValueError(
            f"batch_size={args.batch_size} must be divisible by "
            f"micro_batch_size={args.micro_batch_size}"
        )
    grad_accum = max(1, int(args.batch_size) // int(args.micro_batch_size))
    tokens_per_step = int(args.batch_size) * int(args.block_size)

    if args.check_config:
        print(json.dumps({
            "ok": True,
            "arm": args.arm,
            "param_count": param_count,
            "color_vocab_size": color_vocab,
            "sos_token": sos_token,
            "vocab_size": color_vocab + 1,
            "tied_embedding_head_identity": tied_ok,
            "tokens_per_step": tokens_per_step,
            "gradient_accumulation_steps": grad_accum,
            "master_params": "fp32",
            "autocast_only": args.autocast,
            "arm_settings": settings,
        }, indent=2))
        return

    model.to(device)
    model.train()
    train_path = data_dir / meta.get("train_path", "train_tokens_uint16.bin")
    eval_path = data_dir / meta.get("eval_path", "eval_tokens_uint16.bin")
    train_images = min(int(meta["train_images"]), int(args.iterations) * int(args.batch_size))
    eval_images = int(meta["eval_images"])
    train_loader = make_loader(train_path, images=train_images, micro_batch_size=args.micro_batch_size, seed=args.seed, shuffle=True)
    eval_loader = make_loader(eval_path, images=eval_images, micro_batch_size=args.micro_batch_size, seed=args.seed, shuffle=False)

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
        "color_vocab_size": color_vocab,
        "sos_token": sos_token,
        "vocab_size": color_vocab + 1,
        "tied_embedding_head_identity": tied_ok,
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    ev = evaluate(model, eval_loader, device=device, autocast=args.autocast, max_batches=args.eval_batches)
    write_jsonl(metrics, {"phase": "eval", "step": 0, "tokens_seen": 0, **ev})

    data_iter = iter(train_loader)
    enabled = args.autocast != "none" and device.type == "cuda"
    amp_dtype = autocast_dtype(args.autocast)
    running_loss = 0.0
    running_count = 0
    started = time.time()
    step_times = []

    for step in range(1, int(args.iterations) + 1):
        step_start = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
                out = model(batch)
                loss = out["loss"] / grad_accum
            loss.backward()
            running_loss += float(out["loss"].detach().float().item())
            running_count += 1

        grad_norm = None
        if float(args.max_grad_norm) > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm)).item())
        opt.step()
        step_times.append(time.time() - step_start)
        tokens_seen = step * tokens_per_step
        seconds = time.time() - started

        if step == 1 or step % int(args.log_every_steps) == 0:
            avg_loss = running_loss / max(1, running_count)
            write_jsonl(metrics, {
                "phase": "train",
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": avg_loss,
                "ppl": math.exp(min(20.0, avg_loss)),
                "bits_per_token": avg_loss / math.log(2.0),
                "bits_per_dim": avg_loss / (3.0 * math.log(2.0)),
                "muon_lr": args.muon_lr,
                "vector_lr": args.vector_lr,
                "grad_norm_preclip": grad_norm,
                "seconds": seconds,
                "step_seconds": step_times[-1],
                "optimizer_state_dtypes": optimizer_state_dtypes(opt) if step == 1 else None,
                "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}) if step == 1 else None,
            })
            running_loss = 0.0
            running_count = 0

        if step == 1 or step % int(args.diag_every_steps) == 0:
            write_jsonl(metrics, {
                "phase": "diagnostic",
                "step": step,
                "tokens_seen": tokens_seen,
                **weight_stats(model, opt, cap=diagnostic_cap(args, settings), chunk_rows=args.chunk_rows),
                "optimizer_state_dtypes": optimizer_state_dtypes(opt),
                "seconds": seconds,
            })

        if step % int(args.eval_every_steps) == 0:
            ev = evaluate(model, eval_loader, device=device, autocast=args.autocast, max_batches=args.eval_batches)
            write_jsonl(metrics, {"phase": "eval", "step": step, "tokens_seen": tokens_seen, **ev, "seconds": time.time() - started})

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
