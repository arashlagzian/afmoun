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

from optimizers import build_afmoun_v2, build_hybrid_muon


class TokenBlockDataset(Dataset):
    def __init__(self, path: Path, *, block_size: int, dtype: str = "uint32", max_blocks: int | None = None):
        np_dtype = np.uint32 if dtype == "uint32" else np.uint16
        self.tokens = np.memmap(path, dtype=np_dtype, mode="r")
        self.block_size = int(block_size)
        self.num_blocks = (len(self.tokens) - 1) // self.block_size
        if max_blocks is not None:
            self.num_blocks = min(self.num_blocks, int(max_blocks))

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int):
        start = int(idx) * self.block_size
        x = np.asarray(self.tokens[start : start + self.block_size], dtype=np.int64)
        y = np.asarray(self.tokens[start + 1 : start + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def resolve_data_path(meta_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [meta_dir / p]
    for parent in meta_dir.parents:
        candidates.append(parent / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_datasets(data_dir: Path, block_size: int, train_tokens: int | None, eval_tokens: int | None):
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    dtype = meta.get("dtype", "uint32")
    train_path = resolve_data_path(data_dir, meta["train_path"])
    eval_path = resolve_data_path(data_dir, meta["eval_path"])
    train_blocks = None if train_tokens is None else max(1, train_tokens // block_size)
    eval_blocks = None if eval_tokens is None else max(1, eval_tokens // block_size)
    return (
        TokenBlockDataset(train_path, block_size=block_size, dtype=dtype, max_blocks=train_blocks),
        TokenBlockDataset(eval_path, block_size=block_size, dtype=dtype, max_blocks=eval_blocks),
        meta,
        train_path,
        eval_path,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def set_optimizer_lr(opt, *, base_muon_lr: float, base_vector_lr: float) -> None:
    for group in opt.param_groups:
        role = group.get("role")
        if role in ("matrix", "tied"):
            group["lr"] = float(base_muon_lr)
        elif role in ("vector", "aux"):
            group["lr"] = float(base_vector_lr)


@torch.no_grad()
def evaluate(model, loader, device: str, use_bf16: bool, max_batches: int) -> tuple[float, float]:
    model.eval()
    losses = []
    for i, (x, _) in enumerate(loader):
        if i >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.startswith("cuda") and use_bf16)):
            out = model(input_ids=x, labels=x)
        losses.append(float(out.loss.detach().float().cpu()))
    model.train()
    loss = float(np.mean(losses)) if losses else float("nan")
    return loss, float(math.exp(loss)) if math.isfinite(loss) else float("nan")


def parse_args():
    p = argparse.ArgumentParser(description="NB98-compatible Llama paper worker using released V2 optimizers.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=["muon", "sign", "afmoun"], default="afmoun")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp32-params", action="store_true", default=True)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=3_814)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--train-tokens", type=int, default=2_000_000_000)
    p.add_argument("--eval-tokens", type=int, default=1_000_000)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--aux-weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--seed", type=int, default=43)
    return p.parse_args()


def arm_settings(arm: str) -> dict:
    if arm == "sign":
        return {"optimizer": "afmoun_v2", "tied_cap": 1.0, "tied_scale": 1.0}
    if arm == "afmoun":
        return {"optimizer": "afmoun_v2", "tied_cap": 3.0, "tied_scale": 0.5}
    return {"optimizer": "muon", "tied_cap": None, "tied_scale": None}


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    if args.device.startswith("cuda"):
        cuda_index = 0 if args.device == "cuda" else torch.device(args.device).index
        torch.cuda.set_device(cuda_index)

    micro_batch_size = int(args.micro_batch_size)
    accum_steps = int(args.gradient_accumulation_steps)
    tokens_per_step = micro_batch_size * int(args.block_size) * accum_steps
    args.max_steps = min(int(args.max_steps), max(1, int(args.train_tokens) // tokens_per_step))
    args.eval_batches = max(1, int(args.eval_tokens) // (micro_batch_size * int(args.block_size)))

    train_ds, eval_ds, meta, train_path, eval_path = load_datasets(
        Path(args.data_dir), int(args.block_size), int(args.train_tokens), int(args.eval_tokens)
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        generator=train_generator,
        drop_last=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=micro_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=args.device.startswith("cuda"),
    )

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(Path(args.model_dir), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.fp32_params:
        model = model.float()
    elif args.bf16 and args.device.startswith("cuda"):
        model = model.to(dtype=torch.bfloat16)
    model.to(args.device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    settings = arm_settings(args.arm)
    if args.arm == "muon":
        opt = build_hybrid_muon(
            model,
            muon_lr=args.muon_lr,
            aux_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            aux_weight_decay=args.aux_weight_decay,
        )
    else:
        opt = build_afmoun_v2(
            model,
            muon_lr=args.muon_lr,
            vector_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            tied_cap=float(settings["tied_cap"]),
            tied_scale=float(settings["tied_scale"]),
            rho_hidden=args.rho_hidden,
            rho_output=args.rho_output,
            chunk_rows=args.chunk_rows,
            ns_steps=5,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    config_payload = vars(args) | {
        "arm_settings": settings,
        "data_metadata": meta,
        "train_path_resolved": str(train_path),
        "eval_path_resolved": str(eval_path),
        "param_count": sum(p.numel() for p in model.parameters()),
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "master_params": "fp32" if args.fp32_params else ("bf16" if args.bf16 and args.device.startswith("cuda") else "fp32"),
        "autocast_only": "bf16" if args.fp32_params and args.bf16 and args.device.startswith("cuda") else None,
        "tokens_per_step": tokens_per_step,
        "labels": "input_ids",
        "init_from_config": True,
        "gradient_checkpointing": True,
        "nb98_log_semantics": "window_average_loss",
        "nb98_eval_semantics": "eval_only_at_eval_every_no_step1_eval",
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    def write(row: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)

    start = time.time()
    train_iter = iter(train_loader)
    running_loss = 0.0
    for step in range(1, int(args.max_steps) + 1):
        set_optimizer_lr(opt, base_muon_lr=args.muon_lr, base_vector_lr=args.vector_lr)
        opt.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            try:
                x, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, _ = next(train_iter)
            x = x.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(args.device.startswith("cuda") and args.bf16)):
                out = model(input_ids=x, labels=x)
                loss = out.loss / float(accum_steps)
            loss.backward()
            running_loss += float(loss.detach().float().cpu())

        if float(args.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        opt.step()

        tokens_seen = step * tokens_per_step
        seconds = time.time() - start
        if step == 1 or step % int(args.log_every_steps) == 0:
            denom = int(args.log_every_steps) if step > 1 else 1
            avg_loss = running_loss / float(denom)
            running_loss = 0.0
            write(
                {
                    "phase": "train",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "loss": avg_loss,
                    "ppl": math.exp(min(20.0, avg_loss)),
                    "seconds": seconds,
                }
            )
        if step % int(args.eval_every) == 0:
            ev_loss, ev_ppl = evaluate(model, eval_loader, args.device, bool(args.bf16), int(args.eval_batches))
            write(
                {
                    "phase": "eval",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "eval_loss": ev_loss,
                    "eval_ppl": ev_ppl,
                    "seconds": time.time() - start,
                }
            )
        if tokens_seen >= len(train_ds) * int(args.block_size):
            break


if __name__ == "__main__":
    main()
