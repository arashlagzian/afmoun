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

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from optimizers import build_afmoun_v2, build_hybrid_muon


class TokenMemmapDataset(Dataset):
    def __init__(self, path: Path, *, block_size: int, num_blocks: int, dtype: str) -> None:
        np_dtype = np.uint32 if dtype == "uint32" else np.uint16
        self.tokens = np.memmap(path, dtype=np_dtype, mode="r")
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int):
        start = int(idx) * self.block_size
        end = start + self.block_size
        arr = np.asarray(self.tokens[start : end + 1], dtype=np.int64)
        return torch.from_numpy(arr[:-1].copy())


def resolve_path(cache_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [cache_dir / p]
    for parent in cache_dir.parents:
        candidates.append(parent / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def build_loader(dataset: Dataset, *, micro_batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(micro_batch_size),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model, loader, *, device: torch.device, use_bf16: bool, max_eval_blocks: int) -> tuple[float, float, int]:
    model.eval()
    loss_sum = 0.0
    target_count = 0
    tokens_seen = 0
    blocks_seen = 0
    autocast_enabled = bool(use_bf16 and device.type == "cuda")
    for input_ids in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            outputs = model(input_ids=input_ids, labels=input_ids)
        batch_targets = int(input_ids.shape[0]) * max(1, int(input_ids.shape[1]) - 1)
        loss_sum += float(outputs.loss.detach().cpu()) * batch_targets
        target_count += batch_targets
        tokens_seen += int(input_ids.numel())
        blocks_seen += int(input_ids.shape[0])
        if blocks_seen >= int(max_eval_blocks):
            break
    model.train()
    loss = loss_sum / target_count if target_count > 0 else float("nan")
    ppl = math.exp(min(20.0, loss)) if math.isfinite(loss) else float("nan")
    return loss, ppl, tokens_seen


def set_lr(opt, *, muon_lr: float, vector_lr: float) -> None:
    for group in opt.param_groups:
        role = group.get("role")
        if role in {"matrix", "tied"}:
            group["lr"] = float(muon_lr)
        elif role in {"vector", "aux"}:
            group["lr"] = float(vector_lr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper Qwen2.5-0.5B NB90-compatible multiseed runner.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="runs/01_paper_qwen25_nb90_v2")
    p.add_argument("--arm", choices=["muon", "sign", "afmoun"], default="afmoun")
    p.add_argument("--train-tokens", type=int, default=1_000_000_000)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--max-steps", type=int, default=1_907)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--micro-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=64)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=0.0003)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--aux-weight-decay", type=float, default=0.01)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--diag-every-steps", type=int, default=250)
    p.add_argument("--disable-diagnostics", action="store_true")
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--full-checkpoint-every-steps", type=int, default=0)
    p.add_argument("--resume-from", default="")
    p.add_argument("--fp32-params", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--benchmark-memory", action="store_true", help="Reset CUDA peak stats before training, report exact peak memory, and skip final checkpoint.")
    return p.parse_args()


def arm_settings(arm: str) -> dict:
    if arm == "sign":
        return {"optimizer": "afmoun_v2", "tied_cap": 1.0, "tied_scale": 1.0}
    if arm == "afmoun":
        return {"optimizer": "afmoun_v2", "tied_cap": 3.0, "tied_scale": 0.5}
    return {"optimizer": "muon", "tied_cap": None, "tied_scale": None}


def main() -> None:
    args = parse_args()
    settings = arm_settings(args.arm)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(args.data_dir)
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    train_path = resolve_path(cache_dir, metadata["train_path"])
    eval_path = resolve_path(cache_dir, metadata["eval_path"])
    train_blocks = min(int(args.train_tokens) // int(args.block_size), int(metadata["train_blocks"]))
    eval_blocks = min(max(1, int(args.eval_tokens) // int(args.block_size)), int(metadata["eval_blocks"]))
    token_dtype = metadata.get("dtype", "uint32")

    train_ds = TokenMemmapDataset(train_path, block_size=args.block_size, num_blocks=train_blocks, dtype=token_dtype)
    eval_ds = TokenMemmapDataset(eval_path, block_size=args.block_size, num_blocks=eval_blocks, dtype=token_dtype)
    train_loader = build_loader(train_ds, micro_batch_size=args.micro_batch_size, seed=args.seed, shuffle=True)
    eval_loader = build_loader(eval_ds, micro_batch_size=args.micro_batch_size, seed=args.seed, shuffle=False)
    diag_batch = next(iter(eval_loader))

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(Path(args.model_dir), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    if args.fp32_params:
        model = model.float()
    else:
        dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
        model = model.to(dtype=dtype)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

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
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.resume_from:
        metrics_path.unlink()
    tokens_per_step = int(args.micro_batch_size) * int(args.block_size) * int(args.gradient_accumulation_steps)
    config_payload = vars(args) | {
        "arm_settings": settings,
        "data_metadata": metadata,
        "train_path_resolved": str(train_path),
        "eval_path_resolved": str(eval_path),
        "train_blocks": train_blocks,
        "eval_blocks": eval_blocks,
        "param_count": sum(p.numel() for p in model.parameters()),
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "master_params": "fp32" if args.fp32_params else ("bf16" if args.bf16 and device.type == "cuda" else "fp32"),
        "autocast_only": "bf16" if args.fp32_params and args.bf16 and device.type == "cuda" else None,
        "tokens_per_step": tokens_per_step,
        "actual_train_tokens": int(args.max_steps) * tokens_per_step,
        "labels": "input_ids",
        "init_from_config": True,
        "nb90_matched_except_train_token_budget": True,
        "checkpoint_semantics": "model-only snapshots plus optional rolling full latest checkpoint",
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    def write(row: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)

    def rng_state_payload() -> dict:
        payload = {
            "python_random_state": repr(random.getstate()),
            "numpy_random_state": repr(np.random.get_state()),
            "torch_cpu_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda" and torch.cuda.is_available():
            payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def restore_rng_state(payload: dict) -> None:
        if not payload:
            return
        if "python_random_state" in payload:
            random.setstate(eval(payload["python_random_state"], {"__builtins__": {}}, {}))
        if "numpy_random_state" in payload:
            np.random.set_state(eval(payload["numpy_random_state"], {"__builtins__": {}}, {"array": np.array, "dtype": np.dtype, "uint32": np.uint32}))
        if "torch_cpu_rng_state" in payload:
            torch.set_rng_state(payload["torch_cpu_rng_state"].detach().cpu())
        if device.type == "cuda" and torch.cuda.is_available() and "torch_cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all([x.detach().cpu() for x in payload["torch_cuda_rng_state_all"]])

    def atomic_torch_save(payload: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def save_checkpoints(step: int, current_tokens_seen: int) -> None:
        checkpoint_dir = out_dir / "checkpoints"
        common = {
            "step": int(step),
            "tokens_seen": int(current_tokens_seen),
            "arm": args.arm,
            "seed": int(args.seed),
            "config": config_payload,
        }
        if int(args.checkpoint_every_steps) > 0 and step % int(args.checkpoint_every_steps) == 0:
            model_path = checkpoint_dir / f"model_step{step:07d}_tokens{current_tokens_seen}.pt"
            atomic_torch_save(common | {"model_state_dict": model.state_dict()}, model_path)
            write(
                {
                    "phase": "checkpoint",
                    "kind": "model",
                    "step": step,
                    "tokens_seen": current_tokens_seen,
                    "path": str(model_path),
                    "seconds": time.time() - start,
                }
            )
        if int(args.full_checkpoint_every_steps) > 0 and step % int(args.full_checkpoint_every_steps) == 0:
            full_path = checkpoint_dir / "latest_full.pt"
            atomic_torch_save(
                common
                | {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "rng_state": rng_state_payload(),
                },
                full_path,
            )
            write(
                {
                    "phase": "checkpoint",
                    "kind": "full_latest",
                    "step": step,
                    "tokens_seen": current_tokens_seen,
                    "path": str(full_path),
                    "seconds": time.time() - start,
                }
            )

    start = time.time()

    train_iter = iter(train_loader)
    running_loss = 0.0
    running_count = 0
    tokens_seen = 0
    tokens_pending = 0
    micro_step = 0
    global_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        restore_rng_state(ckpt.get("rng_state", {}))
        global_step = int(ckpt["step"])
        tokens_seen = int(ckpt.get("tokens_seen", global_step * tokens_per_step))
        micro_step = global_step * int(args.gradient_accumulation_steps)
        skip_batches = micro_step
        for _ in range(skip_batches):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                next(train_iter)
        write(
            {
                "phase": "resume",
                "step": global_step,
                "micro_step": micro_step,
                "tokens_seen": tokens_seen,
                "resume_from": args.resume_from,
                "skipped_micro_batches": skip_batches,
                "seconds": time.time() - start,
            }
        )
    else:
        ev_loss, ev_ppl, ev_tokens = evaluate(model, eval_loader, device=device, use_bf16=args.bf16, max_eval_blocks=eval_blocks)
        write(
            {
                "phase": "eval",
                "step": 0,
                "tokens_seen": 0,
                "eval_loss": ev_loss,
                "eval_ppl": ev_ppl,
                "eval_tokens": ev_tokens,
                "seconds": time.time() - start,
            }
        )

    if args.benchmark_memory and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    while global_step < int(args.max_steps):
        try:
            input_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids = next(train_iter)
        input_ids = input_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(args.bf16 and device.type == "cuda")):
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / float(args.gradient_accumulation_steps)
        loss.backward()
        running_loss += float(loss.detach().cpu()) * float(args.gradient_accumulation_steps)
        running_count += 1
        tokens_pending += int(input_ids.numel())
        micro_step += 1
        if micro_step % int(args.gradient_accumulation_steps) != 0:
            continue

        if float(args.max_grad_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        set_lr(opt, muon_lr=args.muon_lr, vector_lr=args.vector_lr)
        opt.step()
        opt.zero_grad(set_to_none=True)

        global_step += 1
        tokens_seen += tokens_pending
        tokens_pending = 0
        seconds = time.time() - start
        capture_diag = False if args.disable_diagnostics else (
            global_step == 1 or global_step % int(args.diag_every_steps) == 0
        )
        if capture_diag:
            model.eval()
            diag_input_ids = diag_batch.to(device, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=bool(args.bf16 and device.type == "cuda"),
                ):
                    model(input_ids=diag_input_ids, labels=None)
            model.train()
        if global_step == 1 or global_step % int(args.log_every_steps) == 0:
            avg_loss = running_loss / max(1, running_count)
            write(
                {
                    "phase": "train",
                    "step": global_step,
                    "micro_step": micro_step,
                    "tokens_seen": tokens_seen,
                    "loss": avg_loss,
                    "ppl": math.exp(min(20.0, avg_loss)),
                    "seconds": seconds,
                }
            )
            running_loss = 0.0
            running_count = 0
        if global_step % int(args.eval_every_steps) == 0:
            ev_loss, ev_ppl, ev_tokens = evaluate(
                model, eval_loader, device=device, use_bf16=args.bf16, max_eval_blocks=eval_blocks
            )
            write(
                {
                    "phase": "eval",
                    "step": global_step,
                    "micro_step": micro_step,
                    "tokens_seen": tokens_seen,
                    "eval_loss": ev_loss,
                    "eval_ppl": ev_ppl,
                    "eval_tokens": ev_tokens,
                    "seconds": time.time() - start,
                }
            )
        save_checkpoints(global_step, tokens_seen)
        if tokens_seen >= train_blocks * int(args.block_size):
            break

    if args.benchmark_memory and device.type == "cuda":
        write(
            {
                "phase": "memory_benchmark",
                "step": global_step,
                "micro_step": micro_step,
                "tokens_seen": tokens_seen,
                "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
                "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
                "seconds": time.time() - start,
            }
        )

    if global_step > 0 and not args.benchmark_memory:
        checkpoint_dir = out_dir / "checkpoints"
        final_model_path = checkpoint_dir / f"model_final_step{global_step:07d}_tokens{tokens_seen}.pt"
        final_common = {
            "step": int(global_step),
            "tokens_seen": int(tokens_seen),
            "arm": args.arm,
            "seed": int(args.seed),
            "config": config_payload,
        }
        atomic_torch_save(final_common | {"model_state_dict": model.state_dict()}, final_model_path)
        atomic_torch_save(
            final_common
            | {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "rng_state": rng_state_payload(),
            },
            checkpoint_dir / "latest_full.pt",
        )
        write(
            {
                "phase": "checkpoint",
                "kind": "final",
                "step": global_step,
                "tokens_seen": tokens_seen,
                "path": str(final_model_path),
                "seconds": time.time() - start,
            }
        )


if __name__ == "__main__":
    main()
