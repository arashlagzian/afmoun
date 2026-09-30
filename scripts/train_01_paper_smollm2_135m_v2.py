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
from optimizers.afmoun_v2 import _finite_cap_oracle_fp32, _row_rms_oracle_fp32


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


def optimizer_state_dtypes(opt) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                out.setdefault(str(key), set()).add(str(value.dtype))
    return {key: sorted(values) for key, values in sorted(out.items())}


def percentile_tensor(x: torch.Tensor, q: float, *, max_values: int = 2_000_000) -> float | None:
    if x.numel() == 0:
        return None
    flat = x.detach().flatten()
    if flat.numel() > int(max_values):
        stride = int(math.ceil(flat.numel() / float(max_values)))
        flat = flat[::stride][: int(max_values)]
    return float(torch.quantile(flat.float(), float(q)).item())


@torch.no_grad()
def tied_geometry_diagnostics(
    model,
    opt,
    *,
    cap: float | None,
    scale: float | None,
    rho_hidden: float,
    rho_output: float,
    chunk_rows: int,
) -> dict:
    tied = model.get_input_embeddings().weight
    tied_group = None
    for group in opt.param_groups:
        if any(p is tied for p in group["params"]):
            tied_group = group
            break
    if tied_group is None:
        return {}

    state = opt.state.get(tied, {})
    momentum = state.get("momentum_buffer")
    if momentum is None:
        return {"tied_state_missing": True}

    mom = momentum.detach()
    rows, width = int(mom.shape[0]), int(mom.shape[1])
    actual_cap = float(tied_group.get("tied_cap", 1.0 if cap is None else cap))
    actual_scale = float(tied_group.get("tied_scale", 1.0 if scale is None else scale))
    lr = float(tied_group.get("lr", 0.0))
    tied_lr = lr * float(rho_output) / float(rho_hidden) * actual_scale / float(width)

    dot_af_sign = 0.0
    dot_af_rms = 0.0
    dot_sign_rms = 0.0
    norm_af = 0.0
    norm_sign = 0.0
    norm_rms = 0.0
    obj_af = 0.0
    obj_sign = 0.0
    obj_rms = 0.0
    capped = 0
    nonzero = 0
    capped_rows = 0
    active_rows = 0
    raw_tail_energy = 0.0
    raw_energy = 0.0
    update_energy = 0.0
    row_norms = []
    entry_abs_samples = []
    papr_raw = []
    papr_af = []

    for start in range(0, rows, int(chunk_rows)):
        end = min(rows, start + int(chunk_rows))
        m = mom[start:end].float()
        if actual_cap == 1.0:
            af = m.sign()
        elif math.isinf(actual_cap) or actual_cap >= math.sqrt(float(width)):
            af = _row_rms_oracle_fp32(m)
        else:
            af = _finite_cap_oracle_fp32(m, cap=actual_cap, bisection_steps=32, max_bracket_steps=128)
        sign = m.sign()
        rms = _row_rms_oracle_fp32(m)

        dot_af_sign += float((af * sign).sum(dtype=torch.float64).item())
        dot_af_rms += float((af * rms).sum(dtype=torch.float64).item())
        dot_sign_rms += float((sign * rms).sum(dtype=torch.float64).item())
        norm_af += float(af.square().sum(dtype=torch.float64).item())
        norm_sign += float(sign.square().sum(dtype=torch.float64).item())
        norm_rms += float(rms.square().sum(dtype=torch.float64).item())
        obj_af += float((m * af).sum(dtype=torch.float64).item())
        obj_sign += float((m * sign).sum(dtype=torch.float64).item())
        obj_rms += float((m * rms).sum(dtype=torch.float64).item())

        nz = m != 0
        nonzero += int(nz.sum().item())
        cap_mask = af.abs() >= (actual_cap - 1e-5)
        capped += int((cap_mask & nz).sum().item())
        active = nz.any(dim=1)
        active_rows += int(active.sum().item())
        capped_rows += int((cap_mask & nz).any(dim=1).sum().item())

        raw_e = rms.square()
        raw_energy += float(raw_e.sum(dtype=torch.float64).item())
        if actual_cap > 0:
            raw_tail_energy += float(raw_e[rms.abs() > actual_cap].sum(dtype=torch.float64).item())

        update = af * tied_lr
        update_energy += float(update.square().sum(dtype=torch.float64).item())

        row_norms.append(m.norm(dim=1).detach().cpu())
        entry_abs_samples.append(m.abs().flatten()[:: max(1, m.numel() // 200_000)].detach().cpu())
        raw_mean_sq = rms.square().mean(dim=1).clamp_min(1e-30)
        af_mean_sq = af.square().mean(dim=1).clamp_min(1e-30)
        papr_raw.append((rms.abs().amax(dim=1).square() / raw_mean_sq).detach().cpu())
        papr_af.append((af.abs().amax(dim=1).square() / af_mean_sq).detach().cpu())

    row_norms_t = torch.cat(row_norms) if row_norms else torch.empty(0)
    entry_abs_t = torch.cat(entry_abs_samples) if entry_abs_samples else torch.empty(0)
    papr_raw_t = torch.cat(papr_raw) if papr_raw else torch.empty(0)
    papr_af_t = torch.cat(papr_af) if papr_af else torch.empty(0)

    def cos(dot: float, a: float, b: float) -> float | None:
        den = math.sqrt(max(a, 0.0) * max(b, 0.0))
        return dot / den if den > 0 else None

    weight_rms = tied.detach().float().square().mean().sqrt().item()
    update_rms = math.sqrt(update_energy / max(1, tied.numel()))

    return {
        "tied_state_missing": False,
        "tied_rule_role": tied_group.get("role"),
        "tied_lmo_cap": actual_cap,
        "tied_lmo_scale": actual_scale,
        "tied_effective_lr": tied_lr,
        "tied_momentum_rms": mom.float().square().mean().sqrt().item(),
        "tied_momentum_max_abs": mom.float().abs().amax().item(),
        "tied_momentum_row_norm_mean": row_norms_t.mean().item() if row_norms_t.numel() else None,
        "tied_momentum_row_norm_p95": percentile_tensor(row_norms_t, 0.95),
        "tied_momentum_row_norm_p99": percentile_tensor(row_norms_t, 0.99),
        "tied_momentum_entry_abs_p99": percentile_tensor(entry_abs_t, 0.99),
        "tied_momentum_entry_abs_max": mom.float().abs().amax().item(),
        "coords_capped_frac": capped / max(1, nonzero),
        "rows_clipped_frac": capped_rows / max(1, active_rows),
        "raw_tail_energy_frac": raw_tail_energy / max(raw_energy, 1e-30),
        "p99_papr_raw": percentile_tensor(papr_raw_t, 0.99),
        "p99_papr_af": percentile_tensor(papr_af_t, 0.99),
        "cos_af_sign": cos(dot_af_sign, norm_af, norm_sign),
        "cos_af_row_rms": cos(dot_af_rms, norm_af, norm_rms),
        "cos_sign_row_rms": cos(dot_sign_rms, norm_sign, norm_rms),
        "objective_af_over_sign": obj_af / obj_sign if abs(obj_sign) > 1e-30 else None,
        "objective_af_over_row_rms": obj_af / obj_rms if abs(obj_rms) > 1e-30 else None,
        "objective_sign_over_row_rms": obj_sign / obj_rms if abs(obj_rms) > 1e-30 else None,
        "tied_update_rms": update_rms,
        "tied_update_to_weight_rms": update_rms / max(weight_rms, 1e-30),
    }


@torch.no_grad()
def diagnostic_payload(
    model,
    opt,
    *,
    step: int,
    micro_step: int,
    tokens_seen: int,
    seconds: float,
    tied_cap: float | None,
    tied_scale: float | None,
    rho_hidden: float,
    rho_output: float,
    chunk_rows: int,
    rich: bool,
) -> dict:
    tied = model.get_input_embeddings().weight
    tied_ptr = tied.data_ptr()
    matrix_rms = []
    for p in model.parameters():
        if p.ndim >= 2 and p.data_ptr() != tied_ptr:
            matrix_rms.append(float(p.detach().float().square().mean().sqrt().cpu()))
    payload = {
        "phase": "diagnostic",
        "step": step,
        "micro_step": micro_step,
        "tokens_seen": tokens_seen,
        "seconds": seconds,
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "optimizer_state_dtypes": optimizer_state_dtypes(opt),
        "tied_name": "model.embed_tokens.weight",
        "tied_rms": float(tied.detach().float().square().mean().sqrt().cpu()),
        "tied_max_abs": float(tied.detach().float().abs().max().cpu()),
        "matrix_rms_mean": float(np.mean(matrix_rms)) if matrix_rms else None,
    }
    if rich:
        payload.update(tied_geometry_diagnostics(
            model,
            opt,
            cap=tied_cap,
            scale=tied_scale,
            rho_hidden=rho_hidden,
            rho_output=rho_output,
            chunk_rows=chunk_rows,
        ))
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper SmolLM2-135M V2 multiseed worker.")
    p.add_argument("--model-dir", required=True)
    p.add_argument(
        "--data-dir",
        required=True,
    )
    p.add_argument("--output-dir", default="runs/01_paper_smollm2_135m_v2")
    p.add_argument("--arm", choices=["muon", "sign", "afmoun"], default="afmoun")
    p.add_argument("--train-tokens", type=int, default=2_500_000_000)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--max-steps", type=int, default=4_768)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--aux-weight-decay", type=float, default=0.0)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--diag-every-steps", type=int, default=250)
    p.add_argument("--disable-diagnostics", action="store_true")
    p.add_argument("--disable-rich-diagnostics", action="store_true")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp32-params", action="store_true", default=True)
    p.add_argument("--benchmark-memory", action="store_true", help="Reset CUDA peak stats before training and report exact peak memory.")
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

    data_dir = Path(args.data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    train_path = resolve_data_path(data_dir, metadata["train_path"])
    eval_path = resolve_data_path(data_dir, metadata["eval_path"])
    train_blocks = min(int(args.train_tokens) // int(args.block_size), int(metadata["train_blocks"]))
    eval_blocks = min(max(1, int(args.eval_tokens) // int(args.block_size)), int(metadata["eval_blocks"]))
    token_dtype = metadata.get("dtype", "uint16")

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
    if metrics_path.exists():
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
        "gradient_checkpointing": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    def write(row: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)

    start = time.time()
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

    train_iter = iter(train_loader)
    running_loss = 0.0
    running_count = 0
    tokens_seen = 0
    tokens_pending = 0
    micro_step = 0
    global_step = 0

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
            write(diagnostic_payload(
                model,
                opt,
                step=global_step,
                micro_step=micro_step,
                tokens_seen=tokens_seen,
                seconds=seconds,
                tied_cap=settings["tied_cap"],
                tied_scale=settings["tied_scale"],
                rho_hidden=args.rho_hidden,
                rho_output=args.rho_output,
                chunk_rows=args.chunk_rows,
                rich=not args.disable_rich_diagnostics,
            ))

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


if __name__ == "__main__":
    main()
