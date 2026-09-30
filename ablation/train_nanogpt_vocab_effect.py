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

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanogpt.optimizers import build_afmoun_v2, build_scion_sign_v2
from nanogpt.optimizers.afmoun_v2 import _finite_cap_oracle_fp32, _row_rms_oracle_fp32
from nanogpt.train_nanogpt import (
    NanoGPT,
    autocast_dtype,
    find_data_file,
    load_metadata,
    make_grad_scaler,
    make_loader,
    schedule_mult,
    set_lrs,
)


def write_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def finite(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


@torch.no_grad()
def weight_stats(model: NanoGPT, *, used_vocab_cutoff: int) -> dict:
    tied = model.get_input_embeddings().weight.detach().float()
    matrix_vals = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            matrix_vals.append(module.weight.detach().float().square().mean())
    matrix_rms = torch.stack(matrix_vals).mean().sqrt().item() if matrix_vals else float("nan")

    used_cut = max(0, min(int(used_vocab_cutoff), int(tied.shape[0])))
    used = tied[:used_cut]
    unused = tied[used_cut:] if used_cut < tied.shape[0] else None

    out = {
        "tied_rms": tied.square().mean().sqrt().item(),
        "tied_max_abs": tied.abs().amax().item(),
        "matrix_rms_mean": matrix_rms,
        "used_vocab_cutoff": used_cut,
        "used_row_rms": used.square().mean().sqrt().item() if used.numel() else None,
        "used_row_max_abs": used.abs().amax().item() if used.numel() else None,
        "unused_row_rms": unused.square().mean().sqrt().item() if unused is not None and unused.numel() else None,
        "unused_row_max_abs": unused.abs().amax().item() if unused is not None and unused.numel() else None,
    }
    return out


def percentile_tensor(x: torch.Tensor, q: float, *, max_values: int = 2_000_000) -> float | None:
    if x.numel() == 0:
        return None
    flat = x.detach().flatten()
    if flat.numel() > int(max_values):
        stride = int(math.ceil(flat.numel() / float(max_values)))
        flat = flat[::stride][: int(max_values)]
    return float(torch.quantile(flat.float(), float(q)).item())


@torch.no_grad()
def tied_geometry_diagnostics(opt, model: NanoGPT, *, cap: float, scale: float, rho_hidden: float, rho_output: float, chunk_rows: int, used_vocab_cutoff: int) -> dict:
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
    lr = float(tied_group.get("lr", 0.0))
    actual_cap = float(tied_group.get("tied_cap", 1.0 if cap == 1.0 else cap))
    actual_scale = float(tied_group.get("tied_scale", scale))
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
    used_update_energy = 0.0
    unused_update_energy = 0.0
    used_count = 0
    unused_count = 0

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

        raw = rms
        raw_e = raw.square()
        raw_energy += float(raw_e.sum(dtype=torch.float64).item())
        raw_tail_energy += float(raw_e[raw.abs() > actual_cap].sum(dtype=torch.float64).item()) if actual_cap > 0 else 0.0

        update = af * float(tied_lr)
        upd_e = update.square()
        update_energy += float(upd_e.sum(dtype=torch.float64).item())
        used_cut = max(0, min(int(used_vocab_cutoff), rows))
        used_start = max(start, 0)
        used_end = min(end, used_cut)
        if used_end > used_start:
            loc0, loc1 = used_start - start, used_end - start
            used_update_energy += float(upd_e[loc0:loc1].sum(dtype=torch.float64).item())
            used_count += int(upd_e[loc0:loc1].numel())
        if end > used_cut:
            loc0 = max(used_cut - start, 0)
            unused_update_energy += float(upd_e[loc0:].sum(dtype=torch.float64).item())
            unused_count += int(upd_e[loc0:].numel())

        rn = m.norm(dim=1)
        row_norms.append(rn.detach().cpu())
        entry_abs_samples.append(m.abs().flatten()[:: max(1, m.numel() // 200_000)].detach().cpu())
        raw_mean_sq = raw.square().mean(dim=1).clamp_min(1e-30)
        af_mean_sq = af.square().mean(dim=1).clamp_min(1e-30)
        papr_raw.append((raw.abs().amax(dim=1).square() / raw_mean_sq).detach().cpu())
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
        "used_update_rms": math.sqrt(used_update_energy / max(1, used_count)) if used_count else None,
        "unused_update_rms": math.sqrt(unused_update_energy / max(1, unused_count)) if unused_count else None,
    }


@torch.no_grad()
def evaluate_with_logit_stats(model, loader, *, device, autocast: str, max_batches: int, logit_stats_batches: int) -> dict:
    model.eval()
    losses = []
    tokens = 0
    centered_sq = 0.0
    logit_count = 0
    max_abs = 0.0
    entropy_sum = 0.0
    target_logit_sum = 0.0
    enabled = autocast != "none" and device.type == "cuda"
    dtype = autocast_dtype(autocast)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            out = model(x, y)
            loss = out["loss"]
            logits = out["logits"]
        losses.append(float(loss.detach().float().item()))
        tokens += int(y.numel())

        if i >= int(logit_stats_batches):
            continue
        lf = logits.detach().float()
        vocab = int(lf.shape[-1])
        logit_sum = lf.sum(dim=-1, dtype=torch.float64)
        logit_sq_sum = lf.square().sum(dim=-1, dtype=torch.float64)
        centered_sq += float((logit_sq_sum - logit_sum.square() / float(vocab)).sum().item())
        logit_count += int(lf.numel())
        max_abs = max(max_abs, float(lf.abs().amax().item()))
        log_probs = torch.log_softmax(lf, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_sum += float(entropy.sum(dtype=torch.float64).item())
        target_logit_sum += float(lf.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum(dtype=torch.float64).item())

    model.train()
    loss = float(sum(losses) / max(1, len(losses)))
    return {
        "eval_loss": loss,
        "eval_ppl": math.exp(min(20.0, loss)),
        "eval_tokens": tokens,
        "logit_stats_batches": int(logit_stats_batches),
        "centered_logit_rms": math.sqrt(centered_sq / max(1, logit_count)),
        "logit_max_abs": max_abs,
        "entropy_mean": entropy_sum / max(1, tokens),
        "target_logit_mean": target_logit_sum / max(1, tokens),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NanoGPT vocab-size tied-table diagnostic sweep worker.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", choices=["sign", "afmoun"], required=True)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--autocast", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--mlp-hidden", type=int, default=3072)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--used-vocab-cutoff", type=int, default=49152)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=954)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=16)
    p.add_argument("--eval-tokens", type=int, default=5_000_000)
    p.add_argument("--logit-stats-batches", type=int, default=16)
    p.add_argument("--eval-every-steps", type=int, default=250)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--diag-every-steps", type=int, default=250)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--full-checkpoint-every-steps", type=int, default=0)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--vector-lr", type=float, default=3e-4)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--matrix-weight-decay", type=float, default=0.1)
    p.add_argument("--rho-hidden", type=float, default=50.0)
    p.add_argument("--rho-output", type=float, default=3000.0)
    p.add_argument("--tied-cap", type=float, default=3.0)
    p.add_argument("--tied-scale", type=float, default=0.5)
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--check-config", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = NanoGPT(
        vocab_size=args.vocab_size,
        layers=args.layers,
        heads=args.heads,
        head_dim=args.head_dim,
        mlp_hidden=args.mlp_hidden,
        block_size=args.block_size,
        scaled_relu_sq=True,
    ).float().to(device)

    if args.arm == "sign":
        opt = build_scion_sign_v2(
            model,
            muon_lr=args.muon_lr,
            vector_lr=args.vector_lr,
            momentum=args.momentum,
            matrix_weight_decay=args.matrix_weight_decay,
            rho_hidden=args.rho_hidden,
            rho_output=args.rho_output,
            chunk_rows=args.chunk_rows,
        )
        settings = {"optimizer": "scion_sign_v2", "tied_cap": 1.0, "tied_scale": 1.0}
    else:
        opt = build_afmoun_v2(
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
        )
        settings = {"optimizer": "afmoun_v2", "tied_cap": args.tied_cap, "tied_scale": args.tied_scale}

    if args.check_config:
        print(json.dumps({
            "ok": True,
            "arm": args.arm,
            "vocab_size": args.vocab_size,
            "used_vocab_cutoff": args.used_vocab_cutoff,
            "param_count": sum(p.numel() for p in model.parameters()),
            "tokens_per_step": int(args.batch_size) * int(args.block_size),
            "iterations": args.iterations,
            "train_tokens": int(args.batch_size) * int(args.block_size) * int(args.iterations),
            "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "optimizer_state_groups": [g.get("role") for g in opt.param_groups],
            "arm_settings": settings,
        }, indent=2))
        return

    data_dir = Path(args.data_dir)
    meta = load_metadata(data_dir)
    train_path = find_data_file(data_dir, ["train_tokens_uint16.bin", "train_tokens_uint32.bin", "train.bin"])
    eval_path = find_data_file(data_dir, ["eval_tokens_uint16.bin", "eval_tokens_uint32.bin", "val_tokens_uint16.bin", "val.bin"])
    token_dtype = "uint32" if "uint32" in train_path.name else meta.get("dtype", "uint16")

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
    config = vars(args) | {
        "arm_settings": settings,
        "data_metadata": meta,
        "train_path_resolved": str(train_path),
        "eval_path_resolved": str(eval_path),
        "param_count": sum(p.numel() for p in model.parameters()),
        "tokens_per_step": tokens_per_step,
        "actual_train_tokens": tokens_per_step * int(args.iterations),
        "master_params": "fp32",
        "autocast_only": args.autocast,
        "activation_mode": "all_scaled",
        "gradient_accumulation_steps": grad_accum,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    scaler = make_grad_scaler(args.autocast == "fp16" and device.type == "cuda")
    data_iter = iter(train_loader)
    enabled = args.autocast != "none" and device.type == "cuda"
    dtype = autocast_dtype(args.autocast)
    step = 0
    micro_step = 0
    start_time = time.time()
    base_lrs = {"matrix": float(args.muon_lr), "vector": float(args.vector_lr), "aux": float(args.vector_lr)}

    initial_eval = evaluate_with_logit_stats(model, eval_loader, device=device, autocast=args.autocast, max_batches=eval_blocks, logit_stats_batches=args.logit_stats_batches)
    write_jsonl(metrics, {"phase": "eval", "step": 0, "tokens_seen": 0, **initial_eval})

    while step < int(args.iterations):
        total_loss = 0.0
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
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                loss = model(x, y)["loss"] / grad_accum
            total_loss += float(loss.detach().float().item())
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        step += 1
        set_lrs(opt, base_lrs, 1.0)
        grad_norm = None
        if float(args.max_grad_norm) > 0.0:
            if scaler is not None:
                scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm)).item())
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        tokens_seen = step * tokens_per_step
        seconds = time.time() - start_time
        if step == 1 or step % int(args.log_every_steps) == 0:
            write_jsonl(metrics, {
                "phase": "train",
                "step": step,
                "micro_step": micro_step,
                "tokens_seen": tokens_seen,
                "loss": total_loss,
                "ppl": math.exp(min(20.0, total_loss)),
                "lr_mult": schedule_mult(step, int(args.iterations), 0.0),
                "matrix_lr": opt.param_groups[0]["lr"],
                "vector_lr": next((g["lr"] for g in opt.param_groups if g.get("role") == "vector"), None),
                "grad_norm_preclip": grad_norm,
                "seconds": seconds,
            })

        if step == 1 or step % int(args.diag_every_steps) == 0:
            diag = {
                "phase": "diagnostic",
                "step": step,
                "tokens_seen": tokens_seen,
                **weight_stats(model, used_vocab_cutoff=args.used_vocab_cutoff),
                **tied_geometry_diagnostics(
                    opt,
                    model,
                    cap=1.0 if args.arm == "sign" else args.tied_cap,
                    scale=1.0 if args.arm == "sign" else args.tied_scale,
                    rho_hidden=args.rho_hidden,
                    rho_output=args.rho_output,
                    chunk_rows=args.chunk_rows,
                    used_vocab_cutoff=args.used_vocab_cutoff,
                ),
                "seconds": seconds,
            }
            write_jsonl(metrics, diag)

        if step % int(args.eval_every_steps) == 0 or step == int(args.iterations):
            ev = evaluate_with_logit_stats(model, eval_loader, device=device, autocast=args.autocast, max_batches=eval_blocks, logit_stats_batches=args.logit_stats_batches)
            write_jsonl(metrics, {"phase": "eval", "step": step, "tokens_seen": tokens_seen, **ev, "seconds": time.time() - start_time})

        if int(args.full_checkpoint_every_steps) > 0 and (step % int(args.full_checkpoint_every_steps) == 0 or step == int(args.iterations)):
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "step": step,
                "tokens_seen": tokens_seen,
                "config": config,
            }, ck_dir / "latest_full.pt")
            write_jsonl(metrics, {"phase": "checkpoint", "kind": "full_latest", "step": step, "tokens_seen": tokens_seen, "path": str(ck_dir / "latest_full.pt"), "seconds": time.time() - start_time})


if __name__ == "__main__":
    main()
