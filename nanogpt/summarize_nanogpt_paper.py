from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ARM_ORDER = (
    "Hybrid Muon",
    "SCION-style Sign",
    "AF-Muon",
    "AdamW tied / AdamW 1D",
    "AF-Muon tied / AdamW 1D",
    "AdamW tied / RMS-LMO 1D",
    "AF-Muon tied / RMS-LMO 1D",
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"bad JSON in {path} line {line_no}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def infer_arm(run_name: str, cfg: dict) -> str:
    arm = cfg.get("arm")
    if arm == "muon":
        return "Hybrid Muon"
    if arm == "sign":
        return "SCION-style Sign"
    if arm == "afmoun":
        return "AF-Muon"
    if arm == "fact_adamw_adamw":
        return "AdamW tied / AdamW 1D"
    if arm == "fact_c3_adamw":
        return "AF-Muon tied / AdamW 1D"
    if arm == "fact_adamw_rms":
        return "AdamW tied / RMS-LMO 1D"
    if arm == "fact_c3_rms":
        return "AF-Muon tied / RMS-LMO 1D"
    low = run_name.lower()
    if "afmoun" in low:
        return "AF-Muon"
    if "scion" in low or "sign" in low:
        return "SCION-style Sign"
    if "hybrid" in low or "muon" in low:
        return "Hybrid Muon"
    return "unknown"


def infer_seed(run_name: str, cfg: dict) -> int | None:
    if cfg.get("seed") is not None:
        return int(cfg["seed"])
    match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:[_-]|$)", run_name.lower())
    return int(match.group(1)) if match else None


def finite_or_none(value):
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def summarize_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return None

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    rows = load_jsonl(metrics_path)
    evals = [r for r in rows if r.get("phase") == "eval" and finite_or_none(r.get("eval_loss")) is not None]
    trains = [r for r in rows if r.get("phase") == "train" and finite_or_none(r.get("loss")) is not None]
    diags = [r for r in rows if r.get("phase") == "diagnostic"]
    ckpts = [r for r in rows if r.get("phase") == "checkpoint"]
    if not evals:
        return None

    last_eval = max(evals, key=lambda r: int(r.get("step", -1)))
    best_eval = min(evals, key=lambda r: float(r.get("eval_loss", "inf")))
    last_train = max(trains, key=lambda r: int(r.get("step", -1))) if trains else {}
    last_diag = max(diags, key=lambda r: int(r.get("step", -1))) if diags else {}
    last_ckpt = max(ckpts, key=lambda r: int(r.get("step", -1))) if ckpts else {}

    return {
        "run": run_dir.name,
        "optimizer": infer_arm(run_dir.name, cfg),
        "seed": infer_seed(run_dir.name, cfg),
        "params": cfg.get("param_count"),
        "master_params": cfg.get("master_params"),
        "autocast": cfg.get("autocast"),
        "activation_mode": cfg.get("activation_mode"),
        "scaled_relu_sq_effective": cfg.get("scaled_relu_sq_effective"),
        "warmdown_frac": cfg.get("warmdown_frac"),
        "max_grad_norm": cfg.get("max_grad_norm"),
        "matrix_lr": cfg.get("muon_lr"),
        "muon_lr_multiplier": cfg.get("muon_lr_multiplier"),
        "multiplier_mode": cfg.get("multiplier_mode"),
        "vector_lr": cfg.get("vector_lr"),
        "matrix_wd": cfg.get("matrix_weight_decay"),
        "aux_wd": cfg.get("aux_weight_decay"),
        "rho_hidden": cfg.get("rho_hidden"),
        "rho_output": cfg.get("rho_output"),
        "tied_rule": cfg.get("arm_settings", {}).get("tied_rule"),
        "vector_rule": cfg.get("arm_settings", {}).get("vector_rule"),
        "tied_cap": cfg.get("arm_settings", {}).get("tied_cap"),
        "tied_scale": cfg.get("arm_settings", {}).get("tied_scale"),
        "step": last_eval.get("step"),
        "tokens": last_eval.get("tokens_seen"),
        "eval_loss": last_eval.get("eval_loss"),
        "eval_ppl": last_eval.get("eval_ppl"),
        "best_step": best_eval.get("step"),
        "best_tokens": best_eval.get("tokens_seen"),
        "best_loss": best_eval.get("eval_loss"),
        "best_ppl": best_eval.get("eval_ppl"),
        "train_loss": last_train.get("loss"),
        "train_ppl": last_train.get("ppl"),
        "grad_norm": last_train.get("grad_norm_preclip"),
        "tied_rms": last_diag.get("tied_rms"),
        "tied_max_abs": last_diag.get("tied_max_abs"),
        "matrix_rms_mean": last_diag.get("matrix_rms_mean"),
        "checkpoint_step": last_ckpt.get("step"),
        "checkpoint_path": last_ckpt.get("path"),
    }


def fmt(x, digits: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return "NA"
    if abs(xf) >= 1e4 or (0 < abs(xf) < 1e-3):
        return f"{xf:.3e}"
    return f"{xf:.{digits}f}"


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["optimizer"] in ARM_ORDER:
            grouped[row["optimizer"]].append(row)

    out = []
    for optimizer in ARM_ORDER:
        group = sorted(grouped.get(optimizer, []), key=lambda r: int(r["seed"]))
        if not group:
            continue
        agg = {
            "optimizer": optimizer,
            "seeds": ",".join(str(r["seed"]) for r in group),
            "n": len(group),
        }
        for key in (
            "params",
            "step",
            "tokens",
            "eval_loss",
            "eval_ppl",
            "best_loss",
            "best_ppl",
            "train_loss",
            "grad_norm",
            "tied_rms",
            "tied_max_abs",
            "matrix_rms_mean",
        ):
            values = np.asarray(
                [float(r[key]) for r in group if finite_or_none(r.get(key)) is not None],
                dtype=np.float64,
            )
            if values.size:
                agg[f"{key}_mean"] = float(values.mean())
                agg[f"{key}_sd"] = float(values.std(ddof=1)) if values.size >= 2 else math.nan
        sample = group[0]
        for key in (
            "master_params",
            "autocast",
            "activation_mode",
            "warmdown_frac",
            "max_grad_norm",
            "matrix_lr",
            "muon_lr_multiplier",
            "multiplier_mode",
            "vector_lr",
            "matrix_wd",
            "aux_wd",
            "rho_hidden",
            "rho_output",
            "tied_cap",
            "tied_scale",
        ):
            agg[key] = sample.get(key)
        out.append(agg)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, rows: list[dict], cols: list[str]) -> None:
    print("\n" + title)
    print("-" * len(title))
    print(" | ".join(cols))
    print(" | ".join(["---"] * len(cols)))
    for row in rows:
        print(" | ".join(fmt(row.get(c)) for c in cols))


def print_latex_rows(rows: list[dict]) -> None:
    print("\n=== latex rows ===")
    for row in rows:
        name = row["optimizer"]
        final_loss = f"{fmt(row.get('eval_loss_mean'))} $\\pm$ {fmt(row.get('eval_loss_sd'))}"
        final_ppl = f"{fmt(row.get('eval_ppl_mean'), 2)} $\\pm$ {fmt(row.get('eval_ppl_sd'), 2)}"
        best_loss = f"{fmt(row.get('best_loss_mean'))} $\\pm$ {fmt(row.get('best_loss_sd'))}"
        best_ppl = f"{fmt(row.get('best_ppl_mean'), 2)} $\\pm$ {fmt(row.get('best_ppl_sd'), 2)}"
        print(f"{name} & {final_loss} & {final_ppl} & {best_loss} & {best_ppl} \\\\")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper summary for NanoGPT AF-Muon runs.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--csv-prefix", default="outputs/nanogpt_paper")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = []
    for run_dir in sorted(run_root.glob("*")):
        if run_dir.is_dir() and run_dir.name != "_logs":
            row = summarize_run(run_dir)
            if row:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no complete NanoGPT runs found under {run_root}")

    rows.sort(key=lambda r: (ARM_ORDER.index(r["optimizer"]) if r["optimizer"] in ARM_ORDER else 99, int(r["seed"])))
    agg = aggregate(rows)

    print("=" * 100)
    print(run_root)
    print("=" * 100)
    print_table(
        "Per-Seed Performance",
        rows,
        ["optimizer", "seed", "step", "tokens", "eval_loss", "eval_ppl", "best_loss", "best_ppl", "train_loss", "grad_norm"],
    )
    print_table(
        "Per-Seed Diagnostics",
        rows,
        ["optimizer", "seed", "tied_rms", "tied_max_abs", "matrix_rms_mean", "matrix_wd", "aux_wd", "tied_rule", "vector_rule", "tied_cap", "tied_scale"],
    )
    print_table(
        "Aggregate Performance",
        agg,
        ["optimizer", "n", "eval_loss_mean", "eval_loss_sd", "eval_ppl_mean", "eval_ppl_sd", "best_loss_mean", "best_loss_sd", "best_ppl_mean", "best_ppl_sd"],
    )
    print_table(
        "Protocol",
        agg,
        ["optimizer", "master_params", "autocast", "activation_mode", "warmdown_frac", "max_grad_norm", "matrix_lr", "muon_lr_multiplier", "matrix_wd", "aux_wd", "rho_output"],
    )
    print_latex_rows(agg)

    prefix = Path(args.csv_prefix)
    write_csv(prefix.with_name(prefix.name + "_per_seed.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_aggregate.csv"), agg)
    print(f"\nWrote CSV: {prefix.with_name(prefix.name + '_per_seed.csv')}")
    print(f"Wrote CSV: {prefix.with_name(prefix.name + '_aggregate.csv')}")


if __name__ == "__main__":
    main()
