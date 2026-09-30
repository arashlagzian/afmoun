from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


LABELS = {
    "hybrid_muon": "Hybrid Muon",
    "scion_c1_s1": "SCION-Sign",
    "afmoun_c3_s0p5": "AF-Muon",
}


def label(name: str, cfg: dict) -> str:
    if cfg.get("arm") == "muon":
        return "Hybrid Muon"
    if cfg.get("arm") == "sign":
        return "SCION-Sign"
    if cfg.get("arm") == "afmoun":
        return "AF-Muon"
    for key, value in LABELS.items():
        if key in name:
            return value
    return name


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fmt(x):
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return str(x)
    if abs(xf) >= 1e4 or (0 < abs(xf) < 1e-3):
        return f"{xf:.3e}"
    return f"{xf:.4f}"


def summarize_run(d: Path) -> dict | None:
    metrics = d / "metrics.jsonl"
    if not metrics.exists():
        return None
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").exists() else {}
    rows = read_rows(metrics)
    train = [r for r in rows if r.get("phase") == "train"]
    evals = [r for r in rows if r.get("phase") == "eval"]
    diags = [r for r in rows if r.get("phase") == "diagnostic"]
    last_eval = evals[-1] if evals else {}
    best_eval = min(evals, key=lambda r: float(r.get("eval_loss", "inf"))) if evals else {}
    last_train = train[-1] if train else {}
    last_diag = diags[-1] if diags else {}
    return {
        "run": d.name,
        "optimizer": label(d.name, cfg),
        "seed": cfg.get("seed"),
        "params": cfg.get("param_count"),
        "master_params": cfg.get("master_params"),
        "autocast": cfg.get("autocast"),
        "step": last_eval.get("step"),
        "tokens": last_eval.get("tokens_seen"),
        "eval_loss": last_eval.get("eval_loss"),
        "eval_ppl": last_eval.get("eval_ppl"),
        "best_loss": best_eval.get("eval_loss"),
        "best_step": best_eval.get("step"),
        "train_loss": last_train.get("loss"),
        "grad_norm": last_train.get("grad_norm_preclip"),
        "tied_rms": last_diag.get("tied_rms"),
        "tied_max_abs": last_diag.get("tied_max_abs"),
        "matrix_rms_mean": last_diag.get("matrix_rms_mean"),
        "matrix_wd": cfg.get("matrix_weight_decay"),
        "aux_wd": cfg.get("aux_weight_decay"),
        "rho_output": cfg.get("rho_output"),
    }


def print_table(title: str, rows: list[dict], cols: list[str]) -> None:
    print("\n" + title)
    print("-" * len(title))
    print(" | ".join(cols))
    print(" | ".join(["---"] * len(cols)))
    for row in rows:
        print(" | ".join(fmt(row.get(c)) for c in cols))


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize NanoGPT AF-Muon runs.")
    p.add_argument("--run-roots", nargs="+", required=True)
    p.add_argument("--csv-out", default="")
    args = p.parse_args()

    all_rows = []
    for root_s in args.run_roots:
        root = Path(root_s)
        print("\n" + "=" * 100)
        print(root)
        print("=" * 100)
        rows = []
        for d in sorted(root.glob("*")):
            if d.is_dir() and d.name != "_logs":
                row = summarize_run(d)
                if row:
                    rows.append(row)
                    all_rows.append(row)
        if not rows:
            print("No metrics.jsonl files found.")
            continue
        print_table("Performance", rows, ["optimizer", "seed", "step", "tokens", "eval_loss", "eval_ppl", "best_loss", "train_loss", "grad_norm"])
        print_table("Diagnostics", rows, ["optimizer", "master_params", "autocast", "tied_rms", "tied_max_abs", "matrix_rms_mean", "matrix_wd", "aux_wd", "rho_output"])

    if args.csv_out and all_rows:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote CSV: {out}")


if __name__ == "__main__":
    main()
