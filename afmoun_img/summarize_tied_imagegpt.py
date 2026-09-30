from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"warning: skipped invalid JSON {path}:{i}: {exc}")
    return rows


def last_phase(rows: list[dict], phase: str) -> dict:
    vals = [r for r in rows if r.get("phase") == phase]
    return vals[-1] if vals else {}


def best_eval(rows: list[dict]) -> dict:
    vals = [r for r in rows if r.get("phase") == "eval" and r.get("step", 0) > 0]
    return min(vals, key=lambda r: r.get("eval_loss", float("inf"))) if vals else {}


def method_name(arm: str) -> str:
    return {"muon": "Hybrid Muon", "sign": "SCION-style Sign", "afmoun": "AF-Muon"}[arm]


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize tied ImageGPT RGB554 runs.")
    p.add_argument("--run-root", required=True)
    p.add_argument("--csv", default="")
    args = p.parse_args()

    root = Path(args.run_root)
    out_rows = []
    for run in sorted(root.glob("*")):
        if not run.is_dir() or run.name == "_logs":
            continue
        cfg_path = run / "config.json"
        metrics_path = run / "metrics.jsonl"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        rows = read_jsonl(metrics_path)
        ev = last_phase(rows, "eval")
        tr = last_phase(rows, "train")
        dg = last_phase(rows, "diagnostic")
        be = best_eval(rows)
        arm = cfg["arm"]
        out_rows.append({
            "method": method_name(arm),
            "arm": arm,
            "seed": cfg["seed"],
            "step": ev.get("step", tr.get("step")),
            "tokens_M": ev.get("tokens_seen", tr.get("tokens_seen", 0)) / 1e6,
            "train_loss": tr.get("loss"),
            "train_bits_per_dim": tr.get("bits_per_dim"),
            "eval_loss": ev.get("eval_loss"),
            "eval_ppl": ev.get("eval_ppl"),
            "eval_bits_per_token": ev.get("eval_bits_per_token"),
            "eval_bits_per_dim": ev.get("eval_bits_per_dim"),
            "best_loss": be.get("eval_loss"),
            "best_bits_per_dim": be.get("eval_bits_per_dim"),
            "best_step": be.get("step"),
            "grad_norm": tr.get("grad_norm_preclip"),
            "tied_rms": dg.get("tied_rms"),
            "tied_max": dg.get("tied_max_abs"),
            "color_tied_rms": dg.get("color_tied_rms"),
            "sos_row_rms": dg.get("sos_row_rms"),
            "matrix_rms": dg.get("matrix_rms_mean"),
            "rows_clip": dg.get("rows_clipped_frac"),
            "coords_capped": dg.get("coords_capped_frac"),
            "tail_energy": dg.get("raw_tail_energy_frac"),
            "cos_af_sign": dg.get("cos_af_sign"),
            "obj_af_sign": dg.get("objective_af_over_sign"),
            "params": cfg.get("param_count"),
            "batch_size": cfg.get("batch_size"),
            "mbs": cfg.get("micro_batch_size"),
            "ga": cfg.get("gradient_accumulation_steps"),
            "run": run.name,
        })

    order = {"muon": 0, "sign": 1, "afmoun": 2}
    out_rows.sort(key=lambda r: (r["seed"], order.get(r["arm"], 9)))
    cols = [
        "method", "seed", "step", "tokens_M", "train_loss", "train_bits_per_dim",
        "eval_loss", "eval_ppl", "eval_bits_per_token", "eval_bits_per_dim",
        "best_loss", "best_bits_per_dim", "best_step", "grad_norm", "tied_rms",
        "tied_max", "color_tied_rms", "sos_row_rms", "matrix_rms", "rows_clip",
        "coords_capped", "tail_energy", "cos_af_sign", "obj_af_sign", "params",
        "batch_size", "mbs", "ga", "run",
    ]
    print(" | ".join(cols))
    print(" | ".join(["---"] * len(cols)))
    for r in out_rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if abs(v) < 1e4 else f"{v:.4e}")
            else:
                vals.append(str(v))
        print(" | ".join(vals))

    by_seed = {}
    for row in out_rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    print()
    for seed, arms in sorted(by_seed.items()):
        if "afmoun" in arms and "sign" in arms:
            print(
                f"seed {seed} AF-Muon minus SCION: "
                f"loss_gap={arms['afmoun']['eval_loss'] - arms['sign']['eval_loss']:+.6f}, "
                f"bpd_gap={arms['afmoun']['eval_bits_per_dim'] - arms['sign']['eval_bits_per_dim']:+.6f}"
            )
        if "afmoun" in arms and "muon" in arms:
            print(
                f"seed {seed} AF-Muon minus Hybrid: "
                f"loss_gap={arms['afmoun']['eval_loss'] - arms['muon']['eval_loss']:+.6f}, "
                f"bpd_gap={arms['afmoun']['eval_bits_per_dim'] - arms['muon']['eval_bits_per_dim']:+.6f}"
            )

    csv_path = Path(args.csv) if args.csv else root / "tied_imagegpt_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({c: row.get(c) for c in cols})
    print(f"\nWrote CSV: {csv_path}")


if __name__ == "__main__":
    main()

