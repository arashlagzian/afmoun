from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SOURCE_NAMES = {
    "input_only": "Sparse input only",
    "output_only": "Dense output only",
    "both": "Sparse input + dense output",
}
SOURCE_ORDER = {"input_only": 0, "output_only": 1, "both": 2}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"warning: skipped invalid JSON {path}:{line_number}: {exc}")
    return rows


def last_phase(rows: list[dict], phase: str) -> dict:
    vals = [row for row in rows if row.get("phase") == phase]
    return vals[-1] if vals else {}


def phase_at_or_before_step(rows: list[dict], phase: str, step: int | None) -> dict:
    vals = [row for row in rows if row.get("phase") == phase]
    if step is not None:
        vals = [row for row in vals if int(row.get("step", -1) or -1) <= int(step)]
    return vals[-1] if vals else {}


def best_eval(rows: list[dict], max_step: int | None = None) -> dict:
    vals = [row for row in rows if row.get("phase") == "eval" and row.get("step", 0) > 0]
    if max_step is not None:
        vals = [row for row in vals if int(row.get("step", -1) or -1) <= int(max_step)]
    return min(vals, key=lambda row: row.get("eval_loss", float("inf"))) if vals else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AF-Muon tied source ablation runs.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--csv", default=None, type=Path)
    args = parser.parse_args()

    rows_out = []
    for run_dir in sorted(args.run_root.glob("*")):
        if not run_dir.is_dir() or run_dir.name == "_logs":
            continue
        cfg_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        metrics = read_jsonl(metrics_path)
        ev = last_phase(metrics, "eval")
        eval_step = int(ev.get("step", 0) or 0) if ev else None
        tr = phase_at_or_before_step(metrics, "train", eval_step)
        dg = last_phase(metrics, "diagnostic")
        be = best_eval(metrics, max_step=eval_step)
        source = cfg.get("tied_grad_source", "both")
        rows_out.append(
            {
                "source": source,
                "source_name": SOURCE_NAMES.get(source, source),
                "seed": cfg.get("seed"),
                "complete": int(tr.get("step", 0) or 0) >= int(cfg.get("iterations", 0) or 0),
                "step": ev.get("step", tr.get("step")),
                "train_step": tr.get("step"),
                "eval_step": ev.get("step"),
                "tokens_M": ev.get("tokens_seen", tr.get("tokens_seen", 0)) / 1e6,
                "train_loss": tr.get("loss"),
                "eval_loss": ev.get("eval_loss"),
                "eval_ppl": ev.get("eval_ppl"),
                "best_loss": be.get("eval_loss"),
                "best_ppl": be.get("eval_ppl"),
                "best_step": be.get("step"),
                "grad_norm": tr.get("grad_norm_preclip"),
                "tied_rms": dg.get("tied_rms"),
                "tied_max": dg.get("tied_max_abs"),
                "matrix_rms": dg.get("matrix_rms_mean"),
                "input_row_frac": dg.get("tied_input_nonzero_rows_frac"),
                "output_row_frac": dg.get("tied_output_nonzero_rows_frac"),
                "total_row_frac": dg.get("tied_total_nonzero_rows_frac"),
                "both_row_frac": dg.get("tied_rows_hit_both_frac"),
                "input_grad_rms": dg.get("tied_input_grad_rms"),
                "output_grad_rms": dg.get("tied_output_grad_rms"),
                "total_grad_rms": dg.get("tied_total_grad_rms"),
                "input_norm_frac": dg.get("tied_input_norm_frac"),
                "output_norm_frac": dg.get("tied_output_norm_frac"),
                "input_output_cos": dg.get("tied_input_output_cos"),
                "input_total_cos": dg.get("tied_input_total_cos"),
                "output_total_cos": dg.get("tied_output_total_cos"),
                "aux_wd": cfg.get("aux_weight_decay"),
                "run": run_dir.name,
            }
        )

    rows_out.sort(key=lambda r: SOURCE_ORDER.get(r["source"], 99))
    cols = [
        "source",
        "source_name",
        "seed",
        "complete",
        "step",
        "train_step",
        "eval_step",
        "tokens_M",
        "train_loss",
        "eval_loss",
        "eval_ppl",
        "best_loss",
        "best_ppl",
        "best_step",
        "grad_norm",
        "tied_rms",
        "tied_max",
        "matrix_rms",
        "input_row_frac",
        "output_row_frac",
        "total_row_frac",
        "both_row_frac",
        "input_grad_rms",
        "output_grad_rms",
        "total_grad_rms",
        "input_norm_frac",
        "output_norm_frac",
        "input_output_cos",
        "input_total_cos",
        "output_total_cos",
        "aux_wd",
        "run",
    ]

    print(" | ".join(cols))
    print(" | ".join(["---"] * len(cols)))
    for row in rows_out:
        values = []
        for col in cols:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        print(" | ".join(values))

    if "both" in {row["source"] for row in rows_out}:
        baseline = next(row for row in rows_out if row["source"] == "both")
        print()
        print("=== Sparse+dense AF-Muon gaps ===")
        for row in rows_out:
            if row["source"] == "both":
                continue
            print(
                f"both - {row['source_name']}: "
                f"eval_loss={baseline['eval_loss'] - row['eval_loss']:+.6f}, "
                f"eval_ppl={baseline['eval_ppl'] - row['eval_ppl']:+.6f}"
            )

    csv_path = args.csv if args.csv is not None else args.run_root / "nanogpt_tied_source_ablation_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({col: row.get(col) for col in cols})
    print(f"\nWrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
