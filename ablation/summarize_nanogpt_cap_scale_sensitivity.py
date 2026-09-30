from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


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


def best_eval(rows: list[dict]) -> dict:
    vals = [row for row in rows if row.get("phase") == "eval" and row.get("step", 0) > 0]
    return min(vals, key=lambda row: row.get("eval_loss", float("inf"))) if vals else {}


def parse_cap_from_name(name: str) -> float | None:
    match = re.search(r"_cap([^_]+)_scale", name)
    if not match:
        return None
    text = match.group(1).replace("p", ".")
    return float("inf") if text == "inf" else float(text)


def parse_scale_from_name(name: str) -> float | None:
    match = re.search(r"_scale([^_]+)_500m", name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def fmt_value(value) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NanoGPT AF-Muon cap/scale sensitivity runs.")
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
        tr = last_phase(metrics, "train")
        ev = last_phase(metrics, "eval")
        dg = last_phase(metrics, "diagnostic")
        be = best_eval(metrics)
        cap = cfg.get("arm_settings", {}).get("tied_cap", cfg.get("tied_cap"))
        scale = cfg.get("arm_settings", {}).get("tied_scale", cfg.get("tied_scale"))
        if cap is None:
            cap = parse_cap_from_name(run_dir.name)
        if scale is None:
            scale = parse_scale_from_name(run_dir.name)
        cap = float(cap)
        scale = float(scale)
        rows_out.append(
            {
                "method": "AF-Muon",
                "seed": cfg.get("seed"),
                "tied_cap": cap,
                "tied_scale": scale,
                "complete": int(tr.get("step", 0) or 0) >= int(cfg.get("iterations", 0) or 0),
                "step": ev.get("step", tr.get("step")),
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
                "coords_capped": dg.get("coords_capped_frac"),
                "rows_clipped": dg.get("rows_clipped_frac"),
                "tail_energy": dg.get("raw_tail_energy_frac"),
                "cos_af_sign": dg.get("cos_af_sign"),
                "obj_af_sign": dg.get("objective_af_over_sign"),
                "run": run_dir.name,
            }
        )

    rows_out.sort(key=lambda r: (float(r["tied_scale"]), float(r["tied_cap"])))
    cols = [
        "method",
        "seed",
        "tied_cap",
        "tied_scale",
        "complete",
        "step",
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
        "coords_capped",
        "rows_clipped",
        "tail_energy",
        "cos_af_sign",
        "obj_af_sign",
        "run",
    ]

    print(" | ".join(cols))
    print(" | ".join(["---"] * len(cols)))
    for row in rows_out:
        print(" | ".join(fmt_value(row.get(col)) for col in cols))

    default = next(
        (
            row for row in rows_out
            if abs(float(row["tied_scale"]) - 0.5) < 1e-12
            and abs(float(row["tied_cap"]) - 3.0) < 1e-12
        ),
        None,
    )
    if default is not None:
        print()
        print("=== Gaps relative to default AF-Muon c=3, s=0.5 ===")
        for row in rows_out:
            print(
                f"cap={fmt_value(row['tied_cap'])} scale={fmt_value(row['tied_scale'])}: "
                f"eval_loss-default={row['eval_loss'] - default['eval_loss']:+.6f}, "
                f"eval_ppl-default={row['eval_ppl'] - default['eval_ppl']:+.6f}"
            )

    csv_path = args.csv if args.csv is not None else args.run_root / "nanogpt_afmoun_cap_scale_sensitivity_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({col: row.get(col) for col in cols})
    print(f"\nWrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
