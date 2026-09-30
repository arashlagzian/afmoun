from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ARM_ORDER = {"muon": 0, "sign": 1, "afmoun": 2}
METHOD = {
    "muon": "Hybrid Muon",
    "sign": "SCION-style Sign",
    "afmoun": "AF-Muon",
}


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


def phase_at_or_before_tokens(rows: list[dict], phase: str, target_tokens: int | None) -> dict:
    vals = [row for row in rows if row.get("phase") == phase]
    if target_tokens is None:
        return vals[-1] if vals else {}
    vals = [
        row for row in vals
        if int(row.get("tokens_seen", -1) or -1) <= int(target_tokens)
    ]
    return max(vals, key=lambda row: int(row.get("tokens_seen", -1) or -1)) if vals else {}


def phase_at_exact_tokens(rows: list[dict], phase: str, target_tokens: int | None) -> dict:
    if target_tokens is None:
        return phase_at_or_before_tokens(rows, phase, None)
    vals = [
        row for row in rows
        if row.get("phase") == phase
        and int(row.get("tokens_seen", -1) or -1) == int(target_tokens)
    ]
    return vals[-1] if vals else {}


def best_eval_at_or_before_tokens(rows: list[dict], target_tokens: int | None) -> dict:
    vals = [row for row in rows if row.get("phase") == "eval" and row.get("step", 0) > 0]
    if target_tokens is not None:
        vals = [
            row for row in vals
            if int(row.get("tokens_seen", -1) or -1) <= int(target_tokens)
        ]
    return min(vals, key=lambda row: row.get("eval_loss", float("inf"))) if vals else {}


def infer_batch_size(run_name: str, cfg: dict) -> int | None:
    if "batch_size" in cfg:
        return int(cfg["batch_size"])
    match = re.search(r"batch(\d+)", run_name)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NanoGPT batch-size sensitivity runs.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--csv", default=None, type=Path)
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="If set, summarize rows at this exact token budget unless --allow-earlier-target is used.",
    )
    parser.add_argument(
        "--allow-earlier-target",
        action="store_true",
        help="Permit selecting the latest row at or before --target-tokens instead of requiring an exact match.",
    )
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
        selector = phase_at_or_before_tokens if args.allow_earlier_target else phase_at_exact_tokens
        ev = selector(metrics, "eval", args.target_tokens)
        if args.target_tokens is not None and not ev:
            raise ValueError(f"{run_dir} has no eval row at exactly {args.target_tokens} tokens")
        eval_tokens = int(ev.get("tokens_seen", args.target_tokens or -1) or -1) if ev else args.target_tokens
        tr = phase_at_or_before_tokens(metrics, "train", eval_tokens)
        dg = selector(metrics, "diagnostic", args.target_tokens)
        be = best_eval_at_or_before_tokens(metrics, args.target_tokens)
        arm = cfg["arm"]
        batch_size = infer_batch_size(run_dir.name, cfg)
        block_size = int(cfg.get("block_size", 1024))
        mbs = int(cfg.get("micro_batch_size", 16))
        tokens_per_step = batch_size * block_size if batch_size else None
        rows_out.append(
            {
                "method": METHOD.get(arm, arm),
                "arm": arm,
                "seed": cfg.get("seed"),
                "batch_size": batch_size,
                "micro_batch_size": mbs,
                "gradient_accumulation_steps": batch_size // mbs if batch_size else None,
                "tokens_per_step": tokens_per_step,
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
                "matrix_wd": cfg.get("matrix_weight_decay"),
                "aux_wd": cfg.get("aux_weight_decay"),
                "run": run_dir.name,
            }
        )

    rows_out.sort(key=lambda r: (int(r["batch_size"] or 10**18), ARM_ORDER.get(r["arm"], 9)))
    cols = [
        "method",
        "arm",
        "seed",
        "batch_size",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "tokens_per_step",
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
        "matrix_wd",
        "aux_wd",
        "run",
    ]

    if args.target_tokens is not None:
        print(f"=== token-matched summary at or before {args.target_tokens} tokens ===")
    else:
        print("=== endpoint summary ===")
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

    print()
    print("=== AF-Muon gaps by batch size ===")
    by_batch = {}
    for row in rows_out:
        by_batch.setdefault(row["batch_size"], {})[row["arm"]] = row
    for batch_size, arms in sorted(by_batch.items()):
        if "afmoun" not in arms:
            continue
        parts = [
            f"batch_size={batch_size}",
            f"tokens/update={int(arms['afmoun']['tokens_per_step'])}",
        ]
        for other in ("muon", "sign"):
            if other in arms:
                parts.append(
                    f"AF-{METHOD[other]} eval_loss={arms['afmoun']['eval_loss'] - arms[other]['eval_loss']:+.6f}"
                )
        print(" | ".join(parts))

    csv_path = args.csv if args.csv is not None else args.run_root / "nanogpt_batch_sensitivity_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({col: row.get(col) for col in cols})
    print(f"\nWrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
