from __future__ import annotations

import argparse
import csv
import json
import math
from statistics import mean, stdev
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    skipped = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            skipped += 1
            print(f"warning: skipped invalid JSON {path}:{i}: {exc}")
    if skipped:
        print(f"warning: skipped {skipped} invalid JSON line(s) in {path}")
    return rows


def last_phase(rows: list[dict], phase: str) -> dict:
    vals = [r for r in rows if r.get("phase") == phase]
    return vals[-1] if vals else {}


def method_name(arm: str) -> str:
    return {"muon": "Hybrid Muon", "sign": "SCION-style Sign", "afmoun": "AF-Muon"}[arm]


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.4e}"
    return str(v)


def mean_sd(vals: list[float]) -> tuple[float | None, float | None]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def aggregate(out_rows: list[dict]) -> list[dict]:
    metrics = [
        "train_loss",
        "aux_loss",
        "eval_loss",
        "eval_ppl",
        "tied_rms",
        "tied_max",
        "matrix_rms",
        "router_entropy",
        "router_max_frac",
        "router_min_frac",
        "router_cv",
        "unused_experts",
    ]
    order = {"muon": 0, "sign": 1, "afmoun": 2}
    agg_rows = []
    for arm in sorted({r["arm"] for r in out_rows}, key=lambda x: order.get(x, 9)):
        rows = [r for r in out_rows if r["arm"] == arm and r.get("eval_loss") is not None]
        row = {
            "method": method_name(arm),
            "arm": arm,
            "n": len(rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda x: x["seed"])),
            "tokens_M": rows[0].get("tokens_M") if rows else None,
            "params": rows[0].get("params") if rows else None,
            "experts": rows[0].get("experts") if rows else None,
            "top_k": rows[0].get("top_k") if rows else None,
            "mbs": rows[0].get("mbs") if rows else None,
            "ga": rows[0].get("ga") if rows else None,
        }
        for metric in metrics:
            mu, sd = mean_sd([r.get(metric) for r in rows])
            row[f"{metric}_mean"] = mu
            row[f"{metric}_sd"] = sd
        agg_rows.append(row)
    return agg_rows


def paired_gaps(out_rows: list[dict]) -> list[dict]:
    by_seed = {}
    for row in out_rows:
        if row.get("eval_loss") is not None:
            by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    pairs = [("afmoun", "sign", "AF-Muon minus SCION-style Sign"), ("afmoun", "muon", "AF-Muon minus Hybrid Muon")]
    gap_rows = []
    for left, right, label in pairs:
        loss_vals = []
        ppl_vals = []
        train_vals = []
        seeds = []
        for seed, arms in sorted(by_seed.items()):
            if left in arms and right in arms:
                seeds.append(seed)
                loss_vals.append(arms[left]["eval_loss"] - arms[right]["eval_loss"])
                ppl_vals.append(arms[left]["eval_ppl"] - arms[right]["eval_ppl"])
                train_vals.append(arms[left]["train_loss"] - arms[right]["train_loss"])
        loss_mu, loss_sd = mean_sd(loss_vals)
        ppl_mu, ppl_sd = mean_sd(ppl_vals)
        train_mu, train_sd = mean_sd(train_vals)
        gap_rows.append({
            "contrast": label,
            "n": len(seeds),
            "seeds": ",".join(str(s) for s in seeds),
            "train_loss_gap_mean": train_mu,
            "train_loss_gap_sd": train_sd,
            "eval_loss_gap_mean": loss_mu,
            "eval_loss_gap_sd": loss_sd,
            "eval_ppl_gap_mean": ppl_mu,
            "eval_ppl_gap_sd": ppl_sd,
        })
    return gap_rows


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c) for c in cols})


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize tied-embedding NanoGPT-MoE runs.")
    p.add_argument("--run-root", required=True)
    p.add_argument("--csv", default="")
    p.add_argument("--aggregate-csv", default="")
    p.add_argument("--gaps-csv", default="")
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
        arm = cfg["arm"]
        out_rows.append({
            "run": run.name,
            "method": method_name(arm),
            "arm": arm,
            "seed": cfg["seed"],
            "step": ev.get("step", tr.get("step")),
            "tokens_M": ev.get("tokens_seen", tr.get("tokens_seen", 0)) / 1e6,
            "eval_loss": ev.get("eval_loss"),
            "eval_ppl": ev.get("eval_ppl"),
            "train_loss": tr.get("lm_loss", tr.get("loss")),
            "aux_loss": tr.get("aux_loss"),
            "grad_norm": tr.get("grad_norm_preclip"),
            "tied_rms": dg.get("tied_rms"),
            "tied_max": dg.get("tied_max_abs"),
            "router_entropy": dg.get("moe_router_entropy_mean"),
            "router_max_frac": dg.get("moe_router_max_frac_mean"),
            "router_min_frac": dg.get("moe_router_min_frac_mean"),
            "router_cv": dg.get("moe_router_cv_mean"),
            "unused_experts": dg.get("moe_router_unused_experts_mean"),
            "params": cfg.get("param_count"),
            "experts": cfg.get("num_experts"),
            "top_k": cfg.get("top_k"),
            "mbs": cfg.get("micro_batch_size"),
            "ga": cfg.get("gradient_accumulation_steps"),
        })

    order = {"muon": 0, "sign": 1, "afmoun": 2}
    out_rows.sort(key=lambda r: (r["seed"], order.get(r["arm"], 9)))
    cols = [
        "method", "seed", "step", "tokens_M", "eval_loss", "eval_ppl", "train_loss",
        "aux_loss", "grad_norm", "tied_rms", "tied_max", "router_entropy",
        "router_max_frac", "router_min_frac", "router_cv", "unused_experts",
        "params", "experts", "top_k", "mbs", "ga", "run",
    ]
    print(" | ".join(cols))
    print(" | ".join(["---"] * len(cols)))
    for r in out_rows:
        print(" | ".join(fmt(r.get(c)) for c in cols))

    csv_path = Path(args.csv) if args.csv else root / "moe_nanogpt_summary.csv"
    write_csv(csv_path, out_rows, cols)
    print(f"\nWrote CSV: {csv_path}")

    agg_rows = aggregate(out_rows)
    agg_cols = [
        "method", "n", "seeds", "tokens_M", "train_loss_mean", "train_loss_sd",
        "aux_loss_mean", "aux_loss_sd", "eval_loss_mean", "eval_loss_sd",
        "eval_ppl_mean", "eval_ppl_sd", "tied_rms_mean", "tied_rms_sd",
        "tied_max_mean", "tied_max_sd", "matrix_rms_mean", "matrix_rms_sd",
        "router_entropy_mean", "router_entropy_sd", "router_max_frac_mean",
        "router_max_frac_sd", "router_min_frac_mean", "router_min_frac_sd",
        "router_cv_mean", "router_cv_sd", "unused_experts_mean",
        "unused_experts_sd", "params", "experts", "top_k", "mbs", "ga",
    ]
    print("\n=== Aggregate over complete seeds ===")
    print(" | ".join(agg_cols))
    print(" | ".join(["---"] * len(agg_cols)))
    for row in agg_rows:
        print(" | ".join(fmt(row.get(c)) for c in agg_cols))
    agg_path = Path(args.aggregate_csv) if args.aggregate_csv else root / "moe_nanogpt_aggregate.csv"
    write_csv(agg_path, agg_rows, agg_cols)
    print(f"\nWrote aggregate CSV: {agg_path}")

    gap_rows = paired_gaps(out_rows)
    gap_cols = [
        "contrast", "n", "seeds", "train_loss_gap_mean", "train_loss_gap_sd",
        "eval_loss_gap_mean", "eval_loss_gap_sd", "eval_ppl_gap_mean",
        "eval_ppl_gap_sd",
    ]
    print("\n=== Paired gaps ===")
    print(" | ".join(gap_cols))
    print(" | ".join(["---"] * len(gap_cols)))
    for row in gap_rows:
        print(" | ".join(fmt(row.get(c)) for c in gap_cols))
    gap_path = Path(args.gaps_csv) if args.gaps_csv else root / "moe_nanogpt_paired_gaps.csv"
    write_csv(gap_path, gap_rows, gap_cols)
    print(f"\nWrote paired-gap CSV: {gap_path}")


if __name__ == "__main__":
    main()
