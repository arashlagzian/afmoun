import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev


RUN_ROOT = Path("runs/01_paper_main_results_smollm2_135m_2p5b_v2_fp32master_matchbatch524k")
OUT_DIR = Path("outputs/smollm2_135m_2p5b_fp32master_matchbatch524k_verified")
OUT_DIR.mkdir(parents=True, exist_ok=True)

bad_lines = []


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines.append({
                    "file": str(path),
                    "line": lineno,
                    "error": str(e),
                    "text": line[:500],
                })
    return rows


def method_from_run(name):
    if "afmoun" in name:
        return "AF-Muon"
    if "scion" in name or "sign" in name:
        return "SCION-style Sign"
    if "hybrid" in name or "_muon_" in name:
        return "Hybrid Muon"
    return name


def seed_from_run(name):
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def fmt(x, nd=4):
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


raw = []
for run_dir in sorted(p for p in RUN_ROOT.iterdir() if p.is_dir() and p.name != "_logs"):
    cfg_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not cfg_path.exists() or not metrics_path.exists():
        continue

    cfg = json.loads(cfg_path.read_text())
    rows = read_jsonl(metrics_path)
    trains = [r for r in rows if r.get("phase") == "train"]
    evals = [r for r in rows if r.get("phase") == "eval"]
    diags = [r for r in rows if r.get("phase") == "diagnostic"]

    last_train = trains[-1] if trains else {}
    last_eval = evals[-1] if evals else {}
    best_eval = min(evals, key=lambda r: r.get("eval_loss", float("inf"))) if evals else {}
    last_diag = diags[-1] if diags else {}

    target_tokens = int(cfg.get("actual_train_tokens") or cfg.get("train_tokens") or 0)
    tokens_seen = int(last_train.get("tokens_seen") or last_eval.get("tokens_seen") or 0)
    complete = tokens_seen >= 0.995 * target_tokens if target_tokens else False

    raw.append({
        "method": method_from_run(run_dir.name),
        "seed": seed_from_run(run_dir.name),
        "run": run_dir.name,
        "complete": complete,
        "step": last_train.get("step") or last_eval.get("step"),
        "tokens_B": tokens_seen / 1e9,
        "target_B": target_tokens / 1e9 if target_tokens else None,
        "train_loss": last_train.get("loss"),
        "eval_loss": last_eval.get("eval_loss"),
        "eval_ppl": last_eval.get("eval_ppl"),
        "best_loss": best_eval.get("eval_loss"),
        "best_ppl": best_eval.get("eval_ppl"),
        "best_step": best_eval.get("step"),
        "grad_norm": last_train.get("grad_norm_preclip") or last_diag.get("grad_norm_preclip"),
        "tied_rms": last_diag.get("tied_rms"),
        "tied_max": last_diag.get("tied_max_abs"),
        "matrix_rms": last_diag.get("matrix_rms_mean"),
        "params": cfg.get("param_count"),
        "fp32_params": cfg.get("fp32_params"),
        "master_params": cfg.get("master_params"),
        "autocast_only": cfg.get("autoc_only") or cfg.get("autocast_only"),
        "bf16": cfg.get("bf16"),
        "matrix_wd": cfg.get("matrix_weight_decay"),
        "aux_wd": cfg.get("aux_weight_decay"),
        "mbs": cfg.get("micro_batch_size"),
        "ga": cfg.get("gradient_accumulation_steps"),
        "tokens_per_step": cfg.get("tokens_per_step"),
    })

raw = sorted(raw, key=lambda r: (r["seed"] if r["seed"] is not None else 999, r["method"]))

if not raw:
    raise SystemExit("No runs found.")

raw_csv = OUT_DIR / "smollm2_135m_2p5b_per_seed_raw.csv"
with raw_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
    w.writeheader()
    w.writerows(raw)

if bad_lines:
    bad_csv = OUT_DIR / "smollm2_135m_2p5b_bad_jsonl_lines.csv"
    with bad_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "line", "error", "text"])
        w.writeheader()
        w.writerows(bad_lines)
else:
    bad_csv = None

methods = ["Hybrid Muon", "SCION-style Sign", "AF-Muon"]
summary = []
for method in methods:
    rows = [r for r in raw if r["method"] == method and r["complete"]]
    if not rows:
        continue

    def msd(key):
        vals = [float(r[key]) for r in rows if r[key] is not None]
        return mean(vals), stdev(vals) if len(vals) > 1 else 0.0

    train_m, train_s = msd("train_loss")
    eval_m, eval_s = msd("eval_loss")
    ppl_m, ppl_s = msd("eval_ppl")
    best_m, best_s = msd("best_loss")
    summary.append({
        "method": method,
        "n": len(rows),
        "seeds": ",".join(str(r["seed"]) for r in rows),
        "params": rows[0]["params"],
        "train_loss_mean": train_m,
        "train_loss_sd": train_s,
        "eval_loss_mean": eval_m,
        "eval_loss_sd": eval_s,
        "eval_ppl_mean": ppl_m,
        "eval_ppl_sd": ppl_s,
        "best_loss_mean": best_m,
        "best_loss_sd": best_s,
        "tokens_B": mean([r["tokens_B"] for r in rows]),
        "mbs": rows[0]["mbs"],
        "ga": rows[0]["ga"],
        "tokens_per_step": rows[0]["tokens_per_step"],
        "matrix_wd": rows[0]["matrix_wd"],
        "aux_wd": rows[0]["aux_wd"],
    })

summary_csv = OUT_DIR / "smollm2_135m_2p5b_summary.csv"
with summary_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

complete_seeds = sorted(set(r["seed"] for r in raw if r["complete"]))
by_seed = {(r["seed"], r["method"]): r for r in raw if r["complete"]}
gap_rows = []
for seed in complete_seeds:
    for label, a_name, b_name in [
        ("AF-Muon minus SCION-style Sign", "AF-Muon", "SCION-style Sign"),
        ("AF-Muon minus Hybrid Muon", "AF-Muon", "Hybrid Muon"),
        ("SCION-style Sign minus Hybrid Muon", "SCION-style Sign", "Hybrid Muon"),
    ]:
        if (seed, a_name) in by_seed and (seed, b_name) in by_seed:
            a = by_seed[(seed, a_name)]
            b = by_seed[(seed, b_name)]
            gap_rows.append({
                "comparison": label,
                "seed": seed,
                "eval_loss_gap": a["eval_loss"] - b["eval_loss"],
                "eval_ppl_gap": a["eval_ppl"] - b["eval_ppl"],
                "train_loss_gap": a["train_loss"] - b["train_loss"],
                "best_loss_gap": a["best_loss"] - b["best_loss"],
            })

gap_csv = OUT_DIR / "smollm2_135m_2p5b_paired_gaps.csv"
with gap_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gap_rows[0].keys()))
    w.writeheader()
    w.writerows(gap_rows)

print("=== Per-run status ===")
print("| method | seed | complete | step | tokens_B | eval_loss | eval_ppl | best_loss | train_loss | aux_wd | run |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for r in raw:
    print(f"| {r['method']} | {r['seed']} | {r['complete']} | {r['step']} | {fmt(r['tokens_B'],3)} | {fmt(r['eval_loss'])} | {fmt(r['eval_ppl'],2)} | {fmt(r['best_loss'])} | {fmt(r['train_loss'])} | {r['aux_wd']} | {r['run']} |")

print()
print("=== Aggregate over complete seeds ===")
print("| method | n | seeds | train_loss | eval_loss | eval_ppl | best_loss |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for r in summary:
    print(f"| {r['method']} | {r['n']} | {r['seeds']} | {r['train_loss_mean']:.4f} +/- {r['train_loss_sd']:.4f} | {r['eval_loss_mean']:.4f} +/- {r['eval_loss_sd']:.4f} | {r['eval_ppl_mean']:.2f} +/- {r['eval_ppl_sd']:.2f} | {r['best_loss_mean']:.4f} +/- {r['best_loss_sd']:.4f} |")

print()
print("=== Paired gaps ===")
for comp in sorted(set(g["comparison"] for g in gap_rows)):
    rows = [g for g in gap_rows if g["comparison"] == comp]
    loss_vals = [g["eval_loss_gap"] for g in rows]
    ppl_vals = [g["eval_ppl_gap"] for g in rows]
    print(f"{comp}: eval_loss_gap={mean(loss_vals):+.6f} +/- {stdev(loss_vals):.6f}, eval_ppl_gap={mean(ppl_vals):+.4f} +/- {stdev(ppl_vals):.4f}")

print()
if bad_csv is not None:
    print(f"WARNING: skipped {len(bad_lines)} malformed JSONL lines.")
    print("Bad-line report:", bad_csv)

print("Wrote:")
print(raw_csv)
print(summary_csv)
print(gap_csv)
