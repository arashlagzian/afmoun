from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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


def phase_at_step(rows: list[dict], phase: str, step: int | None) -> dict:
    if step is None:
        return last_phase(rows, phase)
    vals = [row for row in rows if row.get("phase") == phase and int(row.get("step", -1) or -1) == step]
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


def cap_sort_value(value: float) -> float:
    return 1e30 if math.isinf(float(value)) else float(value)


def fmt_cap(value: float) -> str:
    return r"$\infty$" if math.isinf(float(value)) else f"{float(value):g}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def fmt_mean_sd(values: list[float], digits: int = 4) -> str:
    return f"{mean(values):.{digits}f} $\\pm$ {sample_sd(values):.{digits}f}"


def iter_run_dirs(root: Path):
    for metrics_path in sorted(root.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        if (run_dir / "config.json").exists():
            yield run_dir


def collect(root: Path, eval_step: int | None = None) -> list[dict]:
    rows_out = []
    for run_dir in iter_run_dirs(root):
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        tr = phase_at_or_before_step(metrics, "train", eval_step)
        ev = phase_at_step(metrics, "eval", eval_step)
        dg = phase_at_step(metrics, "diagnostic", eval_step)
        be = best_eval(metrics, max_step=eval_step)
        if not ev:
            continue

        cap = cfg.get("arm_settings", {}).get("tied_cap", cfg.get("tied_cap"))
        scale = cfg.get("arm_settings", {}).get("tied_scale", cfg.get("tied_scale"))
        seed = int(cfg.get("seed"))
        iterations = int(cfg.get("iterations", 0) or 0)
        train_step = int(tr.get("step", 0) or 0)
        observed_eval_step = int(ev.get("step", 0) or 0)
        target_step = int(eval_step if eval_step is not None else observed_eval_step)
        reached_comparison = int(observed_eval_step >= target_step)
        finished_training = int(
            train_step >= iterations
            or any(row.get("phase") == "checkpoint" and int(row.get("step", 0) or 0) >= iterations for row in metrics)
        )

        rows_out.append(
            {
                "seed": seed,
                "tied_cap": float(cap),
                "tied_scale": float(scale),
                "complete": reached_comparison,
                "reached_comparison": reached_comparison,
                "finished_training": finished_training,
                "step": ev.get("step", tr.get("step")),
                "tokens_M": (ev.get("tokens_seen", tr.get("tokens_seen", 0)) or 0) / 1e6,
                "train_step": tr.get("step"),
                "eval_step": ev.get("step"),
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
                "run_dir": str(run_dir),
            }
        )

    rows_out.sort(key=lambda r: (r["seed"], float(r["tied_scale"]), cap_sort_value(r["tied_cap"])))
    return rows_out


def aggregate(raw_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["tied_cap"], row["tied_scale"])].append(row)

    out = []
    for (cap, scale), rows in sorted(grouped.items(), key=lambda kv: (kv[0][1], cap_sort_value(kv[0][0]))):
        seeds = sorted({int(r["seed"]) for r in rows})
        if len(seeds) != len(rows):
            duplicate = sorted(int(r["seed"]) for r in rows)
            raise ValueError(f"duplicate runs for cap={cap}, scale={scale}: seeds={duplicate}")
        out.append(
            {
                "tied_cap": cap,
                "tied_scale": scale,
                "n": len(seeds),
                "seeds": ",".join(map(str, seeds)),
                "complete": int(all(int(r["complete"]) for r in rows)),
                "finished_training": int(all(int(r["finished_training"]) for r in rows)),
                "train_loss_mean": mean([float(r["train_loss"]) for r in rows]),
                "train_loss_sd": sample_sd([float(r["train_loss"]) for r in rows]),
                "eval_loss_mean": mean([float(r["eval_loss"]) for r in rows]),
                "eval_loss_sd": sample_sd([float(r["eval_loss"]) for r in rows]),
                "eval_ppl_mean": mean([float(r["eval_ppl"]) for r in rows]),
                "eval_ppl_sd": sample_sd([float(r["eval_ppl"]) for r in rows]),
                "tied_rms_mean": mean([float(r["tied_rms"]) for r in rows]),
                "tied_rms_sd": sample_sd([float(r["tied_rms"]) for r in rows]),
                "tied_max_mean": mean([float(r["tied_max"]) for r in rows]),
                "tied_max_sd": sample_sd([float(r["tied_max"]) for r in rows]),
                "coords_capped_mean": mean([float(r["coords_capped"]) for r in rows]),
                "coords_capped_sd": sample_sd([float(r["coords_capped"]) for r in rows]),
                "rows_clipped_mean": mean([float(r["rows_clipped"]) for r in rows]),
                "rows_clipped_sd": sample_sd([float(r["rows_clipped"]) for r in rows]),
                "tail_energy_mean": mean([float(r["tail_energy"]) for r in rows]),
                "tail_energy_sd": sample_sd([float(r["tail_energy"]) for r in rows]),
                "cos_af_sign_mean": mean([float(r["cos_af_sign"]) for r in rows]),
                "cos_af_sign_sd": sample_sd([float(r["cos_af_sign"]) for r in rows]),
                "obj_af_sign_mean": mean([float(r["obj_af_sign"]) for r in rows]),
                "obj_af_sign_sd": sample_sd([float(r["obj_af_sign"]) for r in rows]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def print_latex_tables(agg_rows: list[dict]) -> None:
    print("\n=== LaTeX summary table ===")
    print(r"\begin{tabular}{cccccc}")
    print(r"\toprule")
    print(r"Cap $c$ & Scale $s$ & Train loss & Val. loss & Val. PPL & Tied RMS \\")
    print(r"\midrule")
    for row in agg_rows:
        print(
            f"{fmt_cap(row['tied_cap'])} & {row['tied_scale']:g} & "
            f"{row['train_loss_mean']:.4f} $\\pm$ {row['train_loss_sd']:.4f} & "
            f"{row['eval_loss_mean']:.4f} $\\pm$ {row['eval_loss_sd']:.4f} & "
            f"{row['eval_ppl_mean']:.2f} $\\pm$ {row['eval_ppl_sd']:.2f} & "
            f"{row['tied_rms_mean']:.4f} $\\pm$ {row['tied_rms_sd']:.4f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n=== LaTeX diagnostics table ===")
    print(r"\begin{tabular}{cccccc}")
    print(r"\toprule")
    print(r"Cap $c$ & Scale $s$ & Coord. capped & Rows clipped & Tail energy & Obj. ratio \\")
    print(r"\midrule")
    for row in agg_rows:
        print(
            f"{fmt_cap(row['tied_cap'])} & {row['tied_scale']:g} & "
            f"{100 * row['coords_capped_mean']:.2f} $\\pm$ {100 * row['coords_capped_sd']:.2f} & "
            f"{100 * row['rows_clipped_mean']:.2f} $\\pm$ {100 * row['rows_clipped_sd']:.2f} & "
            f"{100 * row['tail_energy_mean']:.2f} $\\pm$ {100 * row['tail_energy_sd']:.2f} & "
            f"{row['obj_af_sign_mean']:.3f} $\\pm$ {row['obj_af_sign_sd']:.3f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multiseed NanoGPT cap/scale sensitivity runs.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--expected-seeds", default="", help="Comma-separated seed list required for each cap/scale cell.")
    parser.add_argument(
        "--eval-step",
        default=None,
        type=int,
        help="Summarize metrics at this exact eval step instead of the final eval.",
    )
    args = parser.parse_args()

    raw_rows = collect(args.run_root, eval_step=args.eval_step)
    if not raw_rows:
        step_msg = f" at step {args.eval_step}" if args.eval_step is not None else ""
        raise SystemExit(f"no runs found under {args.run_root}{step_msg}")
    expected_seeds = sorted(int(x.strip()) for x in args.expected_seeds.split(",") if x.strip())
    if expected_seeds:
        grouped: dict[tuple[float, float], list[int]] = defaultdict(list)
        for row in raw_rows:
            grouped[(row["tied_cap"], row["tied_scale"])].append(int(row["seed"]))
        for (cap, scale), seeds in sorted(grouped.items(), key=lambda kv: (kv[0][1], cap_sort_value(kv[0][0]))):
            unique = sorted(set(seeds))
            if unique != expected_seeds or len(unique) != len(seeds):
                raise ValueError(
                    f"expected seeds {expected_seeds} for cap={cap}, scale={scale}; "
                    f"found seeds {sorted(seeds)}"
                )

    agg_rows = aggregate(raw_rows)
    out_dir = args.out_dir or args.run_root
    suffix = f"_step{args.eval_step}" if args.eval_step is not None else ""
    write_csv(out_dir / f"nanogpt_cap_scale_multiseed_raw{suffix}.csv", raw_rows)
    write_csv(out_dir / f"nanogpt_cap_scale_multiseed_aggregate{suffix}.csv", agg_rows)

    complete = sum(int(r["complete"]) for r in raw_rows)
    print(f"Found {len(raw_rows)} runs; reached_comparison={complete}/{len(raw_rows)}")
    for row in agg_rows:
        print(
            f"c={fmt_cap(row['tied_cap']):>8} s={row['tied_scale']:g} "
            f"n={row['n']} complete={row['complete']} "
            f"val={row['eval_loss_mean']:.4f}±{row['eval_loss_sd']:.4f} "
            f"ppl={row['eval_ppl_mean']:.2f}±{row['eval_ppl_sd']:.2f} "
            f"rms={row['tied_rms_mean']:.4f}±{row['tied_rms_sd']:.4f}"
        )

    print(f"\nWrote: {out_dir / f'nanogpt_cap_scale_multiseed_raw{suffix}.csv'}")
    print(f"Wrote: {out_dir / f'nanogpt_cap_scale_multiseed_aggregate{suffix}.csv'}")
    print_latex_tables(agg_rows)


if __name__ == "__main__":
    main()
