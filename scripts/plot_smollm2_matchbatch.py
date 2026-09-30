import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["Hybrid Muon", "SCION-style Sign", "AF-Muon"]
STYLE = {
    "Hybrid Muon": {"color": "#7b8494", "marker": "s", "linestyle": "--", "label": "Hybrid Muon"},
    "SCION-style Sign": {"color": "#d55e00", "marker": "^", "linestyle": "--", "label": "SCION-style Sign"},
    "AF-Muon": {"color": "#0072b2", "marker": "o", "linestyle": "-", "label": "AF-Muon"},
}


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


def read_jsonl(path):
    rows = []
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_runs(run_root):
    out = []
    for run_dir in sorted(p for p in Path(run_root).iterdir() if p.is_dir() and p.name != "_logs"):
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        method = method_from_run(run_dir.name)
        seed = seed_from_run(run_dir.name)
        rows = read_jsonl(metrics_path)
        train = []
        evals = []
        for row in rows:
            phase = row.get("phase")
            if phase == "train" and row.get("loss") is not None and row.get("tokens_seen") is not None:
                train.append((float(row["tokens_seen"]) / 1e9, float(row["loss"])))
            elif phase == "eval" and row.get("eval_loss") is not None and row.get("tokens_seen") is not None:
                x = float(row["tokens_seen"]) / 1e9
                y = float(row["eval_loss"])
                ppl = float(row.get("eval_ppl", math.exp(min(20.0, y))))
                evals.append((x, y, ppl))
        out.append({"run": run_dir.name, "method": method, "seed": seed, "train": train, "eval": evals})
    return out


def aggregate_series(runs, kind, metric_index=1):
    grouped = defaultdict(list)
    for run in runs:
        series = run[kind]
        for item in series:
            x = item[0]
            y = item[metric_index]
            grouped[(run["method"], x)].append(y)

    by_method = {}
    for method in METHOD_ORDER:
        xs, means, sems = [], [], []
        for (m, x), vals in sorted(grouped.items()):
            if m != method:
                continue
            arr = np.array(vals, dtype=float)
            xs.append(x)
            means.append(float(arr.mean()))
            sems.append(float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0)
        by_method[method] = (np.array(xs), np.array(means), np.array(sems))
    return by_method


def filter_tail(series_by_method, min_tokens_b):
    filtered = {}
    for method, (xs, ys, es) in series_by_method.items():
        mask = xs >= min_tokens_b
        filtered[method] = (xs[mask], ys[mask], es[mask])
    return filtered


def apply_template():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": False,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def plot_panel(ax, data, title):
    for method in METHOD_ORDER:
        xs, ys, sem = data[method]
        if len(xs) == 0:
            continue
        st = STYLE[method]
        ax.plot(
            xs,
            ys,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=1.35,
            markersize=2.6,
            markerfacecolor="white" if method != "AF-Muon" else st["color"],
            markeredgewidth=0.75,
            label=st["label"],
        )
        if np.any(sem > 0):
            ax.fill_between(xs, ys - sem, ys + sem, color=st["color"], alpha=0.10, linewidth=0)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=3)
    ax.grid(axis="y", color="#d7dce2", linewidth=0.55)
    ax.tick_params(labelsize=7.5, length=3)


def make_plot(runs, output, min_tokens_b=None):
    train = aggregate_series(runs, "train", 1)
    val = aggregate_series(runs, "eval", 1)
    ppl = aggregate_series(runs, "eval", 2)
    if min_tokens_b is not None:
        train = filter_tail(train, min_tokens_b)
        val = filter_tail(val, min_tokens_b)
        ppl = filter_tail(ppl, min_tokens_b)

    apply_template()
    fig, axes = plt.subplots(1, 3, figsize=(6.15, 1.92), sharex=False)
    plot_panel(axes[0], train, "(a) Training loss")
    plot_panel(axes[1], val, "(b) Validation loss")
    plot_panel(axes[2], ppl, "(c) Validation perplexity")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_xlim(left=min_tokens_b if min_tokens_b is not None else 0.0)
    fig.supxlabel("Training tokens (billions)", fontsize=8.8, y=-0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.4, bbox_to_anchor=(0.5, 1.10), handlelength=2.6, columnspacing=1.25)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91), w_pad=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="runs/01_paper_main_results_smollm2_135m_2p5b_v2_fp32master_matchbatch524k")
    p.add_argument("--out-dir", default="outputs/smollm2_135m_2p5b_fp32master_matchbatch524k_verified/figs")
    p.add_argument("--tail-tokens-b", type=float, default=0.5)
    args = p.parse_args()

    runs = load_runs(args.run_root)
    if not runs:
        raise SystemExit(f"No runs found under {args.run_root}")

    out_dir = Path(args.out_dir)
    full_pdf = out_dir / "smollm2_135m_matchbatch524k_curves_full.pdf"
    tail_pdf = out_dir / "smollm2_135m_matchbatch524k_curves_tail500m.pdf"
    make_plot(runs, full_pdf, min_tokens_b=None)
    make_plot(runs, tail_pdf, min_tokens_b=args.tail_tokens_b)

    print("Wrote:")
    print(full_pdf)
    print(tail_pdf)


if __name__ == "__main__":
    main()
