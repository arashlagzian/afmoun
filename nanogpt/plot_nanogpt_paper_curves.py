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
ARM_COLORS = {
    "Hybrid Muon": "#6B7280",
    "SCION-style Sign": "#D55E00",
    "AF-Muon": "#0072B2",
    "AdamW tied / AdamW 1D": "#6B7280",
    "AF-Muon tied / AdamW 1D": "#0072B2",
    "AdamW tied / RMS-LMO 1D": "#A23B72",
    "AF-Muon tied / RMS-LMO 1D": "#009E73",
}
ARM_LINESTYLES = {
    "Hybrid Muon": (0, (4.0, 2.0)),
    "SCION-style Sign": (0, (3.0, 1.4, 1.0, 1.4)),
    "AF-Muon": "-",
    "AdamW tied / AdamW 1D": (0, (4.0, 2.0)),
    "AF-Muon tied / AdamW 1D": "-",
    "AdamW tied / RMS-LMO 1D": (0, (3.0, 1.4, 1.0, 1.4)),
    "AF-Muon tied / RMS-LMO 1D": "-",
}
ARM_MARKERS = {
    "Hybrid Muon": "s",
    "SCION-style Sign": "^",
    "AF-Muon": "o",
    "AdamW tied / AdamW 1D": "s",
    "AF-Muon tied / AdamW 1D": "o",
    "AdamW tied / RMS-LMO 1D": "^",
    "AF-Muon tied / RMS-LMO 1D": "D",
}
ARM_LEGEND_LABELS = {
    "AdamW tied / AdamW 1D": "Tied AdamW / 1D AdamW",
    "AF-Muon tied / AdamW 1D": "Tied c=3 / 1D AdamW",
    "AdamW tied / RMS-LMO 1D": "Tied AdamW / 1D RMS-LMO",
    "AF-Muon tied / RMS-LMO 1D": "Tied c=3 / 1D RMS-LMO",
}
PANELS = (
    ("train", "loss", "(a) Training loss"),
    ("eval", "eval_loss", "(b) Validation loss"),
    ("eval", "eval_ppl", "(c) Validation perplexity"),
)


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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_rows(run_root: Path, *, plot_min_tokens: int) -> list[dict]:
    rows = []
    seen: dict[tuple[str, int], Path] = {}
    for metrics in sorted(run_root.glob("*/metrics.jsonl")):
        run_dir = metrics.parent
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        arm = infer_arm(run_dir.name, cfg)
        seed = infer_seed(run_dir.name, cfg)
        if arm == "unknown" or seed is None:
            continue
        key = (arm, seed)
        if key in seen:
            raise ValueError(f"duplicate run for {arm}, seed {seed}: {seen[key]} and {run_dir}")
        seen[key] = run_dir
        for row in load_jsonl(metrics):
            phase = row.get("phase")
            if phase not in {"train", "eval"}:
                continue
            tokens = int(row.get("tokens_seen", 0) or 0)
            if tokens < int(plot_min_tokens):
                continue
            row = dict(row)
            row["arm"] = arm
            row["seed"] = seed
            row["tokens"] = tokens
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no train/eval rows found under {run_root}")
    return rows


def aggregate(rows: list[dict], *, phase: str, key: str) -> dict[str, dict[str, np.ndarray]]:
    by_seed_token: dict[tuple[str, int, int], float] = {}
    for row in rows:
        if row.get("phase") != phase or row.get(key) is None:
            continue
        try:
            value = float(row[key])
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        by_seed_token[(row["arm"], int(row["seed"]), int(row["tokens"]))] = value

    by_arm_token: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (arm, _seed, tokens), value in by_seed_token.items():
        by_arm_token[(arm, tokens)].append(value)

    out = {}
    for arm in ARM_ORDER:
        tokens = sorted(t for (a, t) in by_arm_token if a == arm)
        if not tokens:
            continue
        means, sds, sems, counts = [], [], [], []
        for token in tokens:
            values = np.asarray(by_arm_token[(arm, token)], dtype=np.float64)
            sd = float(values.std(ddof=1)) if values.size >= 2 else math.nan
            means.append(float(values.mean()))
            sds.append(sd)
            sems.append(float(sd / math.sqrt(values.size)) if values.size >= 2 else math.nan)
            counts.append(int(values.size))
        out[arm] = {
            "tokens": np.asarray(tokens, dtype=np.float64),
            "mean": np.asarray(means, dtype=np.float64),
            "sd": np.asarray(sds, dtype=np.float64),
            "sem": np.asarray(sems, dtype=np.float64),
            "count": np.asarray(counts, dtype=np.int64),
        }
    return out


def marker_indices(n: int, max_markers: int = 8) -> list[int]:
    if n <= max_markers:
        return list(range(max(0, n)))
    return sorted(set(np.linspace(0, n - 1, max_markers, dtype=int).tolist()))


def write_summary_csv(path: Path, panel_stats: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["panel", "arm", "tokens_seen", "mean", "sample_sd", "sem", "n_seeds"],
        )
        writer.writeheader()
        for panel, stats in panel_stats.items():
            for arm, data in stats.items():
                for token, mean, sd, sem, count in zip(data["tokens"], data["mean"], data["sd"], data["sem"], data["count"]):
                    writer.writerow(
                        {
                            "panel": panel,
                            "arm": arm,
                            "tokens_seen": int(token),
                            "mean": f"{float(mean):.12g}",
                            "sample_sd": f"{float(sd):.12g}" if math.isfinite(float(sd)) else "",
                            "sem": f"{float(sem):.12g}" if math.isfinite(float(sem)) else "",
                            "n_seeds": int(count),
                        }
                    )


def parse_float_list(text: str) -> list[float] | None:
    if not text:
        return None
    return [float(x) for x in text.split(",") if x.strip()]


def parse_ylim(text: str) -> tuple[float, float] | None:
    vals = parse_float_list(text)
    if not vals:
        return None
    if len(vals) != 2:
        raise ValueError(f"expected two comma-separated values for ylim, got {text!r}")
    return (float(vals[0]), float(vals[1]))


def plot(
    rows: list[dict],
    output: Path,
    *,
    band: str,
    ppl_scale: str,
    xlim_from_data: bool,
    x_max_billions: float | None,
    x_ticks: list[float] | None,
    train_ylim: tuple[float, float] | None,
    eval_loss_ylim: tuple[float, float] | None,
    eval_ppl_ylim: tuple[float, float] | None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )

    panel_stats = {}
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), constrained_layout=False)
    min_token = min(int(r["tokens"]) for r in rows)
    max_token = max(int(r["tokens"]) for r in rows)
    min_b = min_token / 1e9
    max_b = max_token / 1e9
    right = math.ceil(max_b * 2.0 - 1e-10) / 2.0
    left = min_b if xlim_from_data else 0.0
    if xlim_from_data:
        span = max(max_b - min_b, 1e-9)
        left = max(0.0, min_b - 0.03 * span)
        right = max_b + 0.03 * span
    if x_max_billions is not None:
        right = float(x_max_billions)

    for ax, (phase, metric, title) in zip(axes, PANELS):
        stats = aggregate(rows, phase=phase, key=metric)
        panel_stats[metric] = stats
        for arm in ARM_ORDER:
            if arm not in stats:
                continue
            data = stats[arm]
            x = data["tokens"] / 1e9
            y = data["mean"]
            half = data[band]
            color = ARM_COLORS[arm]
            ax.plot(
                x,
                y,
                label=arm,
                color=color,
                linestyle=ARM_LINESTYLES[arm],
                linewidth=1.75 if "AF-Muon" in arm else 1.55,
                marker=ARM_MARKERS[arm],
                markersize=3.7,
                markerfacecolor=color if "AF-Muon" in arm else "white",
                markeredgewidth=0.85,
                markevery=marker_indices(len(x)),
                solid_capstyle="round",
                dash_capstyle="round",
            )
            good = np.isfinite(half) & (half > 0)
            if metric == "eval_ppl" and ppl_scale == "log":
                good &= (y - half) > 0
            if bool(good.any()):
                ax.fill_between(x[good], y[good] - half[good], y[good] + half[good], color=color, alpha=0.12, linewidth=0)
        ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold")
        ax.set_xlim(left, right + (0.03 * right if not xlim_from_data else 0.0))
        if x_ticks:
            ax.set_xticks(x_ticks)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.75)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if metric == "eval_ppl" and ppl_scale == "log":
            ax.set_yscale("log")
        if metric == "loss" and train_ylim:
            ax.set_ylim(*train_ylim)
        elif metric == "eval_loss" and eval_loss_ylim:
            ax.set_ylim(*eval_loss_ylim)
        elif metric == "eval_ppl" and eval_ppl_ylim:
            ax.set_ylim(*eval_ppl_ylim)

    handles, labels = axes[0].get_legend_handles_labels()
    labels = [ARM_LEGEND_LABELS.get(label, label) for label in labels]
    ncol = 2 if len(labels) > 3 else len(labels)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=7.8, handlelength=2.4, columnspacing=1.0)
    fig.supxlabel("Training tokens (billions)", fontsize=9.5, y=0.03)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.25, top=0.74 if len(labels) > 3 else 0.78, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(output.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)
    write_summary_csv(output.with_name(output.name + "_summary.csv"), panel_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NanoGPT paper curves with seed bands.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-min-tokens", type=int, default=0)
    parser.add_argument("--band", choices=["sd", "sem"], default="sd")
    parser.add_argument("--ppl-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--x-max-billions", type=float, default=None)
    parser.add_argument("--x-ticks", default="", help="Comma-separated x ticks in billions, e.g. 0.2,0.5,0.8,1.1,1.3.")
    parser.add_argument("--train-ylim", default="", help="Comma-separated y limits for training loss, e.g. 3.55,3.90.")
    parser.add_argument("--eval-loss-ylim", default="", help="Comma-separated y limits for validation loss.")
    parser.add_argument("--eval-ppl-ylim", default="", help="Comma-separated y limits for validation perplexity.")
    parser.add_argument(
        "--xlim-from-data",
        action="store_true",
        help="Start the x-axis at the first retained token instead of zero; useful for tail-focused views.",
    )
    args = parser.parse_args()

    rows = read_rows(Path(args.run_root), plot_min_tokens=int(args.plot_min_tokens))
    plot(
        rows,
        Path(args.output),
        band=args.band,
        ppl_scale=args.ppl_scale,
        xlim_from_data=bool(args.xlim_from_data),
        x_max_billions=args.x_max_billions,
        x_ticks=parse_float_list(args.x_ticks),
        train_ylim=parse_ylim(args.train_ylim),
        eval_loss_ylim=parse_ylim(args.eval_loss_ylim),
        eval_ppl_ylim=parse_ylim(args.eval_ppl_ylim),
    )
    print(f"wrote {Path(args.output).with_suffix('.pdf')}")
    print(f"wrote {Path(args.output).with_suffix('.svg')}")
    print(f"wrote {Path(args.output).with_suffix('.png')}")
    print(f"wrote {Path(args.output).with_name(Path(args.output).name + '_summary.csv')}")


if __name__ == "__main__":
    main()
