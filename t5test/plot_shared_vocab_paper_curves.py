from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np


SERIES_ORDER = (
    "Hybrid Muon (tied)",
    "Hybrid Muon (untied)",
    "SCION-style Sign (tied)",
    "AF-Muon (tied)",
)
SERIES_COLORS = {
    "Hybrid Muon (tied)": "#6B7280",
    "Hybrid Muon (untied)": "#A3A3A3",
    "SCION-style Sign (tied)": "#D55E00",
    "AF-Muon (tied)": "#0072B2",
}
SERIES_LINESTYLES = {
    "Hybrid Muon (tied)": (0, (4.0, 2.0)),
    "Hybrid Muon (untied)": (0, (1.2, 1.4)),
    "SCION-style Sign (tied)": (0, (3.0, 1.4, 1.0, 1.4)),
    "AF-Muon (tied)": "-",
}
SERIES_MARKERS = {
    "Hybrid Muon (tied)": "s",
    "Hybrid Muon (untied)": "D",
    "SCION-style Sign (tied)": "^",
    "AF-Muon (tied)": "o",
}
PANELS = (
    ("train", "loss", "(a) Training loss"),
    ("eval", "eval_loss", "(b) Validation loss"),
    ("eval", "eval_ppl", "(c) Validation perplexity"),
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSON in {path} line {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def infer_seed(name: str) -> int:
    m = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:[_-]|$)", name)
    if not m:
        raise ValueError(f"could not infer seed from run directory: {name}")
    return int(m.group(1))


def infer_arm(name: str) -> str:
    low = name.lower()
    if "afmoun" in low:
        return "afmoun"
    if "scion" in low or "sign" in low:
        return "sign"
    if "hybrid_muon" in low or "muon" in low:
        return "muon"
    return "unknown"


def rho_ratio_from_run(run_dir: Path, config: dict) -> float | None:
    arm = infer_arm(run_dir.name)
    if arm == "muon":
        return None
    rho_hidden = config.get("rho_hidden")
    rho_output = config.get("rho_output")
    if rho_hidden:
        return float(rho_output) / float(rho_hidden)
    m = re.search(r"(?:^|_)rho([0-9.]+)(?:_|$)", run_dir.name)
    if not m:
        return None
    return float(m.group(1)) / 50.0


def wanted_label(run_dir: Path, config: dict) -> str | None:
    arm = infer_arm(run_dir.name)
    sharing = config.get("sharing", "full")
    rho_ratio = rho_ratio_from_run(run_dir, config)
    if arm == "muon" and sharing == "full":
        return "Hybrid Muon (tied)"
    if arm == "muon" and sharing == "untied":
        return "Hybrid Muon (untied)"
    if arm == "sign" and rho_ratio is not None and abs(rho_ratio - 1.0) < 1e-8:
        return "SCION-style Sign (tied)"
    if arm == "afmoun" and rho_ratio is not None and abs(rho_ratio - 1.0) < 1e-8:
        return "AF-Muon (tied)"
    return None


def read_selected_rows(run_roots: list[Path], *, plot_min_tokens: int) -> tuple[list[dict], list[dict]]:
    curve_rows: list[dict] = []
    final_rows: list[dict] = []
    seen_runs: dict[tuple[str, int], Path] = {}
    for root in run_roots:
        for metrics in sorted(root.glob("*/metrics.jsonl")):
            run_dir = metrics.parent
            config_path = run_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
            label = wanted_label(run_dir, config)
            if label is None:
                continue
            seed = infer_seed(run_dir.name)
            key = (label, seed)
            if key in seen_runs:
                raise ValueError(f"duplicate run for {label}, seed {seed}: {seen_runs[key]} and {run_dir}")
            seen_runs[key] = run_dir
            rows = load_jsonl(metrics)
            for row in rows:
                phase = row.get("phase")
                if phase not in {"train", "eval"}:
                    continue
                tokens = int(row.get("target_tokens_seen", row.get("tokens_seen", 0)) or 0)
                if tokens < int(plot_min_tokens):
                    continue
                row = dict(row)
                row["series"] = label
                row["seed"] = seed
                row["run_dir"] = str(run_dir)
                row["tokens"] = tokens
                curve_rows.append(row)
            evals = [r for r in rows if r.get("phase") == "eval"]
            trains = [r for r in rows if r.get("phase") == "train"]
            if evals:
                last_eval = evals[-1]
                best_eval = min(evals, key=lambda r: float(r.get("eval_loss", float("inf"))))
                last_train = trains[-1] if trains else {}
                final_rows.append(
                    {
                        "series": label,
                        "seed": seed,
                        "sharing": config.get("sharing", "full"),
                        "param_count": config.get("param_count"),
                        "final_step": last_eval.get("step"),
                        "final_tokens": last_eval.get("target_tokens_seen", last_eval.get("tokens_seen")),
                        "final_loss": last_eval.get("eval_loss"),
                        "final_ppl": last_eval.get("eval_ppl"),
                        "best_loss": best_eval.get("eval_loss"),
                        "best_ppl": best_eval.get("eval_ppl"),
                        "best_step": best_eval.get("step"),
                        "last_train_loss": last_train.get("loss"),
                    }
                )
    missing = [label for label in SERIES_ORDER if not any(r["series"] == label for r in final_rows)]
    if missing:
        warnings.warn("missing selected series: " + ", ".join(missing), stacklevel=2)
    return curve_rows, final_rows


def aggregate_curve(rows: list[dict], *, phase: str, key: str) -> dict[str, dict[str, np.ndarray]]:
    by_series_seed_token: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != phase:
            continue
        if key not in row or row[key] is None:
            continue
        value = float(row[key])
        if not math.isfinite(value):
            continue
        by_series_seed_token[(row["series"], int(row["seed"]), int(row["tokens"]))].append(value)

    by_series_token: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (series, seed, tokens), values in by_series_seed_token.items():
        by_series_token[(series, tokens)].append(float(values[-1]))

    out: dict[str, dict[str, np.ndarray]] = {}
    for series in SERIES_ORDER:
        tokens = sorted(t for (s, t) in by_series_token if s == series)
        means, sds, sems, counts = [], [], [], []
        for token in tokens:
            values = np.asarray(by_series_token[(series, token)], dtype=np.float64)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if values.size >= 2 else math.nan
            sem = float(sd / math.sqrt(values.size)) if values.size >= 2 else math.nan
            means.append(mean)
            sds.append(sd)
            sems.append(sem)
            counts.append(int(values.size))
        if tokens:
            out[series] = {
                "tokens": np.asarray(tokens, dtype=np.float64),
                "mean": np.asarray(means, dtype=np.float64),
                "sd": np.asarray(sds, dtype=np.float64),
                "sem": np.asarray(sems, dtype=np.float64),
                "count": np.asarray(counts, dtype=np.int64),
            }
    return out


def aggregate_final(rows: list[dict]) -> list[dict]:
    out = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["series"]].append(row)
    for series in SERIES_ORDER:
        group = sorted(grouped.get(series, []), key=lambda r: int(r["seed"]))
        if not group:
            continue
        result = {"series": series, "seeds": ",".join(str(r["seed"]) for r in group), "n": len(group)}
        for key in ["final_loss", "final_ppl", "best_loss", "best_ppl", "param_count"]:
            values = np.asarray([float(r[key]) for r in group if r.get(key) is not None], dtype=np.float64)
            if values.size:
                result[f"{key}_mean"] = float(values.mean())
                result[f"{key}_sd"] = float(values.std(ddof=1)) if values.size >= 2 else math.nan
        out.append(result)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def nice_xticks(min_billions: float, max_billions: float) -> list[float]:
    if max_billions <= 0.35:
        step = 0.05
    elif max_billions <= 0.8:
        step = 0.1
    else:
        step = 0.5
    start = math.floor(min_billions / step) * step
    end = math.ceil(max_billions / step) * step
    ticks = [round(x, 3) for x in np.arange(start, end + step * 0.5, step)]
    return [t for t in ticks if t >= -1e-9]


def plot(rows: list[dict], output: Path, *, ppl_scale: str, band: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

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
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), constrained_layout=False)
    min_token = min(int(r["tokens"]) for r in rows)
    max_token = max(int(r["tokens"]) for r in rows)
    min_b = min_token / 1e9
    max_b = max_token / 1e9
    xticks = nice_xticks(min_b, max_b)
    for ax, (phase, value_key, title) in zip(axes, PANELS):
        stats = aggregate_curve(rows, phase=phase, key=value_key)
        for series in SERIES_ORDER:
            if series not in stats:
                continue
            s = stats[series]
            x = s["tokens"] / 1e9
            y = s["mean"]
            color = SERIES_COLORS[series]
            ax.plot(
                x,
                y,
                label=series,
                color=color,
                linestyle=SERIES_LINESTYLES[series],
                linewidth=1.65,
                marker=SERIES_MARKERS[series],
                markersize=3.8,
                markevery=max(1, len(x) // 7),
                markerfacecolor="white" if series != "AF-Muon (tied)" else color,
                markeredgewidth=0.9,
            )
            half = s[band]
            good = np.isfinite(half) & (half > 0)
            if bool(good.any()):
                ax.fill_between(x[good], y[good] - half[good], y[good] + half[good], color=color, alpha=0.11, linewidth=0)
        ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold")
        pad = max(0.004, (max_b - min_b) * 0.035)
        ax.set_xlim(max(0.0, min_b - pad), max_b + pad)
        ax.set_xticks(xticks)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.75)
        ax.tick_params(labelsize=8)
        if value_key != "eval_ppl":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if value_key == "eval_ppl" and ppl_scale == "log":
            ax.set_yscale("log")
            ymin, ymax = ax.get_ylim()
            ticks = np.geomspace(max(ymin, 1e-12), ymax, num=4)
            ticks = np.asarray([round(float(t)) for t in ticks], dtype=np.float64)
            ticks = np.unique(ticks)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.yaxis.set_minor_formatter(NullFormatter())

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.4, handlelength=2.4, columnspacing=1.0)
    fig.supxlabel("Training tokens (billions)", fontsize=9.5, y=0.03)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.25, top=0.78, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(output.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot T5 shared-vocab topology control curves.")
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-min-tokens", type=int, default=0)
    parser.add_argument("--ppl-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--band", choices=["sem", "sd"], default="sem")
    args = parser.parse_args()

    curve_rows, final_rows = read_selected_rows(
        [Path(p) for p in args.run_roots],
        plot_min_tokens=int(args.plot_min_tokens),
    )
    if not curve_rows:
        raise RuntimeError("no selected train/eval rows found")
    output = Path(args.output)
    plot(curve_rows, output, ppl_scale=args.ppl_scale, band=args.band)
    write_csv(output.with_name(output.name + "_per_seed.csv"), final_rows)
    write_csv(output.with_name(output.name + "_aggregate.csv"), aggregate_final(final_rows))
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.svg')}")
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_name(output.name + '_per_seed.csv')}")
    print(f"wrote {output.with_name(output.name + '_aggregate.csv')}")


if __name__ == "__main__":
    main()
