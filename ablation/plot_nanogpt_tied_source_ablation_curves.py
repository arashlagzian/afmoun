from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SOURCE_ORDER = ("Sparse input only", "Dense output only", "Sparse input + dense output")
SOURCE_COLORS = {
    "Sparse input only": "#6B7280",
    "Dense output only": "#D55E00",
    "Sparse input + dense output": "#0072B2",
}
SOURCE_LINESTYLES = {
    "Sparse input only": (0, (4.0, 2.0)),
    "Dense output only": (0, (3.0, 1.4, 1.0, 1.4)),
    "Sparse input + dense output": "-",
}
SOURCE_MARKERS = {
    "Sparse input only": "s",
    "Dense output only": "^",
    "Sparse input + dense output": "o",
}
SOURCE_MARKER_FACES = {
    "Sparse input only": "white",
    "Dense output only": "white",
    "Sparse input + dense output": SOURCE_COLORS["Sparse input + dense output"],
}
PANEL_SPECS = {
    "train_loss": ("train", "loss", "Training loss"),
    "eval_loss": ("eval", "eval_loss", "Validation loss"),
    "eval_ppl": ("eval", "eval_ppl", "Validation perplexity"),
}
DEFAULT_PANELS = ("train_loss", "eval_loss", "eval_ppl")
SUPPORTED_FORMATS = ("pdf", "svg", "png")


def paper_rcparams() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "STIX Two Text",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 8.25,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.fontsize": 7.25,
        "legend.frameon": False,
        "legend.handlelength": 2.5,
        "legend.handletextpad": 0.55,
        "legend.columnspacing": 1.25,
        "xtick.labelsize": 7.25,
        "ytick.labelsize": 7.25,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def infer_source(run_name: str, config: dict | None = None) -> str:
    source = (config or {}).get("tied_grad_source")
    if source == "input_only":
        return "Sparse input only"
    if source == "output_only":
        return "Dense output only"
    if source == "both":
        return "Sparse input + dense output"

    normalized = re.sub(r"[^a-z0-9]+", "_", run_name.lower()).strip("_")
    if "sparse_input_only" in normalized or "input_only" in normalized:
        return "Sparse input only"
    if "dense_output_only" in normalized or "output_only" in normalized:
        return "Dense output only"
    if "input_plus_output" in normalized or "_both_" in normalized:
        return "Sparse input + dense output"
    return "unknown"


def infer_seed(run_name: str, config: dict | None = None) -> int | None:
    if config and config.get("seed") is not None:
        return int(config["seed"])
    match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:[_-]|$)", run_name.lower())
    return int(match.group(1)) if match else None


def read_metrics(run_root: Path, *, plot_min_tokens: int = 0) -> list[dict]:
    rows: list[dict] = []
    unknown_runs: list[str] = []
    for metrics_path in sorted(run_root.glob("*/metrics.jsonl")):
        run_dir = metrics_path.parent
        cfg_path = run_dir / "config.json"
        cfg = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        source = infer_source(run_dir.name, cfg)
        seed = infer_seed(run_dir.name, cfg)
        if source == "unknown":
            unknown_runs.append(run_dir.name)
            continue
        if seed is None:
            raise ValueError(f"could not infer seed from run directory: {run_dir.name}")
        with metrics_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {metrics_path} at line {line_number}: {exc.msg}"
                    ) from exc
                phase = row.get("phase")
                if phase not in {"train", "eval"}:
                    continue
                tokens_seen = int(row.get("tokens_seen", 0) or 0)
                if tokens_seen < int(plot_min_tokens):
                    continue
                row["phase"] = phase
                row["source"] = source
                row["seed"] = seed
                row["run_name"] = run_dir.name
                rows.append(row)
    if unknown_runs:
        warnings.warn(
            "ignoring metrics from unrecognized source runs: "
            + ", ".join(sorted(set(unknown_runs))),
            stacklevel=2,
        )
    return rows


def aggregate(rows: list[dict], *, phase: str, value_key: str) -> dict[str, dict[str, np.ndarray]]:
    by_seed: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("phase") != phase:
            continue
        source = str(row.get("source", "unknown"))
        seed = row.get("seed")
        tokens = row.get("tokens_seen")
        value = row.get(value_key)
        if source == "unknown" or seed is None or tokens is None or value is None:
            continue
        try:
            tokens_i = int(tokens)
            value_f = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if tokens_i < 0 or not math.isfinite(value_f):
            continue
        by_seed[source][int(seed)][tokens_i] = value_f

    out: dict[str, dict[str, np.ndarray]] = {}
    for source, seed_points in by_seed.items():
        all_tokens = sorted({token for points in seed_points.values() for token in points})
        means = []
        sds = []
        counts = []
        for token in all_tokens:
            values = np.asarray(
                [points[token] for points in seed_points.values() if token in points],
                dtype=np.float64,
            )
            means.append(float(np.mean(values)))
            sds.append(float(np.std(values, ddof=1)) if values.size >= 2 else math.nan)
            counts.append(int(values.size))
        out[source] = {
            "tokens": np.asarray(all_tokens, dtype=np.int64),
            "mean": np.asarray(means, dtype=np.float64),
            "sd": np.asarray(sds, dtype=np.float64),
            "count": np.asarray(counts, dtype=np.int64),
        }
    return out


def _marker_indices(n_points: int, maximum_markers: int = 8) -> list[int]:
    if n_points <= 0:
        return []
    if n_points <= maximum_markers:
        return list(range(n_points))
    return sorted(set(np.linspace(0, n_points - 1, maximum_markers, dtype=int).tolist()))


def _nice_axis_max(rows: Sequence[dict], token_divisor: float) -> float | None:
    xs = []
    for row in rows:
        try:
            value = float(row.get("tokens_seen")) / token_divisor
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            xs.append(value)
    if not xs:
        return None
    return math.ceil(max(xs) * 2.0 - 1e-10) / 2.0


def _half_width(sd: np.ndarray, count: np.ndarray, uncertainty: str) -> np.ndarray:
    if uncertainty == "none":
        return np.full_like(sd, np.nan)
    if uncertainty == "sd":
        return sd.copy()
    if uncertainty == "sem":
        with np.errstate(invalid="ignore", divide="ignore"):
            return sd / np.sqrt(count.astype(np.float64))
    raise ValueError(f"unsupported uncertainty: {uncertainty}")


def _parse_formats(items: Iterable[str]) -> tuple[str, ...]:
    formats: list[str] = []
    for item in items:
        for fmt in item.split(","):
            normalized = fmt.strip().lower().lstrip(".")
            if normalized not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"unsupported format {fmt!r}; choose from {SUPPORTED_FORMATS}"
                )
            if normalized not in formats:
                formats.append(normalized)
    if not formats:
        raise ValueError("at least one output format is required")
    return tuple(formats)


def write_summary_csv(path: Path, summaries: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "panel",
                "phase",
                "metric",
                "source",
                "tokens_seen",
                "mean",
                "sample_sd",
                "sem",
                "n_seeds",
            ),
        )
        writer.writeheader()
        for panel, panel_summary in summaries.items():
            phase, metric, _ = PANEL_SPECS[panel]
            for source in SOURCE_ORDER:
                if source not in panel_summary:
                    continue
                stats = panel_summary[source]
                for token, mean, sd, count in zip(
                    stats["tokens"], stats["mean"], stats["sd"], stats["count"]
                ):
                    sem = sd / math.sqrt(int(count)) if math.isfinite(float(sd)) else math.nan
                    writer.writerow(
                        {
                            "panel": panel,
                            "phase": phase,
                            "metric": metric,
                            "source": source,
                            "tokens_seen": int(token),
                            "mean": f"{float(mean):.12g}",
                            "sample_sd": f"{float(sd):.12g}" if math.isfinite(float(sd)) else "",
                            "sem": f"{float(sem):.12g}" if math.isfinite(float(sem)) else "",
                            "n_seeds": int(count),
                        }
                    )


def plot_curves(
    rows: list[dict],
    *,
    output: Path,
    panels: Sequence[str],
    uncertainty: str,
    width: float,
    height: float,
    dpi: int,
    formats: Sequence[str],
    x_min: float | None,
) -> tuple[list[Path], dict[str, dict[str, dict[str, np.ndarray]]]]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    token_divisor = 1e9
    x_axis_max = _nice_axis_max(rows, token_divisor)
    summaries = {}
    for panel in panels:
        phase, value_key, _ = PANEL_SPECS[panel]
        summaries[panel] = aggregate(rows, phase=phase, value_key=value_key)
        if not summaries[panel]:
            raise RuntimeError(f"no finite data found for panel {panel!r}")

    with mpl.rc_context(paper_rcparams()):
        fig, axes = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False)
        axes_1d = axes.ravel()
        legend_handles: dict[str, object] = {}

        for panel_index, (ax, panel) in enumerate(zip(axes_1d, panels)):
            _, _, panel_title = PANEL_SPECS[panel]
            panel_summary = summaries[panel]
            for source in SOURCE_ORDER:
                if source not in panel_summary:
                    continue
                stats = panel_summary[source]
                x = stats["tokens"].astype(np.float64) / token_divisor
                mean = stats["mean"]
                half_width = _half_width(stats["sd"], stats["count"], uncertainty)
                color = SOURCE_COLORS[source]
                lower = mean - half_width
                upper = mean + half_width
                valid_band = np.isfinite(half_width) & np.isfinite(lower) & np.isfinite(upper)
                lower = np.maximum(lower, 0.0)
                if np.any(valid_band):
                    ax.fill_between(
                        x,
                        lower,
                        upper,
                        where=valid_band,
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                        zorder=1,
                    )
                (line,) = ax.plot(
                    x,
                    mean,
                    label=source,
                    color=color,
                    linestyle=SOURCE_LINESTYLES[source],
                    linewidth=1.75 if source == "Sparse input + dense output" else 1.55,
                    marker=SOURCE_MARKERS[source],
                    markerfacecolor=SOURCE_MARKER_FACES[source],
                    markeredgecolor=color,
                    markeredgewidth=0.65,
                    markersize=3.2,
                    markevery=_marker_indices(len(x)),
                    solid_capstyle="round",
                    dash_capstyle="round",
                    zorder=3 if source == "Sparse input + dense output" else 2,
                )
                legend_handles.setdefault(source, line)

            panel_letter = chr(ord("a") + panel_index)
            ax.set_title(f"({panel_letter}) {panel_title}", loc="left", pad=3.0)
            if x_axis_max is not None:
                ax.set_xlim(left=0 if x_min is None else x_min, right=x_axis_max + 0.03 * x_axis_max)
            else:
                ax.set_xlim(left=0 if x_min is None else x_min)
            ax.xaxis.set_major_locator(MultipleLocator(0.1) if x_axis_max and x_axis_max <= 0.6 else MaxNLocator(nbins=4, min_n_ticks=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.grid(axis="y", color="#D6DAE1", linewidth=0.5, alpha=0.75)
            ax.grid(False, axis="x")
            for spine in (ax.spines["left"], ax.spines["bottom"]):
                spine.set_color("#4B5563")
                spine.set_linewidth(0.6)

        fig.subplots_adjust(
            left=0.060,
            right=0.995,
            bottom=0.235,
            top=0.77,
            wspace=0.30 if len(panels) >= 3 else 0.24,
        )
        fig.supxlabel("Training tokens (billions)", x=0.54, y=0.045, fontsize=8.0)
        ordered_handles = [legend_handles[s] for s in SOURCE_ORDER if s in legend_handles]
        ordered_labels = [s for s in SOURCE_ORDER if s in legend_handles]
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.965),
            ncol=max(1, len(ordered_handles)),
            borderaxespad=0,
        )

        output_stem = (
            output.with_suffix("")
            if output.suffix.lower().lstrip(".") in SUPPORTED_FORMATS
            else output
        )
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written = []
        for fmt in formats:
            path = output_stem.parent / f"{output_stem.name}.{fmt}"
            kwargs: dict[str, object] = {
                "format": fmt,
                "facecolor": "white",
                "transparent": False,
                "bbox_inches": None,
            }
            if fmt == "png":
                kwargs["dpi"] = dpi
            fig.savefig(path, **kwargs)
            written.append(path)
        plt.close(fig)
    return written, summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot NanoGPT tied-table gradient-source ablation curves."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=tuple(PANEL_SPECS),
        default=list(DEFAULT_PANELS),
    )
    parser.add_argument("--uncertainty", choices=("sd", "sem", "none"), default="none")
    parser.add_argument("--plot-min-tokens", type=int, default=0)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--width", type=float, default=5.5)
    parser.add_argument("--height", type=float, default=2.30)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", nargs="+", default=list(SUPPORTED_FORMATS))
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()

    rows = read_metrics(args.run_root, plot_min_tokens=int(args.plot_min_tokens))
    if not rows:
        raise RuntimeError(f"no train/eval metrics found under {args.run_root}")
    formats = _parse_formats(args.formats)
    written, summaries = plot_curves(
        rows,
        output=args.output,
        panels=args.panels,
        uncertainty=args.uncertainty,
        width=float(args.width),
        height=float(args.height),
        dpi=int(args.dpi),
        formats=formats,
        x_min=args.x_min,
    )

    output_stem = (
        args.output.with_suffix("")
        if args.output.suffix.lower().lstrip(".") in SUPPORTED_FORMATS
        else args.output
    )
    summary_path = args.summary_csv or output_stem.with_name(
        f"{output_stem.name}_summary.csv"
    )
    write_summary_csv(summary_path, summaries)

    for path in written:
        print(f"wrote {path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
