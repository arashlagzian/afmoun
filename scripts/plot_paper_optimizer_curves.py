
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


ARM_ORDER = ("Hybrid Muon", "SCION-style Sign", "AF-Muon")
ARM_COLORS = {
    "Hybrid Muon": "#6B7280",
    "SCION-style Sign": "#D55E00",
    "AF-Muon": "#0072B2",
}
ARM_LINESTYLES = {
    "Hybrid Muon": (0, (4.0, 2.0)),
    "SCION-style Sign": (0, (3.0, 1.4, 1.0, 1.4)),
    "AF-Muon": "-",
}
ARM_MARKERS = {
    "Hybrid Muon": "s",
    "SCION-style Sign": "^",
    "AF-Muon": "o",
}
ARM_MARKER_FACES = {
    "Hybrid Muon": "white",
    "SCION-style Sign": "white",
    "AF-Muon": ARM_COLORS["AF-Muon"],
}

PANEL_SPECS = {
    "train_loss": ("train", "loss", "Training loss", "Cross-entropy loss"),
    "eval_loss": ("eval", "eval_loss", "Validation loss", "Cross-entropy loss"),
    "eval_ppl": ("eval", "eval_ppl", "Validation perplexity", "Perplexity"),
}
DEFAULT_PANELS = ("train_loss", "eval_loss", "eval_ppl")
SUPPORTED_FORMATS = ("pdf", "svg", "png")


def infer_arm(run_name: str) -> str:
    """Infer a paper-facing optimizer label from a run-directory name."""
    normalized = re.sub(r"[^a-z0-9]+", "_", run_name.lower()).strip("_")
    parts = set(normalized.split("_"))

    if (
        "afmoun" in parts
        or "af_muon" in normalized
        or ("af" in parts and ("c3" in parts or "finite" in parts))
    ):
        return "AF-Muon"
    if "scion" in parts or "sign" in parts:
        return "SCION-style Sign"
    if "hybrid_muon" in normalized or "hybrid" in parts or "muon" in parts:
        return "Hybrid Muon"
    return "unknown"


def infer_seed(run_name: str) -> int | None:
    match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:[_-]|$)", run_name.lower())
    return int(match.group(1)) if match else None


def read_metrics(
    run_root: Path,
    *,
    strict_unknown_runs: bool = False,
    plot_min_tokens: int = 0,
) -> list[dict]:
    """Read train/eval JSONL rows and attach canonical arm/seed metadata.

    For publication plots, one directory per (optimizer, seed) is expected.
    Rejecting duplicate seeds prevents accidental pseudoreplication.
    """
    rows: list[dict] = []
    unknown_runs: list[str] = []
    runs_by_arm_seed: dict[tuple[str, int], set[str]] = defaultdict(set)

    for metrics_path in sorted(run_root.glob("*/metrics.jsonl")):
        run_name = metrics_path.parent.name
        arm = infer_arm(run_name)
        seed = infer_seed(run_name)
        if arm == "unknown":
            unknown_runs.append(run_name)
            continue
        if seed is None:
            raise ValueError(f"could not infer seed from recognized run directory: {run_name}")

        runs_by_arm_seed[(arm, seed)].add(run_name)
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
                if not isinstance(row, dict):
                    raise ValueError(
                        f"expected a JSON object in {metrics_path} at line {line_number}"
                    )
                phase = row.get("phase")
                if phase is None:
                    if "eval_loss" in row or "eval_ppl" in row:
                        phase = "eval"
                    elif "loss" in row:
                        phase = "train"
                if phase not in {"train", "eval"}:
                    continue
                try:
                    tokens_seen = int(row.get("tokens_seen", 0))
                except (TypeError, ValueError):
                    tokens_seen = 0
                if tokens_seen < int(plot_min_tokens):
                    continue
                row["phase"] = phase
                row["run_name"] = run_name
                row["arm"] = arm
                row["seed"] = seed
                rows.append(row)

    if unknown_runs and strict_unknown_runs:
        names = ", ".join(sorted(set(unknown_runs)))
        raise ValueError(
            "found metrics for unrecognized run directories: "
            f"{names}. Rename them to include muon/hybrid, scion/sign, or afmoun/af_c3."
        )
    if unknown_runs:
        warnings.warn(
            "ignoring metrics from unrecognized run directories: "
            + ", ".join(sorted(set(unknown_runs))),
            stacklevel=2,
        )

    duplicate_seed_runs = {
        key: names for key, names in runs_by_arm_seed.items() if len(names) > 1
    }
    if duplicate_seed_runs:
        details = "; ".join(
            f"{arm}, seed {seed}: {sorted(names)}"
            for (arm, seed), names in sorted(duplicate_seed_runs.items())
        )
        raise ValueError(
            "multiple run directories share an optimizer/seed and would not be "
            f"independent replicates ({details}). Select the intended directory "
            "or construct one explicit resume chain before plotting."
        )
    return rows


def aggregate(
    rows: list[dict],
    *,
    phase: str,
    value_key: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate one observation per seed/token, then summarize across seeds.

    If a scheduled and a final evaluation are both logged at the same token for
    the same seed, the last finite record wins. This avoids counting duplicated
    JSONL rows as independent seeds.
    """
    by_seed: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row.get("phase") != phase:
            continue
        arm = str(row.get("arm", "unknown"))
        seed = row.get("seed")
        tokens = row.get("tokens_seen")
        value = row.get("lm_loss") if value_key == "loss" and "lm_loss" in row else row.get(value_key)
        if arm == "unknown" or seed is None or tokens is None or value is None:
            continue
        try:
            tokens_i = int(tokens)
            value_f = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if tokens_i < 0 or not math.isfinite(value_f):
            continue
        by_seed[arm][int(seed)][tokens_i] = value_f

    out: dict[str, dict[str, np.ndarray]] = {}
    for arm, seed_points in by_seed.items():
        checkpoint_grids = [frozenset(points) for points in seed_points.values()]
        if len(set(checkpoint_grids)) > 1:
            warnings.warn(
                f"{arm} has unequal {phase}/{value_key} checkpoint grids across "
                "seeds; each token is summarized over the seeds available there.",
                stacklevel=2,
            )

        all_tokens = sorted({token for points in seed_points.values() for token in points})
        means: list[float] = []
        sample_sds: list[float] = []
        counts: list[int] = []
        for token in all_tokens:
            values = np.asarray(
                [points[token] for points in seed_points.values() if token in points],
                dtype=np.float64,
            )
            means.append(float(np.mean(values)))
            sample_sds.append(
                float(np.std(values, ddof=1)) if values.size >= 2 else math.nan
            )
            counts.append(int(values.size))

        out[arm] = {
            "tokens": np.asarray(all_tokens, dtype=np.int64),
            "mean": np.asarray(means, dtype=np.float64),
            "sd": np.asarray(sample_sds, dtype=np.float64),
            "count": np.asarray(counts, dtype=np.int64),
        }
    return out


def _token_scale(rows: Sequence[dict], requested: str) -> tuple[float, str]:
    if requested == "billions":
        return 1e9, "Training tokens (billions)"
    if requested == "millions":
        return 1e6, "Training tokens (millions)"

    finite_tokens: list[float] = []
    for row in rows:
        try:
            value = float(row.get("tokens_seen"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite_tokens.append(value)
    maximum = max(finite_tokens, default=0.0)
    if maximum >= 1e9:
        return 1e9, "Training tokens (billions)"
    return 1e6, "Training tokens (millions)"


def _nice_axis_max(values: Sequence[dict], token_divisor: float) -> float | None:
    finite_tokens: list[float] = []
    for row in values:
        try:
            value = float(row.get("tokens_seen"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            finite_tokens.append(value / token_divisor)
    if not finite_tokens:
        return None
    maximum = max(finite_tokens)
    if token_divisor >= 1e9:
        return math.ceil(maximum * 2.0 - 1e-10) / 2.0
    return math.ceil(maximum * 10.0 - 1e-10) / 10.0


def _uncertainty_half_width(
    sd: np.ndarray,
    count: np.ndarray,
    uncertainty: str,
) -> np.ndarray:
    if uncertainty == "none":
        return np.full_like(sd, np.nan)
    if uncertainty == "sd":
        return sd.copy()
    if uncertainty == "sem":
        with np.errstate(invalid="ignore", divide="ignore"):
            return sd / np.sqrt(count.astype(np.float64))
    raise ValueError(f"unsupported uncertainty type: {uncertainty}")


def _marker_indices(n_points: int, maximum_markers: int = 8) -> list[int]:
    if n_points <= 0:
        return []
    if n_points <= maximum_markers:
        return list(range(n_points))
    return sorted(set(np.linspace(0, n_points - 1, maximum_markers, dtype=int).tolist()))


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


def plot_curves(
    rows: list[dict],
    *,
    title: str,
    output: Path,
    dpi: int,
    panels: Sequence[str],
    uncertainty: str,
    token_unit: str,
    ppl_scale: str,
    width: float,
    height: float,
    formats: Sequence[str],
    x_min: float | None,
    layout: str,
) -> tuple[list[Path], dict[str, dict[str, dict[str, np.ndarray]]]]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    token_divisor, shared_xlabel = _token_scale(rows, token_unit)
    x_axis_max = _nice_axis_max(rows, token_divisor)
    summaries: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for panel in panels:
        phase, value_key, _, _ = PANEL_SPECS[panel]
        summaries[panel] = aggregate(rows, phase=phase, value_key=value_key)
        if not summaries[panel]:
            raise RuntimeError(f"no finite recognized data found for panel {panel!r}")
        panel_summary = summaries[panel]
        missing_arms = [arm for arm in ARM_ORDER if arm not in panel_summary]
        if missing_arms:
            warnings.warn(
                f"{panel} is missing expected optimizer arm(s): "
                + ", ".join(missing_arms),
                stacklevel=2,
            )
        arm_grids = [
            frozenset(stats["tokens"].tolist()) for stats in panel_summary.values()
        ]
        if len(set(arm_grids)) > 1:
            warnings.warn(
                f"{panel} uses unequal checkpoint grids across optimizer arms; "
                "compare values only at matched token counts.",
                stacklevel=2,
            )

    with mpl.rc_context(paper_rcparams()):
        if layout == "vertical":
            fig, axes = plt.subplots(len(panels), 1, figsize=(width, height), squeeze=False)
        else:
            fig, axes = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False)
        axes_1d = axes.ravel()
        legend_handles: dict[str, object] = {}

        for panel_index, (ax, panel) in enumerate(zip(axes_1d, panels)):
            phase, value_key, panel_title, ylabel = PANEL_SPECS[panel]
            panel_summary = summaries[panel]

            for arm in ARM_ORDER:
                if arm not in panel_summary:
                    continue
                stats = panel_summary[arm]
                x = stats["tokens"].astype(np.float64) / token_divisor
                mean = stats["mean"]
                half_width = _uncertainty_half_width(
                    stats["sd"], stats["count"], uncertainty
                )
                color = ARM_COLORS[arm]

                lower = mean - half_width
                upper = mean + half_width
                valid_band = (
                    np.isfinite(half_width)
                    & np.isfinite(lower)
                    & np.isfinite(upper)
                )
                if panel == "eval_ppl" and ppl_scale == "log":
                    valid_band &= lower > 0
                else:
                    lower = np.maximum(lower, 0.0)
                if np.any(valid_band):
                    ax.fill_between(
                        x,
                        lower,
                        upper,
                        where=valid_band,
                        interpolate=False,
                        color=color,
                        alpha=0.13,
                        linewidth=0,
                        zorder=1,
                    )

                (line,) = ax.plot(
                    x,
                    mean,
                    label=arm,
                    color=color,
                    linestyle=ARM_LINESTYLES[arm],
                    linewidth=1.75 if arm == "AF-Muon" else 1.55,
                    marker=ARM_MARKERS[arm],
                    markerfacecolor=ARM_MARKER_FACES[arm],
                    markeredgecolor=color,
                    markeredgewidth=0.65,
                    markersize=3.2,
                    markevery=_marker_indices(len(x)),
                    solid_capstyle="round",
                    dash_capstyle="round",
                    zorder=3 if arm == "AF-Muon" else 2,
                )
                legend_handles.setdefault(arm, line)

            panel_letter = chr(ord("a") + panel_index)
            ax.set_title(f"({panel_letter}) {panel_title}", loc="left", pad=3.0)
            ax.set_ylabel("")
            if x_axis_max is not None:
                right_pad = 0.03 * x_axis_max if token_divisor >= 1e9 else 0.015 * x_axis_max
                ax.set_xlim(left=0 if x_min is None else x_min, right=x_axis_max + right_pad)
            else:
                ax.set_xlim(left=0 if x_min is None else x_min)
            ax.margins(x=0)
            if token_divisor >= 1e9 and x_axis_max is not None and x_axis_max <= 5.0:
                ax.xaxis.set_major_locator(MultipleLocator(0.5))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            if panel == "eval_ppl" and ppl_scale == "log":
                ax.set_yscale("log")
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.grid(axis="y", color="#D6DAE1", linewidth=0.5, alpha=0.75)
            ax.grid(False, axis="x")
            for spine in (ax.spines["left"], ax.spines["bottom"]):
                spine.set_color("#4B5563")
                spine.set_linewidth(0.6)

        top = 0.70 if title else 0.77
        if layout == "vertical":
            top = 0.88 if title else 0.92
            fig.subplots_adjust(
                left=0.115,
                right=0.995,
                bottom=0.090,
                top=top,
                hspace=0.46,
            )
        else:
            fig.subplots_adjust(
                left=0.060,
                right=0.995,
                bottom=0.235,
                top=top,
                wspace=0.30 if len(panels) >= 3 else 0.24,
            )
        fig.supxlabel(shared_xlabel, x=0.54, y=0.020 if layout == "vertical" else 0.045, fontsize=8.0)

        ordered_handles = [legend_handles[arm] for arm in ARM_ORDER if arm in legend_handles]
        ordered_labels = [arm for arm in ARM_ORDER if arm in legend_handles]
        legend_y = 0.955 if layout == "vertical" else (0.885 if title else 0.965)
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.54, legend_y),
            ncol=max(1, len(ordered_handles)),
            borderaxespad=0,
        )
        if title:
            fig.suptitle(title, x=0.54, y=0.985, fontsize=8.75, fontweight="semibold")

        output_stem = output.with_suffix("") if output.suffix.lower().lstrip(".") in SUPPORTED_FORMATS else output
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for fmt in formats:
            path = output_stem.parent / f"{output_stem.name}.{fmt}"
            save_kwargs: dict[str, object] = {
                "format": fmt,
                "facecolor": "white",
                "transparent": False,
                "bbox_inches": None,
            }
            if fmt == "png":
                save_kwargs["dpi"] = dpi
            fig.savefig(path, **save_kwargs)
            written.append(path)
        plt.close(fig)

    return written, summaries


def write_summary_csv(
    path: Path,
    summaries: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    """Write the exact statistics underlying the figure for auditability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "panel",
                "phase",
                "metric",
                "arm",
                "tokens_seen",
                "mean",
                "sample_sd",
                "sem",
                "n_seeds",
            ),
        )
        writer.writeheader()
        for panel, panel_summary in summaries.items():
            phase, metric, _, _ = PANEL_SPECS[panel]
            for arm in ARM_ORDER:
                if arm not in panel_summary:
                    continue
                stats = panel_summary[arm]
                for token, mean, sd, count in zip(
                    stats["tokens"], stats["mean"], stats["sd"], stats["count"]
                ):
                    sem = sd / math.sqrt(int(count)) if math.isfinite(float(sd)) else math.nan
                    writer.writerow(
                        {
                            "panel": panel,
                            "phase": phase,
                            "metric": metric,
                            "arm": arm,
                            "tokens_seen": int(token),
                            "mean": f"{float(mean):.12g}",
                            "sample_sd": f"{float(sd):.12g}" if math.isfinite(float(sd)) else "",
                            "sem": f"{float(sem):.12g}" if math.isfinite(float(sem)) else "",
                            "n_seeds": int(count),
                        }
                    )


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create final-size, seed-aware paper optimizer curves from metrics.jsonl runs."
        )
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output stem or filename; PDF/SVG/PNG extensions are generated as requested.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional in-figure title. Paper captions normally make this unnecessary.",
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=tuple(PANEL_SPECS),
        default=list(DEFAULT_PANELS),
        help="Panels to include. Add eval_ppl for an appendix perplexity panel.",
    )
    parser.add_argument(
        "--uncertainty",
        choices=("sd", "sem", "none"),
        default="sd",
        help="Band around the across-seed mean; sample SD is the paper default.",
    )
    parser.add_argument(
        "--token-unit",
        choices=("auto", "millions", "billions"),
        default="auto",
    )
    parser.add_argument("--ppl-scale", choices=("linear", "log"), default="log")
    parser.add_argument(
        "--plot-min-tokens",
        type=int,
        default=0,
        help="Ignore earlier points in the plotted curves, useful for hiding initialization spikes.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Visible left x-axis limit in the selected token unit; useful with --plot-min-tokens.",
    )
    parser.add_argument(
        "--layout",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Panel arrangement. Use vertical for long-horizon single-seed appendix curves.",
    )
    parser.add_argument("--width", type=float, default=5.5, help="Figure width in inches.")
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Figure height in inches (default: 2.15 for two panels, 2.30 for three).",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(SUPPORTED_FORMATS),
        help="Any of: pdf svg png (space- or comma-separated).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Statistics sidecar path (default: <output-stem>_summary.csv).",
    )
    parser.add_argument(
        "--strict-arms",
        action="store_true",
        help="Fail instead of warning when a metrics directory has an unknown arm.",
    )
    args = parser.parse_args()

    if args.width <= 0:
        parser.error("--width must be positive")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    formats = _parse_formats(args.formats)
    rows = read_metrics(
        args.run_root,
        strict_unknown_runs=args.strict_arms,
        plot_min_tokens=int(args.plot_min_tokens),
    )
    if not rows:
        raise RuntimeError(f"no train/eval metrics found under {args.run_root}")

    height = args.height
    if height is None:
        if args.layout == "vertical":
            height = 5.8 if len(args.panels) >= 3 else 4.0
        else:
            height = 2.30 if len(args.panels) >= 3 else 2.15

    written, summaries = plot_curves(
        rows,
        title=args.title,
        output=args.output,
        dpi=args.dpi,
        panels=args.panels,
        uncertainty=args.uncertainty,
        token_unit=args.token_unit,
        ppl_scale=args.ppl_scale,
        width=args.width,
        height=height,
        formats=formats,
        x_min=args.x_min,
        layout=args.layout,
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
