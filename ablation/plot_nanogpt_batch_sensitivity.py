from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence


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
    "train_loss": ("train_loss", "Training loss"),
    "eval_loss": ("eval_loss", "Validation loss"),
    "eval_ppl": ("eval_ppl", "Validation perplexity"),
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


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("method") or not row.get("tokens_per_step"):
                continue
            try:
                row["tokens_per_step"] = int(float(row["tokens_per_step"]))
                row["batch_size"] = int(float(row["batch_size"]))
                for key, _ in PANEL_SPECS.values():
                    row[key] = float(row[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric value in {path}: {row}") from exc
            rows.append(row)
    return rows


def tick_label(tokens_per_step: int) -> str:
    if tokens_per_step % 1024 == 0:
        return f"{tokens_per_step // 1024}k"
    return str(tokens_per_step)


def write_plot_summary(path: Path, rows: Sequence[dict], panels: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("panel", "method", "batch_size", "tokens_per_step", "value"),
        )
        writer.writeheader()
        for panel in panels:
            key, _ = PANEL_SPECS[panel]
            for row in rows:
                writer.writerow(
                    {
                        "panel": panel,
                        "method": row["method"],
                        "batch_size": row["batch_size"],
                        "tokens_per_step": row["tokens_per_step"],
                        "value": f"{float(row[key]):.12g}",
                    }
                )


def plot(
    rows: Sequence[dict],
    *,
    output: Path,
    panels: Sequence[str],
    width: float,
    height: float,
    dpi: int,
    formats: Sequence[str],
) -> list[Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    by_method: dict[str, list[dict]] = {method: [] for method in ARM_ORDER}
    for row in rows:
        method = row["method"]
        if method in by_method:
            by_method[method].append(row)
    for method in by_method:
        by_method[method].sort(key=lambda row: row["tokens_per_step"])

    x_values = sorted({int(row["tokens_per_step"]) for row in rows})
    x_positions = {value: math.log2(value) for value in x_values}
    x_ticks = [x_positions[value] for value in x_values]
    x_labels = [tick_label(value) for value in x_values]

    with mpl.rc_context(paper_rcparams()):
        fig, axes = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False)
        axes_1d = axes.ravel()
        legend_handles: dict[str, object] = {}

        for panel_index, (ax, panel) in enumerate(zip(axes_1d, panels)):
            key, title = PANEL_SPECS[panel]
            for method in ARM_ORDER:
                method_rows = by_method.get(method, [])
                if not method_rows:
                    continue
                x = [x_positions[int(row["tokens_per_step"])] for row in method_rows]
                y = [float(row[key]) for row in method_rows]
                color = ARM_COLORS[method]
                (line,) = ax.plot(
                    x,
                    y,
                    label=method,
                    color=color,
                    linestyle=ARM_LINESTYLES[method],
                    linewidth=1.75 if method == "AF-Muon" else 1.55,
                    marker=ARM_MARKERS[method],
                    markerfacecolor=ARM_MARKER_FACES[method],
                    markeredgecolor=color,
                    markeredgewidth=0.65,
                    markersize=3.2,
                    solid_capstyle="round",
                    dash_capstyle="round",
                    zorder=3 if method == "AF-Muon" else 2,
                )
                legend_handles.setdefault(method, line)

            panel_letter = chr(ord("a") + panel_index)
            ax.set_title(f"({panel_letter}) {title}", loc="left", pad=3.0)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlim(min(x_ticks) - 0.22, max(x_ticks) + 0.22)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
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
        fig.supxlabel("Tokens per optimizer update", x=0.54, y=0.045, fontsize=8.0)
        ordered_handles = [legend_handles[m] for m in ARM_ORDER if m in legend_handles]
        ordered_labels = [m for m in ARM_ORDER if m in legend_handles]
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
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot NanoGPT batch-size sensitivity from a matched-token CSV."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=tuple(PANEL_SPECS),
        default=list(DEFAULT_PANELS),
    )
    parser.add_argument("--width", type=float, default=5.5)
    parser.add_argument("--height", type=float, default=2.30)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", nargs="+", default=list(SUPPORTED_FORMATS))
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise RuntimeError(f"no rows found in {args.csv}")
    formats = _parse_formats(args.formats)
    written = plot(
        rows,
        output=args.output,
        panels=args.panels,
        width=float(args.width),
        height=float(args.height),
        dpi=int(args.dpi),
        formats=formats,
    )

    output_stem = (
        args.output.with_suffix("")
        if args.output.suffix.lower().lstrip(".") in SUPPORTED_FORMATS
        else args.output
    )
    summary_path = args.summary_csv or output_stem.with_name(
        f"{output_stem.name}_summary.csv"
    )
    write_plot_summary(summary_path, rows, args.panels)

    for path in written:
        print(f"wrote {path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
