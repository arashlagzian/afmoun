from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


OUT = Path("outputs/paper_figures/fig_cap_scale_geometry")

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
CHARCOAL = "#374151"

# Two-coordinate slice used only for visualization. With d=16, sqrt(d)=4, so
# c=1 is sign-like, c=3 visibly clips the row-l2 ball, and c=5 is mostly
# row-RMS-like because the box contains the l2 ball.
VIS_DIM = 16
RADIUS = math.sqrt(VIS_DIM)
CAPS = [1, 3, 5]
SCALES = [0.25, 0.5, 0.75, 1.0]
DEFAULT_C = 3
DEFAULT_S = 0.5


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "STIX Two Text",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "font.size": 7.5,
            "mathtext.fontset": "stix",
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_arrow(ax, start, end, color, *, lw=1.2, alpha=1.0, zorder=6):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def lmo_2d_positive(b: np.ndarray, radius: float, cap: float) -> np.ndarray:
    xs = np.linspace(0.0, min(cap, radius), 6000)
    best = np.zeros(2)
    best_val = -np.inf
    for x in xs:
        y_circle = math.sqrt(max(radius * radius - x * x, 0.0))
        for y in (0.0, min(cap, y_circle), cap):
            if x * x + y * y <= radius * radius + 1e-12 and y <= cap + 1e-12:
                val = b[0] * x + b[1] * y
                if val > best_val:
                    best_val = val
                    best = np.array([x, y])
    return best


def add_intersection_fill(ax, cap: float, *, color: str, alpha: float, zorder: int) -> None:
    rect = Rectangle(
        (-cap, -cap),
        2 * cap,
        2 * cap,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )
    rect.set_clip_path(Circle((0, 0), RADIUS, transform=ax.transData))
    ax.add_patch(rect)


def draw_panel(ax, cap: float) -> None:
    is_default = cap == DEFAULT_C
    ax.set_aspect("equal")
    ax.set_xlim(-5.25, 5.25)
    ax.set_ylim(-4.65, 4.65)
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([-4, 0, 4])
    ax.tick_params(length=2.0, width=0.55, pad=1.2)
    ax.axhline(0, color=LIGHT_GRAY, lw=0.55, zorder=0)
    ax.axvline(0, color=LIGHT_GRAY, lw=0.55, zorder=0)
    ax.set_xlabel(r"$u_1$", labelpad=1)
    if ax is not None:
        ax.set_ylabel(r"$u_2$", labelpad=1)

    title_color = GREEN if is_default else CHARCOAL
    title = rf"$c={int(cap)}$" + (r" default" if is_default else "")
    ax.set_title(title, color=title_color, fontweight="semibold", pad=2.5)

    # Green is exactly the feasible intersection: row-l2 ball intersected with
    # output-side coordinate cap.
    add_intersection_fill(ax, cap, color=GREEN if is_default else GRAY, alpha=0.24 if is_default else 0.12, zorder=1)

    circle_color = BLUE if is_default else GRAY
    box_color = ORANGE if is_default else GRAY
    circle_lw = 1.55 if is_default else 1.05
    box_lw = 1.55 if is_default else 1.05
    box_ls = "-" if is_default else (0, (3, 2))

    ax.add_patch(Circle((0, 0), RADIUS, fill=False, edgecolor=circle_color, lw=circle_lw, zorder=4))
    ax.add_patch(
        Rectangle(
            (-cap, -cap),
            2 * cap,
            2 * cap,
            fill=False,
            edgecolor=box_color,
            linewidth=box_lw,
            linestyle=box_ls,
            zorder=5,
        )
    )

    if is_default:
        ax.text(-4.9, 4.15, r"input lookup: $\|u\|_2\leq\sqrt{d}$", color=BLUE, fontsize=6.7)
        ax.text(-4.9, 3.45, r"output cap: $\|u\|_\infty\leq c$", color=ORANGE, fontsize=6.7)
        ax.text(-2.72, -2.62, r"shared feasible" + "\n" + r"intersection", color="#007A59", fontsize=7.2, ha="center")

        b = np.array([1.0, 0.72])
        u_star = lmo_2d_positive(b, RADIUS, cap)
        b_dir = 1.80 * b / np.linalg.norm(b)
        add_arrow(ax, (0, 0), b_dir, CHARCOAL, lw=0.9, alpha=0.72, zorder=6)
        ax.text(b_dir[0] + 0.12, b_dir[1] - 0.08, r"$b$", color=CHARCOAL, fontsize=7.0)

        for s in SCALES:
            end = s * u_star
            default_s = abs(s - DEFAULT_S) < 1e-12
            color = PURPLE if default_s else GRAY
            lw = 2.25 if default_s else 0.95
            alpha = 1.0 if default_s else 0.42
            add_arrow(ax, (0, 0), end, color, lw=lw, alpha=alpha, zorder=8 if default_s else 7)
            if s in (0.25, 0.75, 1.0):
                ax.text(end[0] + 0.11, end[1] + 0.02, rf"$s={s:g}$", color=GRAY, fontsize=6.2)

        end = DEFAULT_S * u_star
        ax.text(end[0] + 0.12, end[1] - 0.20, r"$s=0.5$", color=PURPLE, fontsize=6.8, fontweight="semibold")
        ax.plot([u_star[0]], [u_star[1]], "o", color=GREEN, ms=3.4, zorder=9)
        ax.text(u_star[0] + 0.12, u_star[1] + 0.02, r"$u^\star=\mathcal{L}_3(b)$", color=GREEN, fontsize=6.8, fontweight="semibold")
    else:
        ax.text(
            -4.65,
            -4.22,
            "gray: ablation setting",
            color=GRAY,
            fontsize=6.2,
        )


def main() -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.55), sharey=True)

    for ax, cap in zip(axes, CAPS):
        draw_panel(ax, cap)
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels([])

    fig.text(
        0.5,
        0.035,
        r"Each panel shows $\mathcal{C}_{\mathrm{tie},c}=\{u:\|u\|_2\leq\sqrt{d},\ \|u\|_\infty\leq c\}$; "
        r"$s$ rescales the selected direction but does not change the intersection.",
        ha="center",
        va="center",
        fontsize=7.1,
        color=CHARCOAL,
    )
    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.20, top=0.86, wspace=0.10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")
    print(f"wrote {OUT}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
