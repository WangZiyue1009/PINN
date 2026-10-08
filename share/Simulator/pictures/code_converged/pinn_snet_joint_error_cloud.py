"""Joint and marginal estimation-error comparison for PINN and SNET.

Both methods are overlaid in one feasibility--optimality plane so their trade-
off is read on identical axes. The top and right axes show the corresponding
marginal distributions. Feasibility errors below 1e-8 are clipped before both
scatter plotting and density estimation, and the display floor is labelled 0.
Only one PDF is exported.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "SNET_error.csv"
OUT_DIR = PROJECT_DIR / "figures_converged"
OUTPUT_PATH = OUT_DIR / "pinn_snet_joint_error_cloud.pdf"

GROUP_ORDER = ("fullnet", "snet")
GROUP_LABELS = {"fullnet": "PINN", "snet": "SNET"}
COLORS = {"fullnet": "#355C9A", "snet": "#D07A1F"}
MARKERS = {"fullnet": "o", "snet": "^"}
LABEL_OFFSETS = {"fullnet": (-8, 12), "snet": (-8, -14)}
LABEL_ALIGNMENTS = {"fullnet": ("right", "bottom"), "snet": ("right", "top")}

FEAS_DISPLAY_FLOOR = 1e-8
FEAS_DISPLAY_MAX = 1e-1
OPT_DISPLAY_MAX = 0.7
FONT_SIZE = 7.6
DARK = "#272727"


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_data() -> dict[str, np.ndarray]:
    required = {"mode", "error_feas", "error_opt"}
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Source data not found: {DATA_PATH}")
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"Missing source-data columns: {missing}")
        rows = list(reader)

    groups: dict[str, list[tuple[float, float]]] = {
        group: [] for group in GROUP_ORDER
    }
    for row in rows:
        group = row["mode"].strip().lower()
        if group not in groups:
            continue
        feasibility = max(float(row["error_feas"]), FEAS_DISPLAY_FLOOR)
        optimality = max(float(row["error_opt"]), 0.0)
        groups[group].append((feasibility, optimality))

    arrays: dict[str, np.ndarray] = {}
    for group in GROUP_ORDER:
        values = np.asarray(groups[group], dtype=float)
        if values.shape != (400, 2):
            raise ValueError(f"Expected 400 two-dimensional records for {group}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite errors found for {group}")
        arrays[group] = values
    return arrays


def robust_bandwidth(values: np.ndarray, minimum: float) -> float:
    values = np.asarray(values, dtype=float)
    standard_scale = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    robust_scale = float((q75 - q25) / 1.349)
    scales = [scale for scale in (standard_scale, robust_scale) if scale > 0]
    scale = min(scales) if scales else minimum
    return max(minimum, 1.06 * scale * len(values) ** (-1.0 / 5.0))


def kde_1d(
    values: np.ndarray, grid: np.ndarray, minimum_bandwidth: float
) -> np.ndarray:
    bandwidth = robust_bandwidth(values, minimum_bandwidth)
    distance = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * distance**2).sum(axis=1)
    density /= np.sqrt(2.0 * np.pi) * len(values) * bandwidth
    return density


def format_scientific(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.2f}\!\times\!10^{{{exponent}}}$"


def annotate_maximum(ax: plt.Axes, group: str, values: np.ndarray) -> None:
    index = int(np.argmax(values[:, 0]))
    x, y = values[index]
    color = COLORS[group]
    marker = MARKERS[group]
    horizontal_alignment, vertical_alignment = LABEL_ALIGNMENTS[group]

    ax.scatter(
        [x],
        [y],
        s=40,
        marker=marker,
        facecolor=color,
        edgecolor="white",
        linewidth=1.35,
        clip_on=False,
        zorder=7,
    )
    ax.scatter(
        [x],
        [y],
        s=25,
        marker=marker,
        facecolor=color,
        edgecolor=DARK,
        linewidth=0.5,
        clip_on=False,
        zorder=8,
    )
    ax.annotate(
        rf"max $e_{{\mathrm{{feas}}}}$ = {format_scientific(float(x))}",
        xy=(x, y),
        xytext=LABEL_OFFSETS[group],
        textcoords="offset points",
        ha=horizontal_alignment,
        va=vertical_alignment,
        color=color,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.55,
            "shrinkA": 1.0,
            "shrinkB": 4.0,
        },
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.3},
        annotation_clip=True,
        zorder=9,
    )


def style_main_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(FEAS_DISPLAY_FLOOR, FEAS_DISPLAY_MAX)
    ax.set_ylim(0.0, OPT_DISPLAY_MAX)
    ax.set_xticks(
        [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        labels=[
            "0",
            r"$10^{-7}$",
            r"$10^{-6}$",
            r"$10^{-5}$",
            r"$10^{-4}$",
            r"$10^{-3}$",
            r"$10^{-2}$",
            r"$10^{-1}$",
        ],
    )
    y_ticks = np.arange(0.0, OPT_DISPLAY_MAX + 0.001, 0.1)
    ax.set_yticks(y_ticks, labels=["0" if value == 0 else f"{value:.1f}" for value in y_ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Feasibility error")
    ax.set_ylabel("Optimality error")
    ax.grid(False)
    ax.tick_params(direction="out", colors=DARK, which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK)
        spine.set_linewidth(0.65)


def make_figure(groups: dict[str, np.ndarray]) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(132 / 25.4, 108 / 25.4))
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="20%", pad=0.0, sharex=ax)
    ax_right = divider.append_axes("right", size="20%", pad=0.0, sharey=ax)

    log_grid = np.linspace(np.log10(FEAS_DISPLAY_FLOOR), np.log10(FEAS_DISPLAY_MAX), 400)
    opt_grid = np.linspace(0.0, OPT_DISPLAY_MAX, 400)
    for group in GROUP_ORDER:
        values = groups[group]
        color = COLORS[group]
        marker = MARKERS[group]
        ax.scatter(
            values[:, 0],
            values[:, 1],
            s=10,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.20,
            alpha=0.43,
            zorder=3,
        )

        feasibility_density = kde_1d(
            np.log10(values[:, 0]), log_grid, minimum_bandwidth=0.16
        )
        optimality_density = kde_1d(
            values[:, 1], opt_grid, minimum_bandwidth=0.007
        )
        ax_top.plot(
            10.0**log_grid,
            feasibility_density,
            color=color,
            linewidth=1.2,
            zorder=3,
        )
        ax_top.fill_between(
            10.0**log_grid,
            feasibility_density,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )
        ax_right.plot(
            optimality_density,
            opt_grid,
            color=color,
            linewidth=1.2,
            zorder=3,
        )
        ax_right.fill_betweenx(
            opt_grid,
            optimality_density,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )

    style_main_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[group],
            linestyle="none",
            markerfacecolor=COLORS[group],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=5.0,
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        handletextpad=0.35,
        borderaxespad=0.35,
    )
    for group in GROUP_ORDER:
        annotate_maximum(ax, group, groups[group])

    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_top.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_top.set_yticks([])
    ax_top.set_ylabel("Density", labelpad=2.0)
    for side in ("top", "right", "left"):
        ax_top.spines[side].set_visible(False)
    ax_top.spines["bottom"].set_visible(True)
    ax_top.spines["bottom"].set_color(DARK)
    ax_top.spines["bottom"].set_linewidth(0.65)

    ax_right.set_xlim(left=0)
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_right.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_right.set_xticks([])
    ax_right.set_xlabel("Density", labelpad=2.0)
    for side in ("top", "right", "bottom"):
        ax_right.spines[side].set_visible(False)
    ax_right.spines["left"].set_visible(True)
    ax_right.spines["left"].set_color(DARK)
    ax_right.spines["left"].set_linewidth(0.65)

    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.15, top=0.88)
    return fig


def main() -> None:
    groups = load_data()
    figure = make_figure(groups)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(f"Created {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
