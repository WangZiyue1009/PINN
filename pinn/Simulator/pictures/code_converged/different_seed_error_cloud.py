"""Compare feasibility--optimality error distributions under four random seeds."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT.parent / "data" / "different_seed.csv"
OUT_DIR = ROOT / "figures_converged"
OUTPUT_PATH = OUT_DIR / "different_seed_error_cloud.pdf"

GROUP_ORDER = ("seed0", "seed6", "seed66", "seed666")
COLORS = {
    "seed0": "#355C9A",
    "seed6": "#D07A1F",
    "seed66": "#4A9C84",
    "seed666": "#8064A2",
}
MARKERS = {"seed0": "o", "seed6": "^", "seed66": "D", "seed666": "s"}
LEGEND_LABELS = {
    "seed0": "Seed = 0",
    "seed6": "Seed = 6",
    "seed66": "Seed = 66",
    "seed666": "Seed = 666",
}
DARK = "#272727"
MAX_LABEL_OFFSETS = {
    "seed0": (-18, 12),
    "seed6": (0, -34),
    "seed66": (-18, -12),
    "seed666": (0, 34),
}
MAX_LABEL_ALIGNMENTS = {
    "seed0": ("right", "bottom"),
    "seed6": ("center", "top"),
    "seed66": ("right", "top"),
    "seed666": ("center", "bottom"),
}

# Match plot_test_error_cloud.py: feasibility values below the display floor
# are clipped before scatter and density estimation; the floor tick reads "0".
FEAS_DISPLAY_FLOOR = 1e-8
FEAS_DISPLAY_MAX = 1e-1
OPT_DISPLAY_MIN = 0.0
OPT_DISPLAY_MAX = 0.5

LOG_FEAS_GRID = np.linspace(
    np.log10(FEAS_DISPLAY_FLOOR), np.log10(FEAS_DISPLAY_MAX), 400
)
OPT_GRID = np.linspace(OPT_DISPLAY_MIN, OPT_DISPLAY_MAX, 400)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13.5,
        "axes.labelsize": 14.4,
        "xtick.labelsize": 12.6,
        "ytick.labelsize": 12.6,
        "legend.fontsize": 14.4,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def load_data() -> dict[str, np.ndarray]:
    """Read all seed groups and apply the shared feasibility display floor."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Source data not found: {DATA_PATH}")

    groups: dict[str, list[tuple[float, float]]] = {
        mode: [] for mode in GROUP_ORDER
    }
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {DATA_PATH.name}")
        required = {"mode", "error_feas", "error_opt"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {DATA_PATH.name}: {sorted(missing)}")

        for row in reader:
            mode = row["mode"].strip().lower()
            if mode not in groups:
                continue
            try:
                feasibility = max(float(row["error_feas"]), FEAS_DISPLAY_FLOOR)
                optimality = float(row["error_opt"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(feasibility) and np.isfinite(optimality) and optimality > 0:
                groups[mode].append((feasibility, optimality))

    arrays: dict[str, np.ndarray] = {}
    for mode in GROUP_ORDER:
        values = np.asarray(groups[mode], dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
            raise ValueError(f"Insufficient two-dimensional data for {mode}")
        if np.any(values[:, 0] > FEAS_DISPLAY_MAX):
            raise ValueError(f"{mode} feasibility error exceeds the x-axis maximum")
        if np.any(values[:, 1] > OPT_DISPLAY_MAX):
            raise ValueError(f"{mode} optimality error exceeds the y-axis maximum")
        arrays[mode] = values
    return arrays


def feasibility_density(feasibility: np.ndarray) -> np.ndarray:
    """KDE of display-clipped feasibility errors in log10 space."""
    kde = gaussian_kde(np.log10(feasibility), bw_method="scott")
    return kde(LOG_FEAS_GRID)


def optimality_density(optimality: np.ndarray) -> np.ndarray:
    """KDE of optimality errors in linear space."""
    kde = gaussian_kde(optimality, bw_method="scott")
    return kde(OPT_GRID)


def format_maximum(value: float) -> str:
    """Format a maximum feasibility error in scientific notation."""
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.2f}\!\times\!10^{{{exponent}}}$"


def annotate_maximum(ax: plt.Axes, mode: str, values: np.ndarray) -> None:
    """Highlight and label one seed's maximum feasibility-error point."""
    max_index = int(np.argmax(values[:, 0]))
    max_x, max_y = values[max_index]
    offset_x, offset_y = MAX_LABEL_OFFSETS[mode]
    horizontal_alignment, vertical_alignment = MAX_LABEL_ALIGNMENTS[mode]
    color = COLORS[mode]
    marker = MARKERS[mode]

    ax.scatter(
        [max_x],
        [max_y],
        s=90,
        marker=marker,
        facecolor=color,
        edgecolor="white",
        linewidth=1.4,
        clip_on=False,
        zorder=5,
    )
    ax.scatter(
        [max_x],
        [max_y],
        s=60,
        marker=marker,
        facecolor=color,
        edgecolor=DARK,
        linewidth=0.6,
        clip_on=False,
        zorder=6,
    )
    ax.annotate(
        rf"max $e_{{\mathrm{{feas}}}}$ = {format_maximum(float(max_x))}",
        xy=(max_x, max_y),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        ha=horizontal_alignment,
        va=vertical_alignment,
        fontweight="bold",
        color=color,
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.8,
            "shrinkA": 1.5,
            "shrinkB": 5.0,
        },
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.35},
        annotation_clip=True,
        zorder=7,
    )


def style_main_axes(ax: plt.Axes) -> None:
    """Apply the visual system of plot_test_error_cloud.py."""
    ax.set_xscale("log")
    ax.set_xlim(FEAS_DISPLAY_FLOOR, FEAS_DISPLAY_MAX)
    ax.set_ylim(OPT_DISPLAY_MIN, OPT_DISPLAY_MAX)
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
    y_ticks = np.arange(OPT_DISPLAY_MIN, OPT_DISPLAY_MAX + 0.001, 0.1)
    ax.set_yticks(y_ticks, labels=[f"{value:.1f}" for value in y_ticks])
    ax.set_xlabel("Feasibility error")
    ax.set_ylabel("Optimality error")
    ax.tick_params(colors="#000000", direction="in", which="both", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.7)


def make_figure(groups: dict[str, np.ndarray]) -> plt.Figure:
    """Draw one joint-error panel with shared marginal distributions."""
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="22%", pad=0.0, sharex=ax)
    ax_right = divider.append_axes("right", size="22%", pad=0.0, sharey=ax)

    densities: dict[str, dict[str, np.ndarray]] = {}
    for mode in GROUP_ORDER:
        feasibility = groups[mode][:, 0]
        optimality = groups[mode][:, 1]
        densities[mode] = {
            "feas": feasibility_density(feasibility),
            "opt": optimality_density(optimality),
        }
        ax.scatter(
            feasibility,
            optimality,
            s=24,
            marker=MARKERS[mode],
            facecolor=COLORS[mode],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.32,
            zorder=4,
        )

    style_main_axes(ax)

    for mode in GROUP_ORDER:
        density = densities[mode]["feas"]
        ax_top.plot(10**LOG_FEAS_GRID, density, color=COLORS[mode], lw=1.6, zorder=3)
        ax_top.fill_between(
            10**LOG_FEAS_GRID,
            density,
            color=COLORS[mode],
            alpha=0.08,
            zorder=2,
        )
    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(labelbottom=False, left=False, labelleft=False)
    ax_top.set_yticks([])
    for side in ("top", "right", "left"):
        ax_top.spines[side].set_visible(False)
    ax_top.spines["bottom"].set_color("#333333")
    ax_top.spines["bottom"].set_linewidth(0.7)
    ax_top.set_ylabel("Density")

    for mode in GROUP_ORDER:
        density = densities[mode]["opt"]
        ax_right.plot(density, OPT_GRID, color=COLORS[mode], lw=1.6, zorder=3)
        ax_right.fill_betweenx(
            OPT_GRID,
            density,
            color=COLORS[mode],
            alpha=0.08,
            zorder=2,
        )
    ax_right.set_xlim(left=0)
    ax_right.tick_params(labelleft=False, bottom=False, labelbottom=False)
    ax_right.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax_right.spines[side].set_visible(False)
    ax_right.spines["left"].set_color("#333333")
    ax_right.spines["left"].set_linewidth(0.7)
    ax_right.set_xlabel("Density")

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[mode],
            linestyle="none",
            markerfacecolor=COLORS[mode],
            markeredgecolor="white",
            markersize=7,
            label=LEGEND_LABELS[mode],
        )
        for mode in GROUP_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
        frameon=False,
    )
    for mode in GROUP_ORDER:
        annotate_maximum(ax, mode, groups[mode])
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_data()
    fig = make_figure(groups)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Created random-seed error cloud: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
