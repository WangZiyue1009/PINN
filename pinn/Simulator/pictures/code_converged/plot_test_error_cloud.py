"""Compare interpolation and extrapolation feasibility-optimality error clouds."""

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
DATA_PATH = ROOT / "data" / "test.csv"
OUT_DIR = ROOT / "figures_converged"
#1764AB
COLORS = {"interpolation": "#1764AB", "extrapolation": "#D17A22"}
MARKERS = {"interpolation": "o", "extrapolation": "^"}
LEGEND_LABELS = {
    "interpolation": "In-distribution test",
    "extrapolation": "Out-of-distribution test",
}
GROUP_ORDER = ("extrapolation","interpolation")

# Values below the display floor are shown at the left boundary, whose tick is
# labelled "0". A true zero cannot be placed on a logarithmic axis.
FEAS_DISPLAY_FLOOR = 1e-8
FEAS_DISPLAY_MAX = 1e-1

# Evaluation grids: feasibility in log10 space, optimality in linear space.
LOG_FEAS_GRID = np.linspace(np.log10(FEAS_DISPLAY_FLOOR), np.log10(FEAS_DISPLAY_MAX), 400)
OPT_GRID = np.linspace(0.0, 0.90, 400)

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
    groups: dict[str, list[tuple[float, float]]] = {
        "interpolation": [],
        "extrapolation": [],
    }
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mode = row["mode"].strip().lower()
            if mode not in groups:
                continue
            # Apply the feasibility truncation once, before any scatter or
            # distribution calculation. The lower-bound value is displayed as
            # "0" on the logarithmic x-axis.
            feasibility = max(float(row["error_feas"]), FEAS_DISPLAY_FLOOR)
            groups[mode].append((feasibility, float(row["error_opt"])))

    arrays = {mode: np.asarray(values, dtype=float) for mode, values in groups.items()}
    for mode, values in arrays.items():
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
            raise ValueError(f"Insufficient two-dimensional data for {mode}")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{mode} errors must be finite and positive")
    return arrays


def feasibility_density(feasibility: np.ndarray) -> np.ndarray:
    """KDE of display-clipped feasibility in log10 space."""
    kde = gaussian_kde(np.log10(feasibility), bw_method="scott")
    return kde(LOG_FEAS_GRID)


def optimality_density(optimality: np.ndarray) -> np.ndarray:
    """KDE of optimality in linear space, evaluated on the optimality grid."""
    kde = gaussian_kde(optimality, bw_method="scott")
    return kde(OPT_GRID)


def style_main_axes(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xlim(FEAS_DISPLAY_FLOOR, FEAS_DISPLAY_MAX)
    ax.set_ylim(0.0, 0.90)
    ax.set_xticks(
        [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        labels=[
            "0", r"$10^{-7}$", r"$10^{-6}$", r"$10^{-5}$",
            r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"
        ],
    )
    y_ticks = np.arange(0.0, 0.901, 0.10)
    ax.set_yticks(y_ticks, labels=[f"{value:.1f}" for value in y_ticks])
    ax.set_xlabel("Feasibility error")
    ax.set_ylabel("Optimality error")
    ax.tick_params(colors="#000000", direction="in", which="both", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.7)


def plot_error_cloud(groups: dict[str, np.ndarray]) -> None:
    """Scatter in the main panel with marginal feasibility/optimality distributions."""
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="22%", pad=0.0, sharex=ax)
    ax_right = divider.append_axes("right", size="22%", pad=0.0, sharey=ax)

    densities = {}
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
            s=32,
            marker=MARKERS[mode],
            facecolor=COLORS[mode],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.8,
            zorder=4,
        )

    style_main_axes(ax)

    # Top marginal: feasibility distribution (shared log-x).
    for mode in GROUP_ORDER:
        density = densities[mode]["feas"]
        ax_top.plot(10**LOG_FEAS_GRID, density, color=COLORS[mode], lw=1.6, zorder=3)
        ax_top.fill_between(10**LOG_FEAS_GRID, density, color=COLORS[mode], alpha=0.18, zorder=2)
    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(labelbottom=False, left=False, labelleft=False)
    ax_top.set_yticks([])
    for side in ("top", "right", "left"):
        ax_top.spines[side].set_visible(False)
    ax_top.spines["bottom"].set_color("#333333")
    ax_top.spines["bottom"].set_linewidth(0.7)
    ax_top.set_ylabel("Density")

    # Right marginal: optimality distribution (shared linear-y).
    for mode in GROUP_ORDER:
        density = densities[mode]["opt"]
        ax_right.plot(density, OPT_GRID, color=COLORS[mode], lw=1.6, zorder=3)
        ax_right.fill_betweenx(OPT_GRID, density, color=COLORS[mode], alpha=0.18, zorder=2)
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
            markersize=8,
            label=LEGEND_LABELS[mode],
        )
        for mode in GROUP_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)

    fig.savefig(OUT_DIR / "test_error_cloud.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "test_error_cloud*.png",
        "test_error_cloud*.svg",
        "test_error_cloud*.tiff",
        "test_error_cloud*.csv",
        "test_error_cloud*.md",
    ):
        for old_output in OUT_DIR.glob(pattern):
            old_output.unlink()
    # remove the previous HDR-variant exports
    for stale in OUT_DIR.glob("test_error_cloud_*.pdf"):
        stale.unlink()
    groups = load_data()
    plot_error_cloud(groups)
    print(f"Created test-error cloud in: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
