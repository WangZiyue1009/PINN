"""Plot the ESS feasibility--optimality error cloud and marginal densities."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ess_error_results.csv"
OUT_DIR = ROOT / "figures_converged"
OUTPUT_PATH = OUT_DIR / "ess_error_cloud.pdf"

COLOR = "#1764AB"
MARKER = "o"

# Match plot_test_error_cloud.py: values below 1e-8 are clipped before both
# plotting and density estimation, and the lower-bound tick is labelled "0".
FEAS_DISPLAY_FLOOR = 1e-8
FEAS_DISPLAY_MAX = 1e-1
OPT_DISPLAY_MIN = 0.0
OPT_DISPLAY_MAX = 2.5

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


def load_data() -> np.ndarray:
    """Load the ESS results and apply the shared feasibility display floor."""
    values: list[tuple[float, float]] = []
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {DATA_PATH.name}")
        required = {"mode", "error_feas", "error_opt"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {DATA_PATH.name}: {sorted(missing)}")

        for row in reader:
            if row["mode"].strip().lower() != "ess":
                continue
            feasibility = max(float(row["error_feas"]), FEAS_DISPLAY_FLOOR)
            optimality = float(row["error_opt"])
            values.append((feasibility, optimality))

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
        raise ValueError("Insufficient two-dimensional ESS data")
    if not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError("ESS errors must be finite and positive after truncation")
    if np.any(array[:, 0] > FEAS_DISPLAY_MAX):
        raise ValueError("ESS feasibility error exceeds the configured x-axis maximum")
    if np.any(array[:, 1] > OPT_DISPLAY_MAX):
        raise ValueError("ESS optimality error exceeds the configured y-axis maximum")
    return array


def feasibility_density(feasibility: np.ndarray) -> np.ndarray:
    """KDE of display-clipped feasibility errors in log10 space."""
    kde = gaussian_kde(np.log10(feasibility), bw_method="scott")
    return kde(LOG_FEAS_GRID)


def optimality_density(optimality: np.ndarray) -> np.ndarray:
    """KDE of optimality errors in linear space."""
    kde = gaussian_kde(optimality, bw_method="scott")
    return kde(OPT_GRID)


def style_main_axes(ax: plt.Axes) -> None:
    """Apply the same axes styling used by plot_test_error_cloud.py."""
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
    y_ticks = np.arange(OPT_DISPLAY_MIN, OPT_DISPLAY_MAX + 0.001, 0.5)
    ax.set_yticks(y_ticks, labels=[f"{value:.1f}" for value in y_ticks])
    ax.set_xlabel("Feasibility error")
    ax.set_ylabel("Optimality error")
    ax.tick_params(colors="#000000", direction="in", which="both", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.7)


def make_figure(values: np.ndarray) -> plt.Figure:
    """Create the ESS joint error cloud with aligned marginal densities."""
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="22%", pad=0.0, sharex=ax)
    ax_right = divider.append_axes("right", size="22%", pad=0.0, sharey=ax)

    feasibility = values[:, 0]
    optimality = values[:, 1]
    density_feasibility = feasibility_density(feasibility)
    density_optimality = optimality_density(optimality)

    ax.scatter(
        feasibility,
        optimality,
        s=32,
        marker=MARKER,
        facecolor=COLOR,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        zorder=4,
    )
    style_main_axes(ax)

    ax_top.plot(10**LOG_FEAS_GRID, density_feasibility, color=COLOR, lw=1.6, zorder=3)
    ax_top.fill_between(
        10**LOG_FEAS_GRID, density_feasibility, color=COLOR, alpha=0.18, zorder=2
    )
    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(labelbottom=False, left=False, labelleft=False)
    ax_top.set_yticks([])
    for side in ("top", "right", "left"):
        ax_top.spines[side].set_visible(False)
    ax_top.spines["bottom"].set_color("#333333")
    ax_top.spines["bottom"].set_linewidth(0.7)
    ax_top.set_ylabel("Density")

    ax_right.plot(density_optimality, OPT_GRID, color=COLOR, lw=1.6, zorder=3)
    ax_right.fill_betweenx(
        OPT_GRID, density_optimality, color=COLOR, alpha=0.18, zorder=2
    )
    ax_right.set_xlim(left=0)
    ax_right.tick_params(labelleft=False, bottom=False, labelbottom=False)
    ax_right.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax_right.spines[side].set_visible(False)
    ax_right.spines["left"].set_color("#333333")
    ax_right.spines["left"].set_linewidth(0.7)
    ax_right.set_xlabel("Density")

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = load_data()
    fig = make_figure(values)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Created ESS error cloud: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
