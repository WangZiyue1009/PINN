"""Joint and marginal error distributions across PINN variants.

The main axis shows every test sample in feasibility--optimality error space.
The top and right axes summarize the corresponding one-dimensional marginal
distributions. Feasibility values are clipped at 1e-8 before both plotting and
marginal-density estimation; the display floor is labelled as zero.
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


# Publication settings. SVG text is configured as editable even though this
# revision intentionally exports PDF only.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["svg.fonttype"] = "none"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "add_theta_test.csv"
OUT_DIR = PROJECT_DIR / "figures_converged"
OUTPUT_PATH = OUT_DIR / "add_theta_joint_error.pdf"
FEAS_DISPLAY_FLOOR = 1e-8
OPT_DISPLAY_FLOOR = 1e-8
FONT_SIZE = 7.6

GROUP_ORDER = ("PINN-3", "PINN-5", "PINN-7")
COLORS = {
    "PINN-3": "#355C9A",  # blue
    "PINN-5": "#D07A1F",  # orange
    "PINN-7": "#4A9C84",  # teal
}
MARKERS = {"PINN-3": "o", "PINN-5": "^", "PINN-7": "D"}
DARK = "#272727"
MAX_LABEL_OFFSETS = {
    "PINN-3": (-12, 0),
    "PINN-5": (-10, -16),
    "PINN-7": (-20, 16),
}

MAX_LABEL_ALIGNMENTS = {
    "PINN-3": ("right", "center"),
    "PINN-5": ("right", "top"),
    "PINN-7": ("right", "bottom"),
}


def apply_publication_style() -> None:
    """Apply a compact single-column Nature-style visual system."""
    plt.rcParams.update(
        {
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
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_data() -> dict[str, np.ndarray]:
    """Load and validate the three PINN-variant groups."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Source data not found: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {"mode", "error_feas", "error_opt"}
    if not rows:
        raise ValueError("The source-data table is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing source-data columns: {sorted(missing)}")

    groups: dict[str, list[tuple[float, float]]] = {
        group: [] for group in GROUP_ORDER
    }
    for row in rows:
        group = row["mode"].strip()
        if group not in groups:
            continue
        groups[group].append(
            (float(row["error_feas"]), float(row["error_opt"]))
        )

    arrays: dict[str, np.ndarray] = {}
    for group in GROUP_ORDER:
        values = np.asarray(groups[group], dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
            raise ValueError(f"Insufficient two-dimensional data for {group}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite errors found for {group}")
        if np.any(values <= 0):
            raise ValueError(f"Errors must be positive for {group}")
        # Apply the same display truncation before scatter and density estimation.
        values[:, 0] = np.maximum(values[:, 0], FEAS_DISPLAY_FLOOR)
        values[:, 1] = np.maximum(values[:, 1], OPT_DISPLAY_FLOOR)
        arrays[group] = values

    return arrays


def robust_bandwidth(values: np.ndarray, minimum: float) -> float:
    """Scott-type bandwidth using a robust scale to limit outlier influence."""
    values = np.asarray(values, dtype=float)
    standard_scale = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    robust_scale = float((q75 - q25) / 1.349)
    valid_scales = [scale for scale in (standard_scale, robust_scale) if scale > 0]
    scale = min(valid_scales) if valid_scales else minimum
    return max(minimum, 1.06 * scale * len(values) ** (-1.0 / 5.0))


def kde_1d(
    values: np.ndarray, grid: np.ndarray, minimum_bandwidth: float
) -> np.ndarray:
    """Evaluate a one-dimensional Gaussian KDE using the robust bandwidth."""
    bandwidth = robust_bandwidth(values, minimum=minimum_bandwidth)
    distance = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * distance**2).sum(axis=1)
    density /= np.sqrt(2.0 * np.pi) * len(values) * bandwidth
    return density


def format_maximum(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.2f}\!\times\!10^{{{exponent}}}$"


def annotate_maximum(ax: plt.Axes, group: str, values: np.ndarray) -> None:
    max_index = int(np.argmax(values[:, 0]))
    max_x, max_y = values[max_index]
    offset_x, offset_y = MAX_LABEL_OFFSETS[group]
    horizontal_alignment, vertical_alignment = MAX_LABEL_ALIGNMENTS[group]
    color = COLORS[group]
    marker = MARKERS[group]

    ax.scatter(
        [max_x],
        [max_y],
        s=38,
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
        s=25,
        marker=marker,
        facecolor=color,
        edgecolor=DARK,
        linewidth=0.5,
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
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=color,
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "linewidth": 0.55,
            "shrinkA": 1.5,
            "shrinkB": 4.0,
        },
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.35},
        annotation_clip=True,
        zorder=7,
    )


def plot_group(
    ax: plt.Axes,
    group: str,
    values: np.ndarray,
) -> None:
    """Plot the raw solid observations for one PINN configuration."""
    color = COLORS[group]
    marker = MARKERS[group]

    ax.scatter(
        values[:, 0],
        values[:, 1],
        s=7.0,
        marker=marker,
        facecolor=color,
        edgecolor="white",
        linewidth=0.18,
        alpha=0.40,
        zorder=3,
    )


def make_figure(groups: dict[str, np.ndarray]) -> plt.Figure:
    apply_publication_style()

    # Match the converged ablation figures while retaining all observations.
    x_min, x_max = FEAS_DISPLAY_FLOOR, 1e-2
    y_min, y_max = OPT_DISPLAY_FLOOR, 0.5

    # The appended marginal axes retain approximately the same main-panel area
    # as the previous compact single-panel version.
    fig, ax = plt.subplots(figsize=(118 / 25.4, 98 / 25.4))
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="20%", pad=0.0, sharex=ax)
    ax_right = divider.append_axes("right", size="20%", pad=0.0, sharey=ax)
    x_log_grid = np.linspace(np.log10(x_min), np.log10(x_max), 320)
    y_grid = np.linspace(y_min, y_max, 260)

    for group in GROUP_ORDER:
        plot_group(ax, group, groups[group])

    # Marginal distributions use exactly the same display-clipped values as
    # the scatter.
    for group in GROUP_ORDER:
        color = COLORS[group]
        feasibility_log = np.log10(groups[group][:, 0])
        optimality = groups[group][:, 1]
        feasibility_density = kde_1d(
            feasibility_log, x_log_grid, minimum_bandwidth=0.16
        )
        optimality_density = kde_1d(
            optimality, y_grid, minimum_bandwidth=0.006
        )
        ax_top.plot(
            10.0**x_log_grid,
            feasibility_density,
            color=color,
            linewidth=1.15,
            zorder=3,
        )
        ax_top.fill_between(
            10.0**x_log_grid,
            feasibility_density,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
        ax_right.plot(
            optimality_density,
            y_grid,
            color=color,
            linewidth=1.15,
            zorder=3,
        )
        ax_right.fill_betweenx(
            y_grid,
            optimality_density,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    major_ticks = [1e-8, 1e-6, 1e-4, 1e-2]
    major_labels = ["0", r"$10^{-6}$", r"$10^{-4}$", r"$10^{-2}$"]
    ax.set_xticks(major_ticks, labels=major_labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticks(
        [OPT_DISPLAY_FLOOR, 0.1, 0.2, 0.3, 0.4, 0.5],
        labels=["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
    )

    ax.set_xlabel("Feasibility error")
    ax.set_ylabel("Optimality error")
    ax.tick_params(direction="out", colors=DARK)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK)
    ax.grid(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[group],
            linestyle="none",
            markerfacecolor=COLORS[group],
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=4.2,
            label=group,
        )
        for group in GROUP_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=1,
        handletextpad=0.3,
        columnspacing=0.65,
        borderaxespad=0.35,
        labelcolor=DARK,
    )

    for group in GROUP_ORDER:
        annotate_maximum(ax, group, groups[group])

    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(
        axis="x", which="both", bottom=False, labelbottom=False
    )
    ax_top.tick_params(
        axis="y", which="both", left=False, labelleft=False
    )
    ax_top.set_yticks([])
    ax_top.set_ylabel("Density", labelpad=2.0)
    for side in ("top", "right", "left"):
        ax_top.spines[side].set_visible(False)
    ax_top.spines["bottom"].set_visible(True)
    ax_top.spines["bottom"].set_color(DARK)
    ax_top.spines["bottom"].set_linewidth(0.65)

    ax_right.set_xlim(left=0)
    ax_right.tick_params(
        axis="y", which="both", left=False, labelleft=False
    )
    ax_right.tick_params(
        axis="x", which="both", bottom=False, labelbottom=False
    )
    ax_right.set_xticks([])
    ax_right.set_xlabel("Density", labelpad=2.0)
    for side in ("top", "right", "bottom"):
        ax_right.spines[side].set_visible(False)
    ax_right.spines["left"].set_visible(True)
    ax_right.spines["left"].set_color(DARK)
    ax_right.spines["left"].set_linewidth(0.65)

    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.15, top=0.88)
    return fig


def save_pdf(fig: plt.Figure) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    groups = load_data()
    fig = make_figure(groups)
    save_pdf(fig)
    print("Created figures_converged/add_theta_joint_error.pdf")


if __name__ == "__main__":
    main()
