"""Compare feasibility and optimality errors across three DER scales.

Feasibility errors at or below the display floor are placed at 1e-8 and the
corresponding tick is labelled as zero. The two metrics are exported as
standalone single-column PDFs in one dedicated folder.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "DER_error.csv"
OUT_DIR = ROOT / "figures_converged"
OUTPUT_DIR = OUT_DIR / "DER_boxplot_panels"
FEAS_OUTPUT_PATH = OUTPUT_DIR / "Feasibility error.pdf"
OPT_OUTPUT_PATH = OUTPUT_DIR / "Optimality error.pdf"
LEGACY_OUTPUT_PATH = OUT_DIR / "DER_boxplot.pdf"

FEAS_DISPLAY_FLOOR = 1e-8
FEAS_DISPLAY_MAX = 1e-2
OPT_DISPLAY_MIN = 0.0
OPT_DISPLAY_MAX = 0.5
FONT_SIZE = 7.6

GROUP_ORDER = ("Base", "Medium", "Large")
COLORS = {
    "Base": "#4A9C62",
    "Medium": "#355C9A",
    "Large": "#D07A1F",
}
DARK = "#272727"
WHITE = "#FFFFFF"


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_data() -> dict[str, np.ndarray]:
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
            mode = row["mode"].strip()
            if mode not in groups:
                continue
            try:
                feasibility = float(row["error_feas"])
                optimality = float(row["error_opt"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(feasibility) and np.isfinite(optimality)):
                continue
            if feasibility < 0 or optimality < 0:
                continue
            groups[mode].append((feasibility, optimality))

    arrays: dict[str, np.ndarray] = {}
    for mode in GROUP_ORDER:
        values = np.asarray(groups[mode], dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
            raise ValueError(f"Insufficient valid data for {mode}")
        # Apply the floor before both boxplot statistics and point rendering.
        values[:, 0] = np.maximum(values[:, 0], FEAS_DISPLAY_FLOOR)
        arrays[mode] = values
    return arrays


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(direction="out", colors=DARK, which="both")
    ax.xaxis.set_minor_locator(NullLocator())
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK)
        spine.set_linewidth(0.65)


def draw_box_and_points(
    ax: plt.Axes,
    values: np.ndarray,
    position: int,
    color: str,
    rng: np.random.Generator,
) -> None:
    ax.boxplot(
        values,
        positions=[position],
        widths=0.54,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"color": DARK, "linewidth": 1.05},
        boxprops={
            "facecolor": color,
            "alpha": 0.25,
            "edgecolor": DARK,
            "linewidth": 0.65,
        },
        whiskerprops={"color": DARK, "linewidth": 0.65},
        capprops={"color": DARK, "linewidth": 0.65},
    )
    jitter = rng.normal(position, 0.052, size=len(values))
    ax.scatter(
        jitter,
        values,
        s=7.5,
        facecolor=color,
        edgecolor=WHITE,
        linewidth=0.18,
        alpha=0.55,
        zorder=4,
    )


def configure_group_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.5, 3.5)
    ax.set_xticks([1, 2, 3], labels=list(GROUP_ORDER))
    style_axis(ax)


def make_feasibility_figure(groups: dict[str, np.ndarray]) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(89 / 25.4, 65 / 25.4))
    rng = np.random.default_rng(20260903)
    for position, mode in enumerate(GROUP_ORDER, start=1):
        draw_box_and_points(
            ax,
            groups[mode][:, 0],
            position,
            COLORS[mode],
            rng,
        )

    configure_group_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(FEAS_DISPLAY_FLOOR, FEAS_DISPLAY_MAX)
    ax.set_yticks(
        [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        labels=[
            "0",
            r"$10^{-7}$",
            r"$10^{-6}$",
            r"$10^{-5}$",
            r"$10^{-4}$",
            r"$10^{-3}$",
            r"$10^{-2}$",
        ],
    )
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Feasibility error")
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.18, top=0.97)
    return fig


def make_optimality_figure(groups: dict[str, np.ndarray]) -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(89 / 25.4, 65 / 25.4))
    rng = np.random.default_rng(20260904)
    for position, mode in enumerate(GROUP_ORDER, start=1):
        draw_box_and_points(
            ax,
            groups[mode][:, 1],
            position,
            COLORS[mode],
            rng,
        )

    configure_group_axis(ax)
    ax.set_ylim(OPT_DISPLAY_MIN, OPT_DISPLAY_MAX)
    ticks = np.arange(0.0, OPT_DISPLAY_MAX + 0.001, 0.1)
    ax.set_yticks(ticks, labels=[f"{value:.1f}" for value in ticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Optimality error")
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.18, top=0.97)
    return fig


def main() -> None:
    groups = load_data()
    feasibility_figure = make_feasibility_figure(groups)
    optimality_figure = make_optimality_figure(groups)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feasibility_figure.savefig(FEAS_OUTPUT_PATH, format="pdf")
    optimality_figure.savefig(OPT_OUTPUT_PATH, format="pdf")
    plt.close(feasibility_figure)
    plt.close(optimality_figure)
    if LEGACY_OUTPUT_PATH.exists():
        LEGACY_OUTPUT_PATH.unlink()
    print(f"Created {FEAS_OUTPUT_PATH.name}")
    print(f"Created {OPT_OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
