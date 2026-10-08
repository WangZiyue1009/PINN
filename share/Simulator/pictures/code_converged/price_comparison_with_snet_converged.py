"""Compare operational performance across aggregation methods.

The source table contains 400 aligned random-price dispatch cases for PINN,
SNET, OPEB, and AAIB.  The true-region cost is validated across method rows
before a unique physical-reference distribution is constructed.  Dispatch
cost and relative disaggregation error are exported as two standalone PDFs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "price_guided_cost_details.csv"
OUT_DIR = PROJECT_DIR / "figures_converged"
OUTPUT_DIR = OUT_DIR / "price_comparison"
COST_OUTPUT_PATH = OUTPUT_DIR / "Dispatch cost.pdf"
ERROR_OUTPUT_PATH = OUTPUT_DIR / "Relative disaggregation error.pdf"
LEGACY_OUTPUT_PATH = OUT_DIR / "price_comparison_with_snet.pdf"

METHOD_ORDER = ("fullnet", "supervised", "power_energy", "cube", "true_model")
SOURCE_METHODS = ("fullnet", "supervised", "power_energy", "cube")
METHOD_LABELS = {
    "fullnet": "PINN",
    "supervised": "DDNN",
    "power_energy": "OPEB",
    "cube": "AAIB",
    "true_model": "True model",
}
COLORS = {
    "fullnet": "#355C9A",
    "supervised": "#D07A1F",
    "power_energy": "#7A6FA8",
    "cube": "#4A9C84",
    "true_model": "#858585",
}

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


def read_source_rows() -> list[dict[str, str]]:
    required = {
        "sample_id",
        "price_id",
        "method_id",
        "true_cost",
        "method_cost",
        "rel_decomp_error",
    }
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Source data not found: {DATA_PATH}")
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"Missing source-data columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("The source-data table is empty")
    return rows


def load_data() -> dict[str, dict[str, np.ndarray]]:
    """Validate 400 aligned cases and build the five plotted distributions."""
    rows = read_source_rows()
    by_case: dict[tuple[int, int], dict[str, dict[str, str]]] = {}
    for row_number, row in enumerate(rows, start=2):
        method = row["method_id"].strip()
        if method not in SOURCE_METHODS:
            continue
        try:
            case_key = (int(row["sample_id"]), int(row["price_id"]))
            values = np.asarray(
                [
                    float(row["true_cost"]),
                    float(row["method_cost"]),
                    float(row["rel_decomp_error"]),
                ],
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value in row {row_number}") from exc
        if not np.isfinite(values).all() or values[2] < 0:
            raise ValueError(f"Invalid metric in row {row_number}")
        case_rows = by_case.setdefault(case_key, {})
        if method in case_rows:
            raise ValueError(f"Duplicate method {method!r} for case {case_key}")
        case_rows[method] = row

    expected = set(SOURCE_METHODS)
    for case_key, case_rows in by_case.items():
        if set(case_rows) != expected:
            raise ValueError(
                f"Case {case_key} contains {sorted(case_rows)}, "
                f"expected {sorted(expected)}"
            )
        true_costs = np.asarray(
            [float(case_rows[method]["true_cost"]) for method in SOURCE_METHODS]
        )
        if not np.allclose(true_costs, true_costs[0], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Inconsistent true_cost values for case {case_key}")

    if len(by_case) != 400:
        raise ValueError(f"Expected 400 dispatch cases, found {len(by_case)}")

    ordered_cases = sorted(by_case)
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for method in SOURCE_METHODS:
        distributions[method] = {
            "cost": np.asarray(
                [float(by_case[key][method]["method_cost"]) for key in ordered_cases]
            ),
            "error": np.asarray(
                [
                    float(by_case[key][method]["rel_decomp_error"])
                    for key in ordered_cases
                ]
            ),
        }
    distributions["true_model"] = {
        "cost": np.asarray(
            [float(by_case[key]["fullnet"]["true_cost"]) for key in ordered_cases]
        ),
        "error": np.zeros(len(ordered_cases), dtype=float),
    }
    return distributions


def draw_violin(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    bounds: tuple[float, float],
) -> None:
    """Draw a violin plus median, IQR, and 5th--95th percentile interval."""
    values = np.asarray(values, dtype=float)
    q05, q25, median, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    spread = float(np.ptp(values))

    if spread <= np.finfo(float).eps * max(1.0, abs(float(values[0]))):
        ax.hlines(
            median,
            position - 0.30,
            position + 0.30,
            color=color,
            linewidth=2.4,
            zorder=4,
            clip_on=False,
        )
    else:
        pad = 0.08 * spread
        grid_low = max(bounds[0], float(values.min()) - pad)
        grid_high = min(bounds[1], float(values.max()) + pad)
        grid = np.linspace(grid_low, grid_high, 360)
        density = gaussian_kde(values, bw_method="scott")(grid)
        half_width = 0.34 * density / density.max()
        ax.fill_betweenx(
            grid,
            position - half_width,
            position + half_width,
            facecolor=color,
            edgecolor=color,
            linewidth=0.75,
            alpha=0.68,
            zorder=2,
        )

    ax.vlines(position, q05, q95, color=DARK, linewidth=0.65, zorder=4)
    ax.vlines(position, q25, q75, color=DARK, linewidth=2.2, zorder=5)
    ax.scatter(
        position,
        median,
        s=12,
        facecolor="white",
        edgecolor=DARK,
        linewidth=0.5,
        zorder=6,
        clip_on=False,
    )

    # Make a dominant exact-zero point mass visible even when the continuous
    # violin is compressed against the lower axis boundary. This applies to
    # PINN (96.25% zeros), AAIB, and the true model, but not DDNN (1% zeros).
    zero_fraction = float(np.mean(values == 0.0))
    if zero_fraction >= 0.5:
        half_width = 0.30 * zero_fraction
        ax.hlines(
            0.0,
            position - half_width,
            position + half_width,
            color=color,
            linewidth=3.6,
            zorder=7,
            clip_on=False,
        )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(direction="out", colors=DARK)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK)
        spine.set_linewidth(0.65)


def configure_method_axis(ax: plt.Axes) -> None:
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    ax.set_xlim(0.45, len(METHOD_ORDER) + 0.55)
    ax.set_xticks(positions, labels=labels, rotation=0, ha="center")
    style_axis(ax)


def annotate_pinn_maximum(ax: plt.Axes, value: float) -> None:
    """Mark only the PINN maximum, vertically above the PINN violin."""
    position = 1.0
    color = COLORS["fullnet"]
    ax.scatter(
        position,
        value,
        s=25,
        facecolor=color,
        edgecolor=DARK,
        linewidth=0.55,
        zorder=7,
        clip_on=False,
    )
    line_top = value + 0.025
    ax.vlines(
        position,
        value + 0.006,
        line_top,
        color=color,
        linewidth=0.55,
        zorder=7,
    )
    ax.text(
        position,
        line_top + 0.008,
        f"max = {value:.2%}",
        rotation=90,
        rotation_mode="anchor",
        ha="left",
        va="center",
        color=color,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.3},
        zorder=8,
    )


def make_cost_figure(data: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    apply_publication_style()
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    fig, ax = plt.subplots(figsize=(89 / 25.4, 65 / 25.4))
    bounds = (0.0, 600.0)
    for position, method in zip(positions, METHOD_ORDER):
        draw_violin(ax, data[method]["cost"], position, COLORS[method], bounds)
    ax.set_ylim(bounds)
    ax.set_yticks(np.arange(0.0, 601.0, 100.0))
    ax.set_ylabel("Dispatch cost (USD)")
    configure_method_axis(ax)
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.18, top=0.955)
    return fig


def make_error_figure(data: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    apply_publication_style()
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    fig, ax = plt.subplots(figsize=(89 / 25.4, 65 / 25.4))
    bounds = (0.0, 1.0)
    for position, method in zip(positions, METHOD_ORDER):
        draw_violin(ax, data[method]["error"], position, COLORS[method], bounds)
    ax.set_ylim(bounds)
    ax.set_yticks(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0", "20%", "40%", "60%", "80%", "100%"],
    )
    ax.set_ylabel("Relative disaggregation error")
    configure_method_axis(ax)
    annotate_pinn_maximum(ax, float(data["fullnet"]["error"].max()))
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.18, top=0.955)
    return fig


def main() -> None:
    data = load_data()
    cost_figure = make_cost_figure(data)
    error_figure = make_error_figure(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cost_figure.savefig(COST_OUTPUT_PATH, format="pdf")
    error_figure.savefig(ERROR_OUTPUT_PATH, format="pdf")
    plt.close(cost_figure)
    plt.close(error_figure)
    if LEGACY_OUTPUT_PATH.exists():
        LEGACY_OUTPUT_PATH.unlink()
    print(f"Validated 400 random-price cases across {len(METHOD_ORDER)} methods")
    print(f"Created {COST_OUTPUT_PATH.name}")
    print(f"Created {ERROR_OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
