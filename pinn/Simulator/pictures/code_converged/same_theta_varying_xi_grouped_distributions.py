"""Grouped distributions for fixed aggregate descriptors and varying devices.

The experiment is a repeated 40-by-10 design.  The boundary-error figures group
the same 40 device-level configurations (xi) by the ten common directions; the
disaggregation figure groups those configurations by the ten common price curves.  Each
group is drawn as a compact raincloud: a half violin, every observation, and
the median/IQR/5--95% interval.

Running this module exports the two boundary-error plots as separate PDFs in one
subdirectory and the selected two-row price-curve figure as another PDF.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures_converged"

ERROR_DATA = DATA_DIR / "same_theta_errors_detail.csv"
DISPATCH_DATA = DATA_DIR / "economic_dispatch_details.csv"
BOUNDARY_OUT_DIR = OUT_DIR / "same_theta_varying_xi_boundary_error_panels"
OUTPUT_FEASIBILITY_PATH = BOUNDARY_OUT_DIR / "Feasibility error.pdf"
OUTPUT_OPTIMALITY_PATH = BOUNDARY_OUT_DIR / "Optimality error.pdf"
OUTPUT_C_PATH = OUT_DIR / "same_theta_varying_xi_disaggregation_by_price.pdf"
LEGACY_BOUNDARY_PATH = OUT_DIR / "same_theta_varying_xi_boundary_errors.pdf"
LEGACY_OUTPUT_PATH = OUT_DIR / "same_theta_varying_xi_grouped_distributions.pdf"

METHOD_ID = "same θ varying ξ"
FEAS_DISPLAY_FLOOR = 1e-8
FONT_SIZE = 7.6

# Same restrained palette used by the converged figures.  Blue denotes the
# direction-based region test; orange denotes the price-based dispatch test.
BLUE = "#355C9A"
BLUE_DARK = "#244775"
ORANGE = "#D07A1F"
ORANGE_DARK = "#985412"
DARK = "#272727"
WHITE = "#FFFFFF"


def apply_publication_style(font_size: float = FONT_SIZE) -> None:
    """Use the established Times New Roman, boxed-axis visual system."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
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


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Source data not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Source data are empty: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    return rows


def validate_rectangular_design(
    first_id: np.ndarray,
    second_id: np.ndarray,
    first_name: str,
    second_name: str,
) -> None:
    """Require exactly one observation for every pair in the repeated design."""
    pairs = list(zip(first_id.tolist(), second_id.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"Duplicate {first_name}--{second_name} pairs were found")
    expected = int(np.unique(first_id).size * np.unique(second_id).size)
    if len(pairs) != expected:
        raise ValueError(
            f"Incomplete repeated design: found {len(pairs)} pairs, expected {expected}"
        )


def load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    error_rows = read_rows(
        ERROR_DATA,
        {"xi_id", "cal_index", "error_feas", "error_opt"},
    )
    xi_id = np.asarray([int(row["xi_id"]) for row in error_rows], dtype=int)
    direction_id = np.asarray(
        [int(row["cal_index"]) for row in error_rows], dtype=int
    )
    feasibility_raw = np.asarray(
        [float(row["error_feas"]) for row in error_rows], dtype=float
    )
    optimality = np.asarray(
        [float(row["error_opt"]) for row in error_rows], dtype=float
    )
    validate_rectangular_design(xi_id, direction_id, "xi_id", "cal_index")
    if np.unique(xi_id).size != 40 or np.unique(direction_id).size != 10:
        raise ValueError("Expected 40 xi scenarios and 10 boundary directions")
    if not (np.isfinite(feasibility_raw).all() and np.isfinite(optimality).all()):
        raise ValueError("Estimation errors must be finite")
    if np.any(feasibility_raw < 0) or np.any(optimality < 0):
        raise ValueError("Estimation errors cannot be negative")

    dispatch_rows = read_rows(
        DISPATCH_DATA,
        {"sample_id", "price_id", "method_id", "rel_decomp_error"},
    )
    dispatch_rows = [
        row for row in dispatch_rows if row["method_id"].strip() == METHOD_ID
    ]
    if not dispatch_rows:
        raise ValueError(f"No dispatch rows found for method_id={METHOD_ID!r}")
    sample_id = np.asarray(
        [int(row["sample_id"]) for row in dispatch_rows], dtype=int
    )
    price_id = np.asarray(
        [int(row["price_id"]) for row in dispatch_rows], dtype=int
    )
    disaggregation = np.asarray(
        [float(row["rel_decomp_error"]) for row in dispatch_rows], dtype=float
    )
    validate_rectangular_design(sample_id, price_id, "sample_id", "price_id")
    if np.unique(sample_id).size != 40 or np.unique(price_id).size != 10:
        raise ValueError("Expected 40 xi scenarios and 10 price curves")
    if not np.isfinite(disaggregation).all() or np.any(disaggregation < 0):
        raise ValueError("Relative disaggregation errors must be finite and nonnegative")

    errors = {
        "xi_id": xi_id,
        "direction_id": direction_id,
        "feasibility_raw": feasibility_raw,
        "feasibility": np.maximum(feasibility_raw, FEAS_DISPLAY_FLOOR),
        "optimality": optimality,
    }
    dispatch = {
        "xi_id": sample_id,
        "price_id": price_id,
        "disaggregation": disaggregation,
    }
    return errors, dispatch


def robust_bandwidth(values: np.ndarray, minimum: float) -> float:
    """Scott-type bandwidth using the smaller valid standard/robust scale."""
    values = np.asarray(values, dtype=float)
    if np.allclose(values, values[0], rtol=0.0, atol=1e-14):
        return minimum
    standard_scale = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    robust_scale = float((q75 - q25) / 1.349)
    valid = [scale for scale in (standard_scale, robust_scale) if scale > 1e-14]
    scale = min(valid) if valid else standard_scale
    return max(minimum, 1.06 * scale * len(values) ** (-1.0 / 5.0))


def kde_1d(
    values: np.ndarray,
    grid: np.ndarray,
    minimum_bandwidth: float,
    reflect_at_zero: bool = False,
) -> np.ndarray:
    """Evaluate a Gaussian KDE, optionally reflecting a nonnegative boundary."""
    bandwidth = robust_bandwidth(values, minimum_bandwidth)
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    if reflect_at_zero:
        reflected = (grid[:, None] + values[None, :]) / bandwidth
        density += np.exp(-0.5 * reflected**2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2.0 * np.pi)
    return density


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(direction="out", colors=DARK, which="both")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK)
        spine.set_linewidth(0.65)


def draw_raincloud_group(
    ax: plt.Axes,
    position: float,
    values: np.ndarray,
    color: str,
    dark_color: str,
    grid: np.ndarray,
    *,
    transformed: bool = False,
    reflect_at_zero: bool = False,
    seed: int,
) -> None:
    """Draw a left half violin, right-side points, and central quantiles."""
    if transformed:
        distribution_values = np.log10(values)
        minimum_bandwidth = 0.10
        density = kde_1d(distribution_values, grid, minimum_bandwidth)
        display_grid = 10.0**grid
    else:
        distribution_values = values
        grid_step = float(np.diff(grid).mean())
        minimum_bandwidth = max(grid_step * 2.5, 1e-5)
        density = kde_1d(
            distribution_values,
            grid,
            minimum_bandwidth,
            reflect_at_zero=reflect_at_zero,
        )
        display_grid = grid

    if float(density.max()) > 0:
        width = 0.34 * density / density.max()
        ax.fill_betweenx(
            display_grid,
            position - width,
            position,
            facecolor=color,
            edgecolor=dark_color,
            linewidth=0.65,
            alpha=0.18,
            zorder=1,
        )

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.055, 0.285, size=len(values))
    ax.scatter(
        position + jitter,
        values,
        s=9.5,
        facecolor=color,
        edgecolor=WHITE,
        linewidth=0.22,
        alpha=0.68,
        clip_on=True,
        zorder=3,
    )

    q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    ax.plot(
        [position, position],
        [q05, q95],
        color=dark_color,
        linewidth=0.7,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        [position, position],
        [q25, q75],
        color=dark_color,
        linewidth=2.4,
        solid_capstyle="round",
        zorder=5,
    )
    ax.scatter(
        [position],
        [median],
        s=13.0,
        facecolor=WHITE,
        edgecolor=dark_color,
        linewidth=0.7,
        zorder=6,
    )


def draw_grouped_panel(
    ax: plt.Axes,
    group_ids: np.ndarray,
    observation_ids: np.ndarray,
    values: np.ndarray,
    color: str,
    dark_color: str,
    grid: np.ndarray,
    *,
    transformed: bool = False,
    reflect_at_zero: bool = False,
    seed_offset: int,
) -> list[int]:
    unique_ids = sorted(int(value) for value in np.unique(group_ids))
    for position, group_id in enumerate(unique_ids, start=1):
        group_mask = group_ids == group_id
        order = np.argsort(observation_ids[group_mask])
        group_values = values[group_mask][order]
        if len(group_values) != 40:
            raise ValueError(f"Group {group_id} contains {len(group_values)} rather than 40 rows")
        draw_raincloud_group(
            ax,
            float(position),
            group_values,
            color,
            dark_color,
            grid,
            transformed=transformed,
            reflect_at_zero=reflect_at_zero,
            seed=seed_offset + position,
        )
    ax.set_xlim(0.48, len(unique_ids) + 0.52)
    ax.set_xticks(np.arange(1, len(unique_ids) + 1), labels=[str(i) for i in unique_ids])
    style_axis(ax)
    return unique_ids


def plot_feasibility(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    grid = np.linspace(np.log10(FEAS_DISPLAY_FLOOR), -3.0, 360)
    draw_grouped_panel(
        ax,
        data["direction_id"],
        data["xi_id"],
        data["feasibility"],
        BLUE,
        BLUE_DARK,
        grid,
        transformed=True,
        seed_offset=100,
    )
    ax.set_yscale("log")
    ax.set_ylim(FEAS_DISPLAY_FLOOR, 1e-3)
    ax.set_yticks(
        [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        labels=[
            "0",
            r"$10^{-7}$",
            r"$10^{-6}$",
            r"$10^{-5}$",
            r"$10^{-4}$",
            r"$10^{-3}$",
        ],
    )
    ax.set_xlabel("Direction ID")
    ax.set_ylabel("Feasibility error")


def plot_optimality(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    grid = np.linspace(0.0, 0.5, 360)
    draw_grouped_panel(
        ax,
        data["direction_id"],
        data["xi_id"],
        data["optimality"],
        BLUE,
        BLUE_DARK,
        grid,
        # The same jitter seed as panel a keeps each xi scenario horizontally
        # aligned between the feasibility and optimality panels.
        seed_offset=100,
    )
    ax.set_ylim(0.0, 0.5)
    ax.set_yticks(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        labels=["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
    )
    ax.set_xlabel("Direction ID")
    ax.set_ylabel("Optimality error")


def draw_price_subset(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    price_ids: tuple[int, ...],
) -> None:
    """Draw five selected price curves using one-based display labels."""
    mask = np.isin(data["price_id"], np.asarray(price_ids))
    subset_price = data["price_id"][mask]
    subset_xi = data["xi_id"][mask]
    subset_error = data["disaggregation"][mask]

    grid = np.linspace(0.0, 0.06, 360)
    plotted_ids = draw_grouped_panel(
        ax,
        subset_price,
        subset_xi,
        subset_error,
        ORANGE,
        ORANGE_DARK,
        grid,
        reflect_at_zero=True,
        seed_offset=300,
    )
    ax.set_ylim(-0.0018, 0.06)
    ax.set_yticks(
        [0.0, 0.02, 0.04, 0.06],
        labels=["0", "2%", "4%", "6%"],
    )
    ax.set_xticks(
        np.arange(1, len(plotted_ids) + 1),
        labels=[str(price_id + 1) for price_id in plotted_ids],
    )


def make_boundary_figures(
    errors: dict[str, np.ndarray],
) -> tuple[plt.Figure, plt.Figure]:
    apply_publication_style(FONT_SIZE * 1.5)
    feasibility_figure, feasibility_axis = plt.subplots(figsize=(120 / 25.4, 60 / 25.4))
    optimality_figure, optimality_axis = plt.subplots(figsize=(120 / 25.4, 60 / 25.4))
    plot_feasibility(feasibility_axis, errors)
    plot_optimality(optimality_axis, errors)
    for figure in (feasibility_figure, optimality_figure):
        figure.subplots_adjust(left=0.20, right=0.985, bottom=0.27, top=0.97)
    return feasibility_figure, optimality_figure


def make_disaggregation_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    """Wrap the ten price curves into two compact rows of five groups."""
    apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(105 / 25.4, 100 / 25.4))
    draw_price_subset(axes[0], data, (0, 1, 2, 3, 4))
    draw_price_subset(axes[1], data, (5, 6, 7, 8, 9))
    fig.supylabel("Relative disaggregation error", x=0.11, fontsize=FONT_SIZE)
    fig.supxlabel("Price-curve ID", x=0.59, y=0.07, fontsize=FONT_SIZE)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.14, top=0.965, hspace=0.20)
    return fig


def main() -> None:
    errors, dispatch = load_data()
    feasibility_figure, optimality_figure = make_boundary_figures(errors)
    disaggregation_figure = make_disaggregation_figure(dispatch)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BOUNDARY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    feasibility_figure.savefig(OUTPUT_FEASIBILITY_PATH, format="pdf")
    optimality_figure.savefig(OUTPUT_OPTIMALITY_PATH, format="pdf")
    disaggregation_figure.savefig(OUTPUT_C_PATH, format="pdf")
    plt.close(feasibility_figure)
    plt.close(optimality_figure)
    plt.close(disaggregation_figure)
    if LEGACY_BOUNDARY_PATH.exists():
        LEGACY_BOUNDARY_PATH.unlink()
    if LEGACY_OUTPUT_PATH.exists():
        LEGACY_OUTPUT_PATH.unlink()
    print(f"Created {OUTPUT_FEASIBILITY_PATH.relative_to(OUT_DIR)}")
    print(f"Created {OUTPUT_OPTIMALITY_PATH.relative_to(OUT_DIR)}")
    print(f"Created {OUTPUT_C_PATH.name}")


if __name__ == "__main__":
    main()
