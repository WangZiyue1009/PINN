"""Generate the original low-, typical-, and high-price violin figures together.

This script consolidates the three original plotting programs into one data and
rendering pipeline. It preserves the separate figures, source tables, absolute
cost scales, scenario colors, and distribution summaries used by the originals.
Only three PDF figures are written to the final output subdirectory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures_converged" / "parameter_trend_violins_by_price"


@dataclass(frozen=True)
class ScenarioConfig:
    key: str
    xlabel: str
    colors: tuple[str, str, str, str]


@dataclass(frozen=True)
class PriceConfig:
    key: str
    data_path: Path
    output_name: str
    font_size: float
    figure_size: tuple[float, float]
    kde_pad_fraction: float
    kde_points: int
    fixed_y_bounds: tuple[float, float] | None = None
    fixed_y_ticks: tuple[float, ...] | None = None


SCENARIOS = (
    ScenarioConfig(
        key="ev_count",
        xlabel="Number of EVs",
        colors=("#C7DCEF", "#8FB9D9", "#4F8FC4", "#1764AB"),
    ),
    ScenarioConfig(
        key="temp_mean",
        xlabel="Mean temperature (deg C)",
        colors=("#F7D7B5", "#EEB77A", "#DD8A3A", "#B95F0B"),
    ),
    ScenarioConfig(
        key="pv_energy",
        xlabel="PV-availability hours",
        colors=("#D2E8D2", "#9CCB9C", "#58A65C", "#287A3A"),
    ),
)

PRICE_CONFIGS = (
    PriceConfig(
        key="low",
        data_path=DATA_DIR / "parameter_trend_cost_details_low.csv",
        output_name="parameter_trend_cost_violins_low.pdf",
        font_size=11.0,
        figure_size=(7.5, 2.8),
        kde_pad_fraction=0.15,
        kde_points=400,
    ),
    PriceConfig(
        key="typical",
        data_path=DATA_DIR / "parameter_trend_cost_details_typical.csv",
        output_name="parameter_trend_cost_violins_typical.pdf",
        font_size=12.0,
        figure_size=(7.2, 4.1 * 2.0 / 3.0),
        kde_pad_fraction=0.08,
        kde_points=280,
        fixed_y_bounds=(200.0, 500.0),
        fixed_y_ticks=tuple(np.arange(200.0, 501.0, 50.0)),
    ),
    PriceConfig(
        key="high",
        data_path=DATA_DIR / "parameter_trend_cost_details_high.csv",
        output_name="parameter_trend_cost_violins_high.pdf",
        font_size=11.0,
        figure_size=(7.5, 2.8),
        kde_pad_fraction=0.10,
        kde_points=400,
    ),
)


def apply_mpl_style(font_size: float) -> None:
    """Apply the typography and line styling used in the original figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": 12.0 if font_size >= 12 else 10.5,
            "ytick.labelsize": 12.0 if font_size >= 12 else 10.5,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0 if font_size >= 12 else 3.5,
            "ytick.major.size": 3.0 if font_size >= 12 else 3.5,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_data(path: Path) -> dict[str, dict[float, np.ndarray]]:
    """Load and validate one price-regime source table."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path.name}")
        required = {"parameter_type", "x_value", "cost"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
        rows = list(reader)

    result: dict[str, dict[float, np.ndarray]] = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["parameter_type"] == scenario.key]
        levels = sorted({float(row["x_value"]) for row in scenario_rows})
        if len(levels) != 4:
            raise ValueError(f"{path.name}/{scenario.key} must contain four levels")

        level_data: dict[float, np.ndarray] = {}
        for level in levels:
            values = np.asarray(
                [
                    float(row["cost"])
                    for row in scenario_rows
                    if float(row["x_value"]) == level
                ],
                dtype=float,
            )
            if len(values) < 2 or not np.isfinite(values).all():
                raise ValueError(f"Invalid costs in {path.name}/{scenario.key}/{level:g}")
            level_data[level] = values
        result[scenario.key] = level_data
    return result


def format_level(scenario_key: str, level: float) -> str:
    return f"{int(level)}" if scenario_key == "ev_count" else f"{level:.1f}"


def dynamic_y_axis(
    data: dict[str, dict[float, np.ndarray]],
) -> tuple[tuple[float, float], np.ndarray]:
    """Reproduce the automatic shared y-axis used for low and high prices."""
    values = np.concatenate(
        [array for scenario_data in data.values() for array in scenario_data.values()]
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite cost data")

    data_min = float(values.min())
    data_max = float(values.max())
    data_range = data_max - data_min
    margin = max(abs(data_max) * 0.10, 1.0) if data_range < 1e-12 else 0.08 * data_range
    y_min = max(0.0, data_min - margin)
    y_max = data_max + margin

    raw_step = (y_max - y_min) / 6.0
    if raw_step <= 0:
        tick_step = 1.0
    else:
        magnitude = 10.0 ** np.floor(np.log10(raw_step))
        normalized = raw_step / magnitude
        if normalized <= 1:
            factor = 1.0
        elif normalized <= 2:
            factor = 2.0
        elif normalized <= 5:
            factor = 5.0
        else:
            factor = 10.0
        tick_step = factor * magnitude

    tick_start = np.floor(y_min / tick_step) * tick_step
    tick_end = np.ceil(y_max / tick_step) * tick_step
    ticks = np.arange(tick_start, tick_end + 0.5 * tick_step, tick_step)
    return (float(tick_start), float(tick_end)), ticks


def kde_grid(
    values: np.ndarray,
    config: PriceConfig,
    y_bounds: tuple[float, float],
) -> np.ndarray:
    data_min = float(values.min())
    data_max = float(values.max())
    data_range = data_max - data_min
    if data_range < 1e-12:
        pad = max(abs(data_min) * config.kde_pad_fraction, 1.0)
    else:
        pad = config.kde_pad_fraction * data_range

    lower = data_min - pad
    upper = data_max + pad
    if config.fixed_y_bounds is not None:
        lower = max(y_bounds[0], lower)
        upper = min(y_bounds[1], upper)
    if upper <= lower:
        upper = lower + max(abs(lower) * 1e-6, 1e-6)
    return np.linspace(lower, upper, config.kde_points)


def draw_violin(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    config: PriceConfig,
    y_bounds: tuple[float, float],
    width: float = 0.38,
) -> int:
    """Draw one original-style violin and return its 1.5-IQR outlier count."""
    values = np.asarray(values, dtype=float)
    grid = kde_grid(values, config, y_bounds)
    try:
        density = gaussian_kde(values, bw_method="scott")(grid)
    except np.linalg.LinAlgError:
        # Deterministic microscopic jitter only for a singular KDE input.
        scale = max(abs(float(values[0])) * 1e-6, 1e-8)
        jitter = np.linspace(-scale, scale, len(values))
        density = gaussian_kde(values + jitter, bw_method="scott")(grid)

    density_max = float(density.max())
    half_width = width * density / density_max if density_max > 0 else np.zeros_like(grid)
    ax.fill_betweenx(
        grid,
        position - half_width,
        position + half_width,
        facecolor=color,
        edgecolor="#222222",
        linewidth=0.7,
        alpha=0.82 if config.key == "typical" else 0.85,
        zorder=2,
    )

    q05, q25, q50, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    ax.vlines(position, q05, q95, color="#111111", lw=0.8, zorder=4)
    ax.vlines(position, q25, q75, color="#111111", lw=2.8, zorder=4)
    ax.scatter(
        position,
        q50,
        s=24 if config.key == "typical" else 22,
        facecolor="white",
        edgecolor="#000000",
        linewidth=0.75,
        zorder=5,
    )

    iqr = q75 - q25
    outliers = values[
        (values < q25 - 1.5 * iqr) | (values > q75 + 1.5 * iqr)
    ]
    return int(len(outliers))


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(colors="#000000")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.7)


def plot_price_regime(
    config: PriceConfig,
    data: dict[str, dict[float, np.ndarray]],
    qa_dir: Path | None,
) -> int:
    apply_mpl_style(config.font_size)
    if config.fixed_y_bounds is None:
        y_bounds, y_ticks = dynamic_y_axis(data)
    else:
        y_bounds = config.fixed_y_bounds
        if config.fixed_y_ticks is None:
            raise ValueError(f"Fixed y ticks missing for {config.key}")
        y_ticks = np.asarray(config.fixed_y_ticks)

    positions = np.arange(1, 5)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=config.figure_size,
        sharey=True,
        constrained_layout=True,
    )

    total_outliers = 0
    for ax, scenario in zip(axes, SCENARIOS):
        levels = sorted(data[scenario.key])
        for position, level, color in zip(positions, levels, scenario.colors):
            total_outliers += draw_violin(
                ax,
                data[scenario.key][level],
                float(position),
                color,
                config,
                y_bounds,
            )
        ax.set_xlim(0.48 if config.key == "typical" else 0.45, 4.52 if config.key == "typical" else 4.55)
        ax.set_ylim(y_bounds)
        ax.set_xticks(
            positions,
            labels=[format_level(scenario.key, level) for level in levels],
        )
        ax.set_xlabel(scenario.xlabel)
        style_axis(ax)

    axes[0].set_ylabel("Cost ($)")
    axes[0].set_yticks(y_ticks)
    if config.fixed_y_bounds is None:
        axes[0].set_yticklabels([f"{tick:,.0f}" for tick in y_ticks])
        for ax in axes[1:]:
            ax.tick_params(left=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / config.output_name
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    if qa_dir is not None:
        qa_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            qa_dir / output_path.with_suffix(".png").name,
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    return total_outliers


def clean_non_pdf_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for config in PRICE_CONFIGS:
        stem = Path(config.output_name).stem
        for suffix in (".png", ".svg", ".tiff", ".csv", ".md"):
            stale = OUT_DIR / f"{stem}{suffix}"
            if stale.exists():
                stale.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=None,
        help="Optional temporary directory for Python-rendered PNG previews.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_non_pdf_outputs()
    for config in PRICE_CONFIGS:
        data = load_data(config.data_path)
        outliers = plot_price_regime(config, data, args.qa_dir)
        sample_sizes = {
            len(values)
            for scenario_data in data.values()
            for values in scenario_data.values()
        }
        print(
            f"{config.key}: {config.output_name}; "
            f"n per level={sorted(sample_sizes)}; outliers={outliers}"
        )


if __name__ == "__main__":
    main()
