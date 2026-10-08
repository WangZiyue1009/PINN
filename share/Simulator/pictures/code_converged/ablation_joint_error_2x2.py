"""Separate feasibility--optimality distributions for four ablation experiments.

Each figure uses solid observations and the filled, outlined main component of a
90% two-dimensional high-density region. Source CSV schemas are preserved and
read independently.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, NullLocator


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "figures_converged"
OUTPUT_DIR = OUT_DIR / "ablation_joint_error_panels"
FEAS_DISPLAY_FLOOR = 1e-8
OPT_DISPLAY_FLOOR = 1e-8
FONT_SIZE = 7.6

BLUE = "#355C9A"
ORANGE = "#D07A1F"
TEAL = "#4A9C84"
PURPLE = "#8064A2"
DARK = "#272727"
GRID = "#E3E3E3"

MAX_LABEL_OFFSETS = {
    ("a", "full_all"): (12, 16),
    ("a", "no_pretrain"): (8, -12),
    ("b", "full_all"): (0, 16),
    ("b", "no_feas_loss"): (0, 18),
    ("b", "no_opt_loss"): (8, 10),
    ("c", "three stage λ"): (12, 16),
    ("c", "fixed λ"): (8, 10),
    ("c", "adaptive λ"): (8, -12),
    ("d", "one"): (10, -16),
    ("d", "three"): (12, 0),
    ("d", "five"): (-12, 6),
    ("d", "seven"): (0, 18),
}

MAX_LABEL_ALIGNMENTS = {
    ("b", "full_all"): ("center", "bottom"),
    ("b", "no_feas_loss"): ("center", "bottom"),
    ("d", "seven"): ("center", "bottom"),
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    color: str
    marker: str


@dataclass(frozen=True)
class ExperimentSpec:
    panel: str
    title: str
    data_path: Path
    methods: tuple[MethodSpec, ...]
    legend_loc: str
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    y_scale: str


EXPERIMENTS = (
    ExperimentSpec(
        panel="a",
        title="Pretraining",
        data_path=DATA_DIR / "no_pre_full.csv",
        methods=(
            MethodSpec("full_all", "With pretraining", BLUE, "o"),
            MethodSpec("no_pretrain", "Without pretraining", PURPLE, "^"),
        ),
        legend_loc="upper right",
        x_bounds=(FEAS_DISPLAY_FLOOR, 1e2),
        y_bounds=(OPT_DISPLAY_FLOOR, 1.0),
        y_scale="linear",
    ),
    ExperimentSpec(
        panel="b",
        title="Physics-informed losses",
        data_path=DATA_DIR / "no_feas_opt.csv",
        methods=(
            MethodSpec("full_all", "Full model", BLUE, "o"),
            MethodSpec("no_feas_loss", "Only optimality loss", ORANGE, "^"),
            MethodSpec("no_opt_loss", "Only feasibility loss", TEAL, "D"),
        ),
        legend_loc="upper right",
        x_bounds=(FEAS_DISPLAY_FLOOR, 1e2),
        y_bounds=(OPT_DISPLAY_FLOOR, 5.0),
        y_scale="linear",
    ),
    ExperimentSpec(
        panel="c",
        title="Loss weighting lambda",
        data_path=DATA_DIR / "λ.csv",
        methods=(
            MethodSpec("three stage λ", "Three-stage", BLUE, "o"),
            MethodSpec("fixed λ", "Fixed", ORANGE, "^"),
            MethodSpec("adaptive λ", "Adaptive", TEAL, "D"),
        ),
        legend_loc="upper right",
        x_bounds=(FEAS_DISPLAY_FLOOR, 1e2),
        y_bounds=(OPT_DISPLAY_FLOOR, 1.0),
        y_scale="linear",
    ),
    ExperimentSpec(
        panel="d",
        title="Boundary-sampling directions",
        data_path=DATA_DIR / "n_cal.csv",
        methods=(
            MethodSpec("one", r"$n_{\mathrm{cal}}=1$", ORANGE, "o"),
            MethodSpec("three", r"$n_{\mathrm{cal}}=3$", BLUE, "s"),
            MethodSpec("five", r"$n_{\mathrm{cal}}=5$", TEAL, "D"),
            MethodSpec("seven", r"$n_{\mathrm{cal}}=7$", PURPLE, "^"),
        ),
        legend_loc="upper right",
        x_bounds=(FEAS_DISPLAY_FLOOR, 1e2),
        y_bounds=(OPT_DISPLAY_FLOOR, 0.5),
        y_scale="linear",
    ),
)


def apply_publication_style() -> None:
    mpl.rcParams.update(
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


def load_experiment(spec: ExperimentSpec) -> dict[str, np.ndarray]:
    """Read one source table while discarding blank/non-positive placeholder rows."""
    if not spec.data_path.exists():
        raise FileNotFoundError(f"Source data not found: {spec.data_path}")

    with spec.data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {spec.data_path.name}")
        required = {"mode", "error_feas", "error_opt"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {spec.data_path.name}: {sorted(missing)}")
        rows = list(reader)

    expected = {method.key for method in spec.methods}
    groups: dict[str, list[tuple[float, float]]] = {key: [] for key in expected}
    for row in rows:
        key = row["mode"].strip().lower()
        if key not in groups:
            continue
        try:
            feasibility = float(row["error_feas"])
            optimality = float(row["error_opt"])
        except (TypeError, ValueError):
            continue
        if (
            np.isfinite(feasibility)
            and np.isfinite(optimality)
            and feasibility > 0
            and optimality > 0
        ):
            groups[key].append((feasibility, optimality))

    arrays: dict[str, np.ndarray] = {}
    for method in spec.methods:
        values = np.asarray(groups[method.key], dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 10:
            raise ValueError(f"Insufficient valid data for {spec.panel}/{method.key}")
        # Apply the display truncation before scatter and density estimation.
        values[:, 0] = np.maximum(values[:, 0], FEAS_DISPLAY_FLOOR)
        values[:, 1] = np.maximum(values[:, 1], OPT_DISPLAY_FLOOR)
        arrays[method.key] = values
    return arrays


def robust_bandwidth(values: np.ndarray, minimum: float) -> float:
    standard_scale = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25, 75])
    robust_scale = float((q75 - q25) / 1.349)
    valid = [scale for scale in (standard_scale, robust_scale) if scale > 0]
    scale = min(valid) if valid else minimum
    return max(minimum, 1.06 * scale * len(values) ** (-1.0 / 5.0))


def kde_2d_display(
    values: np.ndarray,
    x_log_grid: np.ndarray,
    y_grid: np.ndarray,
    y_scale: str,
) -> np.ndarray:
    """Evaluate a robust KDE in the coordinate system displayed by the panel."""
    x = np.log10(values[:, 0])
    y = np.log10(values[:, 1]) if y_scale == "log" else values[:, 1]
    bandwidth_x = robust_bandwidth(x, minimum=0.16)
    bandwidth_y = robust_bandwidth(y, minimum=0.08 if y_scale == "log" else 0.035)
    grid_x, grid_y = np.meshgrid(x_log_grid, y_grid)
    density = np.zeros_like(grid_x)
    for x_i, y_i in zip(x, y):
        distance = ((grid_x - x_i) / bandwidth_x) ** 2
        distance += ((grid_y - y_i) / bandwidth_y) ** 2
        density += np.exp(-0.5 * distance)
    density /= 2.0 * np.pi * len(values) * bandwidth_x * bandwidth_y
    return density


def density_threshold(density: np.ndarray, enclosed_mass: float = 0.90) -> float:
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered)
    if cumulative[-1] <= 0:
        raise ValueError("KDE has zero total density")
    cumulative /= cumulative[-1]
    index = min(int(np.searchsorted(cumulative, enclosed_mass)), len(ordered) - 1)
    return float(ordered[index])


def main_density_component(density: np.ndarray, threshold: float) -> np.ndarray:
    """Keep the threshold component containing the global density maximum."""
    above = density >= threshold
    peak = tuple(int(index) for index in np.unravel_index(np.argmax(density), density.shape))
    connected = np.zeros_like(above, dtype=bool)
    stack = [peak]
    n_rows, n_cols = density.shape
    while stack:
        row, col = stack.pop()
        if connected[row, col] or not above[row, col]:
            continue
        connected[row, col] = True
        if row > 0:
            stack.append((row - 1, col))
        if row + 1 < n_rows:
            stack.append((row + 1, col))
        if col > 0:
            stack.append((row, col - 1))
        if col + 1 < n_cols:
            stack.append((row, col + 1))
    return np.where(connected, density, 0.0)


def labelled_decades(log_min: float, log_max: float, max_ticks: int = 7) -> list[float]:
    exponent_min = int(np.ceil(log_min))
    exponent_max = int(np.floor(log_max))
    if exponent_max < exponent_min:
        return [10.0 ** ((log_min + log_max) / 2.0)]
    count = exponent_max - exponent_min + 1
    step = max(1, int(np.ceil(count / max_ticks)))
    exponents = list(range(exponent_min, exponent_max + 1, step))
    if exponents[-1] != exponent_max:
        exponents.append(exponent_max)
    return [10.0**exponent for exponent in exponents]


def set_feasibility_ticks(ax: plt.Axes, x_max: float) -> None:
    """Label the clipped left boundary as zero and retain sparse log ticks."""
    exponents = (-8, -6, -4, -2, 0, 2)
    ticks = [10.0**exponent for exponent in exponents if 10.0**exponent <= x_max]
    labels = ["0"] + [rf"$10^{{{exponent}}}$" for exponent in exponents[1:] if 10.0**exponent <= x_max]
    ax.set_xticks(ticks, labels=labels)


def set_y_log_ticks(ax: plt.Axes, log_min: float, log_max: float) -> None:
    """Use readable 1-2-5 ticks when the displayed y span is short."""
    if log_max - log_min <= 2.2:
        candidates: list[float] = []
        for exponent in range(int(np.floor(log_min)) - 1, int(np.ceil(log_max)) + 2):
            for multiplier in (1.0, 2.0, 5.0):
                value = multiplier * 10.0**exponent
                if 10.0**log_min <= value <= 10.0**log_max:
                    candidates.append(value)
        ax.yaxis.set_major_locator(FixedLocator(candidates))

        def compact_number(value: float, _position: float) -> str:
            if value >= 0.01:
                return f"{value:g}"
            return f"{value:.0e}"

        ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
    else:
        ax.yaxis.set_major_locator(FixedLocator(labelled_decades(log_min, log_max, 6)))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))


def format_maximum(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    if exponent == 0:
        return f"{value:.2f}"
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.2f}\!\times\!10^{{{exponent}}}$"


def annotate_maximum(
    ax: plt.Axes,
    spec: ExperimentSpec,
    method: MethodSpec,
    values: np.ndarray,
) -> None:
    max_index = int(np.argmax(values[:, 0]))
    max_x, max_y = values[max_index]
    offset_x, offset_y = MAX_LABEL_OFFSETS[(spec.panel, method.key)]
    horizontal_alignment, vertical_alignment = MAX_LABEL_ALIGNMENTS.get(
        (spec.panel, method.key),
        (
            "left" if offset_x >= 0 else "right",
            "bottom" if offset_y >= 0 else "top",
        ),
    )

    ax.scatter(
        [max_x],
        [max_y],
        s=38,
        marker=method.marker,
        facecolor=method.color,
        edgecolor="white",
        linewidth=1.4,
        clip_on=False,
        zorder=5,
    )
    ax.scatter(
        [max_x],
        [max_y],
        s=25,
        marker=method.marker,
        facecolor=method.color,
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
        color=method.color,
        arrowprops={
            "arrowstyle": "-",
            "color": method.color,
            "linewidth": 0.55,
            "shrinkA": 1.5,
            "shrinkB": 4.0,
        },
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.35},
        annotation_clip=True,
        zorder=7,
    )


def draw_group(
    ax: plt.Axes,
    method: MethodSpec,
    values: np.ndarray,
    x_log_grid: np.ndarray,
    y_grid: np.ndarray,
    y_scale: str,
) -> None:
    density = kde_2d_display(values, x_log_grid, y_grid, y_scale)
    threshold = density_threshold(density, 0.90)
    main_density = main_density_component(density, threshold)
    displayed_y_grid = 10.0**y_grid if y_scale == "log" else y_grid
    grid_x, grid_y = np.meshgrid(10.0**x_log_grid, displayed_y_grid)

    ax.contourf(
        grid_x,
        grid_y,
        main_density,
        levels=[threshold, float(density.max())],
        colors=[method.color],
        alpha=0.11,
        zorder=1,
    )
    ax.contour(
        grid_x,
        grid_y,
        main_density,
        levels=[threshold],
        colors=[method.color],
        linewidths=1.05,
        alpha=0.96,
        zorder=2,
    )
    ax.scatter(
        values[:, 0],
        values[:, 1],
        s=7.0,
        marker=method.marker,
        facecolor=method.color,
        edgecolor="white",
        linewidth=0.18,
        alpha=0.28,
        rasterized=False,
        zorder=3,
    )


def style_panel(
    ax: plt.Axes,
    spec: ExperimentSpec,
    groups: dict[str, np.ndarray],
) -> None:
    x_log_min, x_log_max = np.log10(spec.x_bounds)
    x_log_grid = np.linspace(x_log_min, x_log_max, 230)
    if spec.y_scale == "log":
        y_grid_min, y_grid_max = np.log10(spec.y_bounds)
        y_grid = np.linspace(y_grid_min, y_grid_max, 190)
    elif spec.y_scale == "linear":
        y_grid_min, y_grid_max = spec.y_bounds
        y_grid = np.linspace(y_grid_min, y_grid_max, 420)
    else:
        raise ValueError(f"Unsupported y scale: {spec.y_scale}")

    for method in spec.methods:
        draw_group(ax, method, groups[method.key], x_log_grid, y_grid, spec.y_scale)

    ax.set_xscale("log")
    if spec.y_scale == "log":
        ax.set_yscale("log")
    ax.set_xlim(spec.x_bounds)
    ax.set_ylim(spec.y_bounds)

    set_feasibility_ticks(ax, spec.x_bounds[1])
    if spec.y_scale == "log":
        set_y_log_ticks(ax, y_grid_min, y_grid_max)
    elif spec.y_bounds[1] <= 0.5:
        ax.set_yticks(
            [OPT_DISPLAY_FLOOR, 0.1, 0.2, 0.3, 0.4, 0.5],
            labels=["0", "0.1", "0.2", "0.3", "0.4", "0.5"],
        )
    elif spec.y_bounds[1] <= 1.0:
        ax.set_yticks(
            [OPT_DISPLAY_FLOOR, 0.25, 0.5, 0.75, 1.0],
            labels=["0", "0.25", "0.5", "0.75", "1"],
        )
    else:
        ax.set_yticks(
            [OPT_DISPLAY_FLOOR, 1.0, 2.0, 3.0, 4.0, 5.0],
            labels=["0", "1", "2", "3", "4", "5"],
        )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    ax.grid(False)
    ax.tick_params(direction="out", colors=DARK)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK)
    handles = [
        Line2D(
            [0],
            [0],
            marker=method.marker,
            linestyle="none",
            markerfacecolor=method.color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=4.2,
            label=method.label,
        )
        for method in spec.methods
    ]
    ax.legend(
        handles=handles,
        loc=spec.legend_loc,
        ncol=2 if len(handles) == 4 else 1,
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.65,
        borderaxespad=0.35,
        labelcolor=DARK,
        fontsize=FONT_SIZE,
    )

    for method in spec.methods:
        annotate_maximum(ax, spec, method, groups[method.key])


def make_figure(spec: ExperimentSpec, groups: dict[str, np.ndarray]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(89 / 25.4, 72 / 25.4))
    style_panel(ax, spec, groups)
    ax.set_xlabel("Feasibility error")
    ax.set_ylabel("Optimality error")
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.17, top=0.985)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-preview-dir",
        type=Path,
        default=None,
        help="Optional temporary directory for Python-only PNG visual QA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_groups = [load_experiment(spec) for spec in EXPERIMENTS]
    apply_publication_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec, groups in zip(EXPERIMENTS, all_groups):
        fig = make_figure(spec, groups)
        output_path = OUTPUT_DIR / f"{spec.title}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        if args.qa_preview_dir is not None:
            args.qa_preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = args.qa_preview_dir / f"{spec.title}.png"
            fig.savefig(preview_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        print(f"Created {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
