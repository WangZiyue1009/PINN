"""Plot the full-network feasibility-optimality training phase portrait."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "fullnet_20260506_025651.json"
OUT_DIR = ROOT / "figures_converged" / "fullnet_phase_space"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13.5,
        "axes.labelsize": 14.4,
        "xtick.labelsize": 12.6,
        "ytick.labelsize": 12.6,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def start_preserving_mean(values: np.ndarray, window: int = 300) -> np.ndarray:
    """Causal moving mean: iteration 1 is raw; later points use up to 300 values."""
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    out = np.empty_like(values, dtype=float)
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        right = idx + 1
        out[idx] = (cumulative[right] - cumulative[left]) / (right - left)
    return out


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    feasibility = np.asarray(payload["feas_error"], dtype=float)
    optimality = np.asarray(payload["opt_error"], dtype=float)
    if feasibility.ndim != 1 or optimality.ndim != 1:
        raise ValueError("feas_error and opt_error must be one-dimensional")
    if len(feasibility) != len(optimality):
        raise ValueError("feas_error and opt_error must have equal lengths")
    if not np.isfinite(feasibility).all() or not np.isfinite(optimality).all():
        raise ValueError("feas_error or opt_error contains non-finite values")
    iteration = np.arange(1, len(feasibility) + 1)
    return iteration, feasibility, optimality


def export_source_data(
    iteration: np.ndarray, feasibility: np.ndarray, optimality: np.ndarray
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "fullnet_phase_space_source_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "feasibility_error", "optimality_error"])
        writer.writerows(zip(iteration, feasibility, optimality))


def plot_phase_space(
    iteration: np.ndarray, feasibility: np.ndarray, optimality: np.ndarray
) -> dict[int, tuple[float, float]]:
    f_s = start_preserving_mean(feasibility, 300)
    o_s = start_preserving_mean(optimality, 300)
    # Values at or below 10^-6 share the displayed zero boundary on the log axis.
    f_plot = np.maximum(f_s, 1e-6)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    points = np.column_stack([f_plot, o_s]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(0, iteration[-1])
    path = LineCollection(
        segments, cmap="viridis", norm=norm, linewidth=1.25, alpha=0.9
    )
    path.set_array(iteration[:-1])
    ax.add_collection(path)

    ax.scatter(
        f_plot[0],
        o_s[0],
        s=42,
        facecolor="white",
        edgecolor="#000000",
        linewidth=1.1,
        zorder=5,
    )
    ax.annotate(
        "Start",
        (f_plot[0], o_s[0]),
        xytext=(0, -14),
        ha="center",
        va="top",
        textcoords="offset points",
    )

    phase_iterations = (6000, 16000, 36000)
    phase_offsets = ((-12, 0), (0, -14), (0, 14))
    phase_alignments = (("right", "center"), ("center", "top"), ("center", "bottom"))
    phase_points: dict[int, tuple[float, float]] = {}
    for phase_number, (phase_iteration, offset, alignment) in enumerate(
        zip(phase_iterations, phase_offsets, phase_alignments), start=1
    ):
        idx = phase_iteration - 1
        x_value, y_value = f_s[idx], o_s[idx]
        x_display = max(float(x_value), 1e-6)
        phase_points[phase_iteration] = (float(x_value), float(y_value))
        ax.scatter(
            x_display,
            y_value,
            s=48,
            marker="s",
            facecolor="white",
            edgecolor="#000000",
            linewidth=1.2,
            zorder=6,
        )
        ax.annotate(
            f"Phase {phase_number} End",
            (x_display, y_value),
            xytext=offset,
            ha=alignment[0],
            va=alignment[1],
            textcoords="offset points",
            zorder=7,
        )

    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e-1)
    ax.set_ylim(0, 0.25)
    ax.set_xticks(
        [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        labels=["0", r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"],
    )
    y_ticks = np.arange(0, 0.251, 0.05)
    ax.set_yticks(
        y_ticks,
        labels=["0" if np.isclose(value, 0) else f"{value:.2f}" for value in y_ticks],
    )
    ax.set_xlabel("Feasibility error")
    ax.set_ylabel("Optimality error")
    ax.tick_params(colors="#000000")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.7)

    cax = ax.inset_axes([0.56, 0.84, 0.38, 0.035])
    cbar = fig.colorbar(path, cax=cax, orientation="horizontal")
    cbar.set_label("Training iteration")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.labelpad = 4
    cbar.set_ticks(np.arange(0, iteration[-1] + 1, 12000))
    cbar.outline.set_linewidth(0.6)

    fig.savefig(OUT_DIR / "fullnet_phase_space.pdf", bbox_inches="tight")
    plt.close(fig)
    return phase_points


def write_readme(phase_points: dict[int, tuple[float, float]]) -> None:
    lines = [
        "# Full-network phase portrait",
        "",
        "Data source: the top-level feas_error and opt_error arrays.",
        "The trajectory uses a start-preserving causal mean of up to 300 iterations.",
        "The feasibility axis is logarithmic; values at or below 10^-6 share the tick labelled 0.",
        "",
        "Phase markers (smoothed coordinates):",
    ]
    for iteration, (feasibility, optimality) in phase_points.items():
        lines.append(
            f"- Iteration {iteration}: feasibility={feasibility:.8g}, "
            f"optimality={optimality:.8g}"
        )
    lines.extend(["", "Export: PDF only.", ""])
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.svg", "*.tiff", "*.png"):
        for old_output in OUT_DIR.glob(pattern):
            old_output.unlink()
    iteration, feasibility, optimality = load_data()
    export_source_data(iteration, feasibility, optimality)
    phase_points = plot_phase_space(iteration, feasibility, optimality)
    write_readme(phase_points)
    print(f"Created full-network phase portrait in: {OUT_DIR}")


if __name__ == "__main__":
    main()
