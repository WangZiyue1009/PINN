"""Generate the pretraining feasibility-optimality phase portrait."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import median_filter, uniform_filter1d


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "pretrainnet_20260504_195204.json"
OUT_DIR = ROOT / "figures_converged" / "pretrain_error_options"


FEAS = "#1764AB"
OPT = "#D17A22"
FEAS_LIGHT = "#9CC4E4"
OPT_LIGHT = "#F1C38D"
NEUTRAL = "#5B6573"
LIGHT_GREY = "#E8EBEF"
FINAL_BAND = "#EEF1F4"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13.5,
        "axes.labelsize": 14.4,
        "axes.titlesize": 15.3,
        "xtick.labelsize": 12.6,
        "ytick.labelsize": 12.6,
        "legend.fontsize": 12.6,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


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
    iteration = np.arange(len(feasibility))
    return iteration, feasibility, optimality


def smooth(values: np.ndarray, window: int = 101) -> np.ndarray:
    return uniform_filter1d(values, size=window, mode="nearest")


def start_preserving_smooth(values: np.ndarray, window: int = 51) -> np.ndarray:
    """Causal moving mean that preserves the raw start and averages the endpoint."""
    if window < 1:
        raise ValueError("window must be a positive integer")
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    out = np.empty_like(values, dtype=float)
    for idx in range(len(values)):
        left = max(0, idx - window + 1)
        right = idx + 1
        out[idx] = (cumulative[right] - cumulative[left]) / (right - left)
    return out


def rolling_summary(
    values: np.ndarray, window: int = 201
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    q10, q90 = np.quantile(windows, [0.10, 0.90], axis=1)
    med = median_filter(values, size=window, mode="nearest")
    return med, q10, q90


def rolling_cv(values: np.ndarray, window: int = 201) -> np.ndarray:
    mean = uniform_filter1d(values, size=window, mode="nearest")
    second = uniform_filter1d(values**2, size=window, mode="nearest")
    std = np.sqrt(np.maximum(second - mean**2, 0))
    return std / np.maximum(mean, np.finfo(float).eps)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#000000")
    ax.xaxis.label.set_color("#000000")
    ax.yaxis.label.set_color("#000000")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=16.2,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def option_1_overlay(
    iteration: np.ndarray, feas: np.ndarray, opt: np.ndarray
) -> None:
    """Single-panel overlay: compact and immediately readable."""
    f_s = smooth(feas)
    o_s = smooth(opt)
    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    ax.plot(iteration, feas, color=FEAS_LIGHT, lw=0.45, alpha=0.38)
    ax.plot(iteration, opt, color=OPT_LIGHT, lw=0.45, alpha=0.42)
    ax.plot(iteration, f_s, color=FEAS, lw=1.7, label="Feasibility error")
    ax.plot(iteration, o_s, color=OPT, lw=1.7, label="Optimality error")
    ax.axvspan(iteration[int(0.9 * len(iteration))], iteration[-1], color=FINAL_BAND, zorder=0)
    ax.text(
        0.945,
        0.80,
        "Final 10%",
        transform=ax.transAxes,
        ha="center",
        color=NEUTRAL,
        fontsize=6.8,
    )
    ax.set(xlabel="Training iteration", ylabel="Error", xlim=(0, iteration[-1]), ylim=(0, None))
    ax.legend(loc="upper right", ncol=2, handlelength=2.8, columnspacing=1.2)
    style_axis(ax)
    save_figure(fig, "option_1_overlay")


def option_2_log_facets(
    iteration: np.ndarray, feas: np.ndarray, opt: np.ndarray
) -> None:
    """Log-scale facets: exposes relative convergence over several orders."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 4.8),
        sharex=True,
        constrained_layout=True,
    )
    final_start = iteration[int(0.9 * len(iteration))]
    specs = [
        (axes[0], feas, FEAS, FEAS_LIGHT, "Feasibility error"),
        (axes[1], opt, OPT, OPT_LIGHT, "Optimality error"),
    ]
    for idx, (ax, values, color, light, label) in enumerate(specs):
        positive = values.copy()
        positive[positive <= 0] = np.nan
        smoothed = smooth(values)
        smoothed[smoothed <= 0] = np.nan
        ax.plot(iteration, positive, color=light, lw=0.45, alpha=0.45)
        ax.plot(iteration, smoothed, color=color, lw=1.65)
        ax.axvspan(final_start, iteration[-1], color=FINAL_BAND, zorder=0)
        final_median = np.nanmedian(positive[int(0.9 * len(values)) :])
        ax.axhline(final_median, color=color, lw=0.8, ls=(0, (3, 2)), alpha=0.85)
        ax.text(
            0.985,
            0.83,
            f"Final median = {final_median:.3g}",
            transform=ax.transAxes,
            ha="right",
            color=color,
            fontsize=6.8,
        )
        ax.set_yscale("log")
        ax.set_ylabel(label)
        panel_label(ax, chr(ord("a") + idx))
        style_axis(ax)
    axes[1].set_xlabel("Training iteration")
    axes[1].set_xlim(0, iteration[-1])
    save_figure(fig, "option_2_log_facets")


def option_3_stability(
    iteration: np.ndarray, feas: np.ndarray, opt: np.ndarray
) -> None:
    """Rolling quantiles and variability: focuses on stability rather than spikes."""
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72])
    ax_f = fig.add_subplot(gs[0, 0])
    ax_o = fig.add_subplot(gs[0, 1], sharex=ax_f)
    ax_cv = fig.add_subplot(gs[1, :], sharex=ax_f)

    for ax, values, color, light, title, label in [
        (ax_f, feas, FEAS, FEAS_LIGHT, "Feasibility error", "a"),
        (ax_o, opt, OPT, OPT_LIGHT, "Optimality error", "b"),
    ]:
        med, q10, q90 = rolling_summary(values)
        ax.fill_between(iteration, q10, q90, color=light, alpha=0.55, linewidth=0)
        ax.plot(iteration, med, color=color, lw=1.55)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Error")
        panel_label(ax, label)
        style_axis(ax)

    ax_f.legend(
        handles=[
            Line2D([0], [0], color=FEAS, lw=1.55, label="Rolling median"),
            Patch(facecolor=FEAS_LIGHT, alpha=0.55, edgecolor="none", label="10th-90th percentile"),
        ],
        loc="upper right",
        handlelength=2.2,
    )

    cv_f = rolling_cv(feas)
    cv_o = rolling_cv(opt)
    ax_cv.plot(iteration, cv_f, color=FEAS, lw=1.25, label="Feasibility error")
    ax_cv.plot(iteration, cv_o, color=OPT, lw=1.25, label="Optimality error")
    ax_cv.axhline(1.0, color="#888888", lw=0.7, ls=(0, (3, 2)))
    ax_cv.set(
        xlabel="Training iteration",
        ylabel="Rolling coefficient\nof variation",
        xlim=(0, iteration[-1]),
    )
    ax_cv.legend(loc="upper right", ncol=2)
    panel_label(ax_cv, "c")
    style_axis(ax_cv)
    save_figure(fig, "option_3_stability")


def option_4_phase_space(
    iteration: np.ndarray, feas: np.ndarray, opt: np.ndarray
) -> None:
    """Single-panel joint error phase portrait."""
    # The causal window preserves the raw start and averages the final 51 iterations.
    # A symmetric-log y-axis retains the genuine zero optimality error at iteration 0.
    f_s = start_preserving_smooth(feas, 51)
    o_s = start_preserving_smooth(opt, 51)
    # Keep both errors in their original units; only the feasibility axis is logarithmic.
    o_phase = o_s

    fig, ax_phase = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    points = np.column_stack([f_s, o_phase]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(iteration[0], iteration[-1])
    lc = LineCollection(segments, cmap="viridis", norm=norm, linewidth=1.35, alpha=0.9)
    lc.set_array(iteration[:-1])
    ax_phase.add_collection(lc)
    ax_phase.scatter(
        f_s[0], o_phase[0], s=42, facecolor="white", edgecolor="#222222", linewidth=1.1, zorder=4
    )
    ax_phase.scatter(
        f_s[-1],
        o_phase[-1],
        s=48,
        marker="s",
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.2,
        zorder=4,
    )
    ax_phase.annotate(
        "Start",
        (f_s[0], o_phase[0]),
        xytext=(0, 12),
        ha="center",
        va="bottom",
        textcoords="offset points",
    )
    ax_phase.annotate(
        "End",
        (f_s[-1], o_phase[-1]),
        xytext=(12, 0),
        ha="left",
        va="center",
        textcoords="offset points",
    )
    ax_phase.set_xscale("log")
    ax_phase.set_xlim(1e-2, 1e0)
    ax_phase.set_ylim(0, 0.12)
    ax_phase.set_xticks([1e-2, 1e-1, 1e0], labels=["0.01", "0.1", "1"])
    y_ticks = np.arange(0, 0.121, 0.02)
    ax_phase.set_yticks(
        y_ticks,
        labels=["0" if np.isclose(value, 0) else f"{value:.2f}" for value in y_ticks],
    )
    ax_phase.set(
        xlabel="Feasibility error",
        ylabel="Optimality error",
    )
    style_axis(ax_phase)
    for side in ("top", "right"):
        ax_phase.spines[side].set_visible(True)
        ax_phase.spines[side].set_color("#333333")
        ax_phase.spines[side].set_linewidth(0.7)

    cax = ax_phase.inset_axes([0.06, 0.075, 0.44, 0.035])
    cbar = fig.colorbar(lc, cax=cax, orientation="horizontal")
    cbar.set_label("Training iteration")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.labelpad = 4
    cbar.outline.set_linewidth(0.6)

    save_figure(fig, "option_4_phase_space")


def export_source_data(
    iteration: np.ndarray, feas: np.ndarray, opt: np.ndarray
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "pretrain_error_source_data.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "feasibility_error", "optimality_error"])
        writer.writerows(zip(iteration, feas, opt))


def make_overview() -> None:
    names = [
        ("Option 1 | Overlay", "option_1_overlay.png"),
        ("Option 2 | Log facets", "option_2_log_facets.png"),
        ("Option 3 | Stability", "option_3_stability.png"),
        ("Option 4 | Phase space", "option_4_phase_space.png"),
    ]
    panels = []
    for title, filename in names:
        image = Image.open(OUT_DIR / filename).convert("RGB")
        image.thumbnail((1500, 1000), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (1520, 1060), "white")
        draw = ImageDraw.Draw(panel)
        times_path = Path("C:/Windows/Fonts/times.ttf")
        try:
            font = ImageFont.truetype(str(times_path), 34)
        except OSError:
            font = ImageFont.load_default(size=34)
        draw.text((20, 14), title, fill="#202020", font=font)
        panel.paste(image, ((1520 - image.width) // 2, 58))
        panels.append(panel)
    sheet = Image.new("RGB", (3040, 2120), "white")
    for idx, panel in enumerate(panels):
        sheet.paste(panel, ((idx % 2) * 1520, (idx // 2) * 1060))
    sheet.save(OUT_DIR / "all_options_overview.png", dpi=(200, 200))


def write_readme(iteration: np.ndarray, feas: np.ndarray, opt: np.ndarray) -> None:
    phase_end = np.column_stack(
        [start_preserving_smooth(feas, 51), start_preserving_smooth(opt, 51)]
    )[-1]
    text = f"""# Pretraining error phase portrait

The output contains a single joint feasibility-optimality phase portrait.

- Data source: the top-level feas_error and opt_error arrays.
- Feasibility-error axis: logarithmic, 10^-2 to 10^0.
- Optimality-error axis: linear, 0 to 0.12.
- Start marker: raw iteration 0.
- End marker: mean of the final 51 iterations ({phase_end[0]:.6g}, {phase_end[1]:.6g}).

Data summary

- Iterations: {len(iteration):,}
- Smoothing: start-preserving causal moving mean (up to 51 iterations).

Export: PDF only.
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    iteration, feas, opt = load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "option_1_*",
        "option_2_*",
        "option_3_*",
        "all_options_overview.png",
        "*.svg",
        "*.tiff",
        "*.png",
    ):
        for old_output in OUT_DIR.glob(pattern):
            old_output.unlink()
    export_source_data(iteration, feas, opt)
    option_4_phase_space(iteration, feas, opt)
    write_readme(iteration, feas, opt)
    print(f"Created the single-panel phase portrait in: {OUT_DIR}")


if __name__ == "__main__":
    main()
