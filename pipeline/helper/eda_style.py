"""
eda_style.py — shared minimalist/modern matplotlib styling for all EDA plots.

One import + apply_style() gives every figure the same clean look: white canvas,
no top/right spines, a single light y-grid behind the data, muted slate text, and an
indigo/teal palette that matches the Hugging Face Space theme. Use finalize(ax, ...) to
de-spine + label a plot, and the SERIES palette for colors.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# palette (matches the app's indigo/slate Soft theme)
INK = "#1f2937"      # primary text (slate-800)
MUTED = "#6b7280"    # secondary text / ticks (slate-500)
GRID = "#e5e7eb"     # hairline grid / spines (slate-200)
PRIMARY = "#6366f1"  # indigo-500
SECONDARY = "#14b8a6"  # teal-500
ACCENT = "#f59e0b"   # amber-500
CORAL = "#f43f5e"    # rose-500
SERIES = [PRIMARY, SECONDARY, ACCENT, CORAL]
SEQ_CMAP = "mako"    # clean sequential colormap for heatmaps (seaborn)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 110,
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.labelcolor": MUTED,
        "axes.labelsize": 10,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def finalize(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    """De-spine, label, and apply a single light grid on the chosen axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return ax
