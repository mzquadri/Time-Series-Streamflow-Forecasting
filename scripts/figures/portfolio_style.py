"""Shared light-ground styling for this repository's figures."""

from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt

#: tEXt key holding the findings a figure was drawn under.
FINDINGS_KEY = "RunFindings"

PAPER = "#FFFFFF"
INK = "#111827"
MUTED = "#4B5563"
FAINT = "#9CA3AF"
HAIR = "#E5E7EB"

BLUE, BLUE_SOFT = "#2563EB", "#BFDBFE"
GREEN, GREEN_SOFT = "#059669", "#A7F3D0"
AMBER, AMBER_SOFT = "#D97706", "#FDE68A"
RED, RED_SOFT = "#DC2626", "#FECACA"
SLATE, SLATE_SOFT = "#64748B", "#CBD5E1"

CYCLE = [BLUE, GREEN, AMBER, SLATE, RED]

#: Seven distinct hues, for the seven series. Kept light and non-neon.
BLUE_BG, GREEN_BG, AMBER_BG = "#EFF6FF", "#ECFDF5", "#FFFBEB"
TEAL, TEAL_SOFT = "#0891B2", "#A5F3FC"
FONTS = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.dpi": 200,
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": FONTS,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": HAIR,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })


def clean(ax, *, left: bool = True, bottom: bool = True, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=0)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, color=HAIR, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def fits(fig, artist, *, margin: float = 0.02) -> tuple[bool, bool]:
    """Whether a drawn text artist stays inside the canvas, horizontally and vertically.

    Both directions are checked. Only checking the width let a caption run off the
    bottom edge unnoticed more than once.
    """
    fig.canvas.draw()
    box = artist.get_window_extent(fig.canvas.get_renderer())
    horizontal = box.x1 <= fig.bbox.x1 * (1.0 - margin) and box.x0 >= 0.0
    vertical = box.y0 >= fig.bbox.y1 * 0.008
    return horizontal, vertical


def title_block(fig, title: str, subtitle: str = "", *, y: float = 0.96,
                size: float = 20, x: float = 0.065) -> None:
    fig.text(x, y, title, fontsize=size, color=INK, fontweight="600", va="top")
    if subtitle:
        gap = (size * 1.45) / (fig.get_figheight() * 72.0)
        fig.text(x, y - gap, subtitle, fontsize=11.2, color=MUTED, va="top",
                 linespacing=1.55)


def footnote(fig, lines, *, y: float = 0.085, x: float = 0.065, size: float = 9.4) -> None:
    step = (size * 1.75) / (fig.get_figheight() * 72.0)
    for i, line in enumerate(lines):
        artist = fig.text(x, y - i * step, line, fontsize=size, color=FAINT, va="top",
                          linespacing=1.5)
        horizontal, vertical = fits(fig, artist)
        if not horizontal:
            print(f"    caption too wide for the canvas: {line[:58]}...")
        if not vertical:
            print(f"    caption runs off the bottom edge: {line[:58]}...")


def save(fig, out_dir, name: str, *, findings: dict | None = None) -> None:
    """Write the figure, recording the findings it was drawn under.

    What is stamped is the set of conclusions, not the numbers. The numbers here
    are not all reproducible across machines: XGBoost's RMSE is 3.12 on the
    machine that produced the committed run and 3.42 on the CI runner, because
    the tree builder's floating point behaviour depends on the platform, and the
    README says so. Stamping values would make a figure drawn here disagree with
    a run made there for a reason that is not a disagreement.

    The conclusions do not move, and those are what the figures assert. The
    near-tie between persistence and the drift baseline, 0.0002 apart in RMSE, is
    deliberately left out: their relative order is not something this repository
    claims is stable.

    matplotlib merges this with its own defaults, so the Software entry naming
    the version that rendered the file is still written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = None if findings is None else {
        FINDINGS_KEY: json.dumps(findings, sort_keys=True)}
    fig.savefig(out_dir / f"{name}.png", metadata=metadata)
    plt.close(fig)
    print(f"  wrote {name}.png")
