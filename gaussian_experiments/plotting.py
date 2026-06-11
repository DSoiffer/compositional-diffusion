"""Shared plotting helpers for the FKC experiments."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from evaluation import sliced_w2


def annotate_metric(ax, text: str) -> None:
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3),
    )


def fmt_sw2(metrics: dict) -> str:
    return f"SW2={metrics['sw2']:.4f}"


def print_metrics(prefix: str, metrics: dict) -> None:
    print(f"{prefix}SW2={metrics['sw2']:.4f}  MMD^2={metrics['mmd2']:.5f}")


def overlay_modes(ax, modes: np.ndarray | None) -> None:
    """Plot black-x markers at each row of modes (shape (M, 2)). No-op if None."""
    if modes is None or len(modes) == 0:
        return
    ax.scatter(
        modes[:, 0], modes[:, 1],
        marker="x", s=80, c="k", linewidths=2.0, alpha=0.9, zorder=10,
    )


def _format_axes(ax, grid_range: float) -> None:
    ax.set_xlim(-grid_range, grid_range)
    ax.set_ylim(-grid_range, grid_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_distributions_overview(
    true_samples: list[np.ndarray],
    learned_samples: list[np.ndarray],
    *,
    true_labels: list[str],
    learned_labels: list[str],
    colors: list[str],
    grid_range: float,
    suptitle: str = "",
    modes_per_panel: list[np.ndarray | None] | None = None,
    metric_seed: int | None = None,
):
    """2-row scatter grid.

    Top row: one panel per true distribution. Bottom row: one panel per
    learned distribution with SW2 vs the matching true samples overlaid.
    Mode markers are plotted in both rows when `modes_per_panel` is given
    (one entry per column).

    All input sample arrays are (N, 2) numpy arrays.
    """
    n_cols = len(true_samples)
    assert len(learned_samples) == n_cols
    assert len(true_labels) == n_cols and len(learned_labels) == n_cols
    assert len(colors) == n_cols
    if modes_per_panel is not None:
        assert len(modes_per_panel) == n_cols

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 11), squeeze=False)

    for j in range(n_cols):
        ax = axes[0, j]
        s = true_samples[j]
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.3, color=colors[j])
        if modes_per_panel is not None:
            overlay_modes(ax, modes_per_panel[j])
        ax.set_title(true_labels[j])

    for j in range(n_cols):
        ax = axes[1, j]
        s = learned_samples[j]
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.3, color=colors[j])
        if modes_per_panel is not None:
            overlay_modes(ax, modes_per_panel[j])
        ax.set_title(learned_labels[j])
        sw2 = sliced_w2(s, true_samples[j], seed=metric_seed)
        annotate_metric(ax, f"SW2={sw2:.4f}")

    for row in axes:
        for ax in row:
            _format_axes(ax, grid_range)

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.4)
    return fig


def plot_comparison_summary(
    gt_samples: np.ndarray,
    analytical_samples: np.ndarray,
    learned_samples: np.ndarray,
    *,
    analytical_metrics: dict,
    learned_metrics: dict,
    k_particles: int,
    grid_range: float,
    gt_title: str = "Ground truth",
    overlay_modes_arr: np.ndarray | None = None,
):
    """1x3: ground truth vs FKC (analytical scores) vs FKC (learned scores).

    Each panel scatter-plots the corresponding sample set with SW2 overlaid
    on the two FKC panels. Optional mode markers are drawn on all three.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    n = min(len(gt_samples), len(analytical_samples), len(learned_samples))

    axes[0].scatter(gt_samples[:n, 0], gt_samples[:n, 1],
                    s=2, alpha=0.3, color="green")
    axes[0].set_title(gt_title)

    axes[1].scatter(analytical_samples[:n, 0], analytical_samples[:n, 1],
                    s=2, alpha=0.3, color="teal")
    axes[1].set_title(f"FKC (analytical scores, K={k_particles})")
    annotate_metric(axes[1], fmt_sw2(analytical_metrics))

    axes[2].scatter(learned_samples[:n, 0], learned_samples[:n, 1],
                    s=2, alpha=0.3, color="red")
    axes[2].set_title(f"FKC (learned scores, K={k_particles})")
    annotate_metric(axes[2], fmt_sw2(learned_metrics))

    for ax in axes:
        overlay_modes(ax, overlay_modes_arr)
        _format_axes(ax, grid_range)

    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.4)
    return fig


def save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {path}")
