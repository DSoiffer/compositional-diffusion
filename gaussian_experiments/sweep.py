"""Sweep over (training_size, particle_count) for the FKC sampler.

For each (training_size, K) pair, runs n_runs independent trials, aggregates
the metrics to mean/std, prints tables, and writes a grid plot of samples
plus a LaTeX table file.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from feynman_kac import feynman_kac_sample
from evaluation import compute_distribution_metrics

TrainFn = Callable[[Any], list]


METRIC_KEYS = ("sw2", "mmd2")
METRIC_TITLES_TXT = {
    "sw2": "Sliced-W2",
    "mmd2": "MMD^2",
}
METRIC_TITLES_TEX = {
    "sw2": r"$\mathrm{SW}_2$",
    "mmd2": r"$\mathrm{MMD}^2$",
}


def _aggregate(metric_dicts: list[dict]) -> dict[str, tuple[float, float]]:
    keys = metric_dicts[0].keys()
    return {
        k: (
            float(np.mean([m[k] for m in metric_dicts])),
            float(np.std([m[k] for m in metric_dicts])),
        )
        for k in keys
    }


def _fmt_cell(mean: float, std: float, n_runs: int) -> str:
    if n_runs > 1:
        return f"{mean:.4f}+/-{std:.4f}"
    return f"{mean:.4f}"


def _fmt_cell_latex(mean: float, std: float, n_runs: int) -> str:
    if n_runs > 1:
        return f"${mean:.4f} \\pm {std:.4f}$"
    return f"${mean:.4f}$"


def sweep_training_and_particles(
    train_and_get_score_fns: TrainFn,
    betas: list[float],
    schedule,
    gt_dist,
    device,
    *,
    training_sizes: Sequence[Any],
    particle_counts: Sequence[Any],
    n_output: int = 1000,
    n_runs: int = 1,
    out_path: str | None = None,
    out_tex_path: str | None = None,
    tex_caption_prefix: str = "",
    grid_range: float = 12.0,
    ndim: int = 2,
    seed: int = 1,
    n_steps: int = 500,
    plot_color: str = "red",
    ts_label: Callable[[Any], str] | None = None,
    pc_label: Callable[[Any], str] | None = None,
    title: str = "Training-vs-particles sweep",
    n_gt_samples: int = 5000,
    overlay_means: np.ndarray | None = None,
) -> dict:
    """Run feynman_kac_sample over a (training_size x particle_count) grid.

    train_and_get_score_fns: training_size -> list[score_fn]. training_size
    may be any value (int sample count, "analytical", etc.).
    particle_counts: K values (per-swarm SMC ensemble size).
    n_output: number of output samples per cell.
    n_runs: independent runs per cell (re-seeded and re-trained per run).
    """
    if ts_label is None:
        ts_label = lambda ts: str(ts)
    if pc_label is None:
        pc_label = lambda pc: f"K={pc}"

    metrics_by_cell: dict[tuple[Any, Any], list[dict]] = {}
    samples_by_cell: dict[tuple[Any, Any], np.ndarray] = {}

    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

        print("\n" + "#" * 70)
        print(f"# Sweep run {run_idx + 1}/{n_runs} (seed={run_seed})")
        print("#" * 70)

        for ts in training_sizes:
            print(f"\n    training_size={ts_label(ts)}")
            score_fns = train_and_get_score_fns(ts)

            for pc in particle_counts:
                x_final = feynman_kac_sample(
                    score_fns,
                    betas,
                    schedule,
                    ndim=ndim,
                    n_output=n_output,
                    n_particles=pc,
                    n_steps=n_steps,
                    device=device,
                    verbose=False,
                )
                samples = x_final.cpu().numpy()
                m = compute_distribution_metrics(
                    samples,
                    gt_dist,
                    n_gt_samples=n_gt_samples,
                    seed=run_seed,
                )
                metrics_by_cell.setdefault((ts, pc), []).append(m)
                if run_idx == 0:
                    samples_by_cell[(ts, pc)] = samples
                metric_str = "  ".join(f"{METRIC_TITLES_TXT[k]}={m[k]:.4f}" for k in METRIC_KEYS)
                print(f"  {pc_label(pc):<14} {metric_str}")

    aggregated = {cell: _aggregate(metrics_by_cell[cell]) for cell in metrics_by_cell}

    _print_table(aggregated, training_sizes, particle_counts, n_runs, ts_label, pc_label)

    if out_path is not None:
        _plot_grid(
            samples_by_cell,
            aggregated,
            training_sizes,
            particle_counts,
            ts_label,
            pc_label,
            grid_range,
            plot_color,
            out_path,
            title,
            overlay_means=overlay_means,
        )

    if out_tex_path is not None:
        _write_latex_tables(
            aggregated,
            training_sizes,
            particle_counts,
            n_runs,
            ts_label,
            pc_label,
            out_tex_path,
            caption_prefix=tex_caption_prefix,
        )

    return {
        "metrics": metrics_by_cell,
        "aggregated": aggregated,
        "samples": samples_by_cell,
    }


def _print_table(
    aggregated: dict,
    training_sizes: Sequence[Any],
    particle_counts: Sequence[Any],
    n_runs: int,
    ts_label: Callable[[Any], str],
    pc_label: Callable[[Any], str],
) -> None:
    suffix = " (mean +/- std)" if n_runs > 1 else ""
    cell_w = 22 if n_runs > 1 else 12
    pc_strs = [pc_label(pc) for pc in particle_counts]
    ts_strs = [ts_label(ts) for ts in training_sizes]
    pc_w = max(8, max(len(s) for s in pc_strs))
    col_w = max(cell_w, max(len(s) for s in ts_strs))

    for key in METRIC_KEYS:
        title = METRIC_TITLES_TXT[key]
        print(f"\n=== {title}{suffix}  " f"(rows = particles K, cols = training samples N) ===")
        header = f"  {'K \\ N':<{pc_w}} " + " ".join(f"{ts:>{col_w}}" for ts in ts_strs)
        print(header)
        print("  " + "-" * (pc_w + 1 + len(ts_strs) * (col_w + 1)))
        for pc, pc_s in zip(particle_counts, pc_strs):
            cells = [_fmt_cell(*aggregated[(ts, pc)][key], n_runs) for ts in training_sizes]
            print(f"  {pc_s:<{pc_w}} " + " ".join(f"{c:>{col_w}}" for c in cells))


def _build_metric_df(
    aggregated: dict,
    key: str,
    training_sizes: Sequence[Any],
    particle_counts: Sequence[Any],
    n_runs: int,
    ts_label: Callable[[Any], str],
    pc_label: Callable[[Any], str],
) -> pd.DataFrame:
    data: dict[str, list] = {r"$K \backslash N$": [pc_label(pc) for pc in particle_counts]}
    for ts in training_sizes:
        data[ts_label(ts)] = [
            _fmt_cell_latex(*aggregated[(ts, pc)][key], n_runs) for pc in particle_counts
        ]
    return pd.DataFrame(data)


def _write_latex_tables(
    aggregated: dict,
    training_sizes: Sequence[Any],
    particle_counts: Sequence[Any],
    n_runs: int,
    ts_label: Callable[[Any], str],
    pc_label: Callable[[Any], str],
    out_path: str,
    *,
    caption_prefix: str = "",
) -> None:
    n_cols = len(training_sizes)
    col_spec = "l|" + "c" * n_cols

    blocks: list[str] = []
    blocks.append("% Auto-generated by sweep.sweep_training_and_particles")
    blocks.append(f"% Rows = particle counts (K); cols = training sizes (N); n_runs = {n_runs}")
    if caption_prefix:
        blocks.append(f"% Context: {caption_prefix}")
    blocks.append("")

    run_note = f" (mean $\\pm$ std over {n_runs} runs)" if n_runs > 1 else ""
    sep = ": " if caption_prefix else ""

    for key in METRIC_KEYS:
        df = _build_metric_df(
            aggregated,
            key,
            training_sizes,
            particle_counts,
            n_runs,
            ts_label,
            pc_label,
        )
        cap = f"{caption_prefix}{sep}{METRIC_TITLES_TEX[key]}{run_note}"
        blocks.append(
            df.to_latex(
                index=False,
                escape=False,
                caption=cap,
                label=f"tab:sweep_{key}",
                position="h",
                column_format=col_spec,
            )
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(blocks))
    print(f"Saved LaTeX tables to {out_path}")


def _plot_grid(
    samples_by_cell: dict,
    aggregated: dict,
    training_sizes: Sequence[Any],
    particle_counts: Sequence[Any],
    ts_label: Callable[[Any], str],
    pc_label: Callable[[Any], str],
    grid_range: float,
    plot_color: str,
    out_path: str,
    title: str,
    overlay_means: np.ndarray | None = None,
) -> None:
    n_rows = len(training_sizes)
    n_cols = len(particle_counts)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows + 0.5),
        squeeze=False,
    )
    for i, ts in enumerate(training_sizes):
        for j, pc in enumerate(particle_counts):
            ax = axes[i, j]
            s = samples_by_cell[(ts, pc)]
            ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.3, color=plot_color)
            if overlay_means is not None:
                ax.scatter(
                    overlay_means[:, 0], overlay_means[:, 1],
                    marker="x", s=80, c="k", linewidths=2.0, alpha=0.9, zorder=10,
                )
            agg = aggregated[(ts, pc)]
            label_lines = []
            if "sw2" in agg:
                label_lines.append(f"SW2={agg['sw2'][0]:.3f}")
            if "mmd2" in agg:
                label_lines.append(f"MMD$^2$={agg['mmd2'][0]:.4f}")
            ax.text(
                0.02,
                0.98,
                "\n".join(label_lines),
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            )
            ax.set_xlim(-grid_range, grid_range)
            ax.set_ylim(-grid_range, grid_range)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(pc_label(pc), fontsize=11)
            if j == 0:
                ax.set_ylabel(ts_label(ts), fontsize=11)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")
