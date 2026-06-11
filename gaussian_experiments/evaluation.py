"""Evaluation metrics: compare FK product samples against ground truth."""

from __future__ import annotations

from typing import Any

import numpy as np
import ot
from scipy.spatial.distance import cdist

Distribution = Any


def sliced_w2(
    samples_a: np.ndarray, samples_b: np.ndarray, n_projections: int = 2000, seed: int | None = None
) -> float:
    return float(
        ot.sliced_wasserstein_distance(
            samples_a,
            samples_b,
            n_projections=n_projections,
            p=2,
            seed=seed,
        )
    )


def mmd_rbf(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    *,
    gamma: float | None = None,
    max_n: int = 5000,
    seed: int | None = None,
) -> float:
    """Unbiased MMD^2 with RBF kernel. Median heuristic when gamma is None."""
    rng = np.random.default_rng(seed)
    samples_a = np.asarray(samples_a, dtype=np.float64)
    samples_b = np.asarray(samples_b, dtype=np.float64)

    if len(samples_a) > max_n:
        samples_a = samples_a[rng.choice(len(samples_a), max_n, replace=False)]
    if len(samples_b) > max_n:
        samples_b = samples_b[rng.choice(len(samples_b), max_n, replace=False)]

    d_xx = cdist(samples_a, samples_a, metric="sqeuclidean")
    d_yy = cdist(samples_b, samples_b, metric="sqeuclidean")
    d_xy = cdist(samples_a, samples_b, metric="sqeuclidean")

    if gamma is None:
        med = float(np.median(np.sqrt(d_xy)))
        gamma = 1.0 / (2.0 * max(med, 1e-8) ** 2)

    k_xx = np.exp(-gamma * d_xx)
    k_yy = np.exp(-gamma * d_yy)
    k_xy = np.exp(-gamma * d_xy)

    n, m = len(samples_a), len(samples_b)
    mmd2 = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2.0 * k_xy.mean()
    )
    return float(mmd2)


def compute_distribution_metrics(
    samples: np.ndarray,
    gt_dist,
    n_gt_samples: int = 5000,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute Sliced-W2 and MMD^2 between samples and gt_dist."""
    gt_samples = gt_dist.sample(n_gt_samples)
    return {
        "sw2": sliced_w2(samples, gt_samples, seed=seed),
        "mmd2": mmd_rbf(samples, gt_samples, seed=seed),
    }
