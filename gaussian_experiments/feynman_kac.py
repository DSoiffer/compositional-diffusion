"""Feynman-Kac corrector (FKC) sampler for diffusion product distributions."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from tqdm import trange

from noise_schedule import VPSchedule

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _batched_systematic_resample(log_weights: torch.Tensor) -> torch.Tensor:
    """Batched systematic resample.

    log_weights: (N, K) -> indices: (N, K) where each row is an independent
    systematic resample of K particles using a per-row uniform draw. Slot
    order is shuffled within each row so any single slot is an unbiased
    stratified draw from the weighted distribution.
    """
    N, K = log_weights.shape
    w = torch.softmax(log_weights, dim=1)
    bins = torch.cumsum(w, dim=1)
    u0 = torch.rand(N, 1, device=log_weights.device) / K
    u = u0 + torch.arange(K, device=log_weights.device, dtype=torch.float32) / K
    idx = torch.searchsorted(bins, u).clamp(max=K - 1)
    perm = torch.argsort(torch.rand(N, K, device=log_weights.device), dim=1)
    return idx.gather(1, perm)


def feynman_kac_sample(
    score_fns: list[ScoreFn],
    betas: list[float],
    schedule: VPSchedule,
    ndim: int,
    n_output: int,
    *,
    n_particles: int = 8,
    n_steps: int = 500,
    g_clip: float | None = 15.0,
    device: torch.device | str = "cuda",
    verbose: bool = True,
) -> torch.Tensor:
    """Sample n_output samples from p(x) propto prod q_i(x)^{beta_i} via FKC.

    Runs n_output independent swarms of n_particles (=K) particles each,
    vectorized as a single batch of size N*K through the reverse SDE.
    Resampling is applied per-swarm at every step. At the end, each swarm
    draws one sample from its weighted ensemble.
    """
    device = torch.device(device)
    dt = 1.0 / n_steps
    beta_sum = sum(betas)
    K, N = n_particles, n_output
    B = N * K

    x = torch.randn(B, ndim, device=device)
    log_w = torch.zeros(N, K, device=device)

    cached_scores: list[torch.Tensor] | None = None

    iterator = trange(n_steps) if verbose else range(n_steps)
    with torch.no_grad():
        for step in iterator:
            t_val = max(step * dt, 1e-5)
            t_val = min(t_val, 1.0 - 1e-5)
            t = torch.full((B, 1), t_val, device=device)

            # Evaluate scores
            if cached_scores is None:
                scores = [fn(t, x) for fn in score_fns]
            else:
                scores = cached_scores
            weighted_score = sum(b * s for b, s in zip(betas, scores))

            # Reverse SDE step
            a_t = schedule.drift_coeff(t)
            sigma_sde = schedule.diffusion(t)
            drift = a_t * x + sigma_sde**2 * weighted_score
            x = x + drift * dt + sigma_sde * np.sqrt(dt) * torch.randn_like(x)

            # FKC weight increment
            t_next = max(min((step + 1) * dt, 1.0 - 1e-5), 1e-5)
            t_n = torch.full((B, 1), t_next, device=device)

            scores_new = [fn(t_n, x) for fn in score_fns]
            ws_new = sum(b * s for b, s in zip(betas, scores_new))
            sigma_t = schedule.diffusion(t_n)
            s2_scalar = (sigma_t**2).squeeze(1)

            norm_ws_sq = (ws_new**2).sum(dim=1)
            weighted_norm_sq = sum(b * (s**2).sum(dim=1) for b, s in zip(betas, scores_new))
            div_f = ndim * schedule.drift_coeff(t_n).squeeze(1)

            g = (beta_sum - 1.0) * div_f + 0.5 * s2_scalar * (norm_ws_sq - weighted_norm_sq)
            if g_clip is not None:
                g = g.clamp(-g_clip, g_clip)
            log_w = log_w + (g * dt).view(N, K)

            cached_scores = scores_new

            # Per-swarm resampling
            idx_local = _batched_systematic_resample(log_w)
            offset = (torch.arange(N, device=device) * K)[:, None]
            flat_idx = (idx_local + offset).view(-1)
            x = x[flat_idx].clone()
            log_w = torch.zeros_like(log_w)
            cached_scores = [s[flat_idx] for s in cached_scores]

    final_idx_local = _batched_systematic_resample(log_w)[:, 0]
    pick_global = final_idx_local + torch.arange(N, device=device) * K
    return x[pick_global]


@torch.no_grad()
def naive_composed_sample(
    score_fns: list[ScoreFn],
    betas: list[float],
    schedule: VPSchedule,
    ndim: int,
    n_output: int,
    *,
    n_steps: int = 500,
    device: torch.device | str = "cuda",
    verbose: bool = False,
) -> torch.Tensor:
    """Naive composed-score reverse-SDE sampling: no FKC correction.

    At each step, evaluates all score functions at the shared point x_t,
    forms s_combined = sum_i beta_i s_i(x_t, t), and takes one step of 
    the reverse SDE. It is equivalent to the FKC sampler with K=1 (but 
    more efficient since it does not perform unnecessary steps).
    """
    device = torch.device(device)
    dt = 1.0 / n_steps
    sqrt_dt = float(np.sqrt(dt))
    x = torch.randn(n_output, ndim, device=device)
    t_buf = torch.empty(n_output, 1, device=device)

    iterator = trange(n_steps) if verbose else range(n_steps)
    for step in iterator:
        t_val = max(min(step * dt, 1.0 - 1e-5), 1e-5)
        t_buf.fill_(t_val)
        scores = [fn(t_buf, x) for fn in score_fns]
        weighted_score = sum(b * s for b, s in zip(betas, scores))
        a_t = schedule.drift_coeff(t_buf)
        sigma_sde = schedule.diffusion(t_buf)
        drift = a_t * x + sigma_sde**2 * weighted_score
        x = x + drift * dt + sigma_sde * sqrt_dt * torch.randn_like(x)

    return x
