"""Distribution utilities: DiagonalGaussian, IsotropicGMM, and exact products
and ratios over them.

Both classes implement the duck-typed Distribution interface used by the
sampler and evaluation code: sample(n) and log_prob(x).
"""

import numpy as np
from scipy.special import logsumexp


class DiagonalGaussian:
    """Single Gaussian with diagonal covariance: N(mu, diag(variances))."""

    def __init__(self, mean, variances):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.variances = np.asarray(variances, dtype=np.float64)
        self.D = len(self.mean)
        assert self.variances.shape == (self.D,)

    @property
    def precisions(self):
        return 1.0 / self.variances

    def log_prob(self, x):
        x = np.atleast_2d(x)
        diff = x - self.mean
        mahal = -0.5 * np.sum(diff ** 2 / self.variances, axis=-1)
        log_norm = -0.5 * np.sum(np.log(2 * np.pi * self.variances))
        return log_norm + mahal

    def score(self, x):
        x = np.atleast_2d(x)
        return -(x - self.mean) / self.variances

    def sample(self, n):
        return (np.random.randn(n, self.D) * np.sqrt(self.variances)
                + self.mean)


def analytical_product_ratio_diagonal(*numerators, denominators=None):
    """Exact product/ratio of zero-mean DiagonalGaussians.

    Returns the DiagonalGaussian proportional to
        prod_i N(0, Sigma_i) / prod_j N(0, Sigma_j)
    """
    if denominators is None:
        denominators = []
    D = numerators[0].D
    prec = np.zeros(D)
    for g in numerators:
        prec += g.precisions
    for g in denominators:
        prec -= g.precisions
    if np.any(prec <= 0):
        raise ValueError(
            f"Product/ratio precision has non-positive entries: {prec}. "
            "The resulting distribution is improper."
        )
    return DiagonalGaussian(np.zeros(D), 1.0 / prec)


class IsotropicGMM:
    """GMM with shared isotropic covariance: p(x) = sum_k pi_k N(x; mu_k, var*I)."""

    def __init__(self, means, std=0.7, weights=None):
        self.means = np.asarray(means, dtype=np.float64)
        self.K, self.D = self.means.shape
        self.std = float(std)
        self.var = self.std ** 2
        if weights is None:
            self.weights = np.ones(self.K) / self.K
        else:
            w = np.asarray(weights, dtype=np.float64)
            self.weights = w / w.sum()
        self.log_weights = np.log(self.weights + 1e-300)

    def _log_component_densities(self, x):
        x = np.atleast_2d(x)
        diff = x[:, None, :] - self.means[None, :, :]
        mahal = -0.5 * np.sum(diff ** 2, axis=-1) / self.var
        log_norm = -0.5 * self.D * np.log(2 * np.pi * self.var)
        return self.log_weights[None, :] + log_norm + mahal

    def log_prob(self, x):
        return logsumexp(self._log_component_densities(x), axis=-1)

    def sample(self, n):
        comps = np.random.choice(self.K, size=n, p=self.weights)
        return np.random.randn(n, self.D) * self.std + self.means[comps]


def _exact_product_pair(gmm_a: IsotropicGMM, gmm_b: IsotropicGMM) -> IsotropicGMM:
    """Exact product of two IsotropicGMMs as a new IsotropicGMM with K_a*K_b
    components.

    N(x; mu_j, s_a^2 I) * N(x; mu_k, s_b^2 I) = c_{jk} N(x; mu_{jk}, s_new^2 I)
    with s_new^2 = s_a^2 s_b^2 / (s_a^2 + s_b^2),
    mu_{jk} = (s_b^2 mu_j + s_a^2 mu_k) / (s_a^2 + s_b^2),
    log c_{jk} = -D/2 log(2 pi (s_a^2 + s_b^2)) - ||mu_j - mu_k||^2 / (2 (s_a^2 + s_b^2)).
    """
    D = gmm_a.D
    sa2, sb2 = gmm_a.var, gmm_b.var
    s_sum = sa2 + sb2

    new_var = (sa2 * sb2) / s_sum
    new_std = np.sqrt(new_var)

    mu_j = gmm_a.means[:, None, :]
    mu_k = gmm_b.means[None, :, :]
    new_means = (sb2 * mu_j + sa2 * mu_k) / s_sum

    diff = mu_j - mu_k
    mahal = -np.sum(diff ** 2, axis=-1) / (2.0 * s_sum)
    log_c = -0.5 * D * np.log(2.0 * np.pi * s_sum) + mahal

    log_w = gmm_a.log_weights[:, None] + gmm_b.log_weights[None, :] + log_c
    log_w_flat = log_w.reshape(-1)
    new_means_flat = new_means.reshape(-1, D)

    log_w_flat -= logsumexp(log_w_flat)
    weights = np.exp(log_w_flat)

    return IsotropicGMM(new_means_flat, std=new_std, weights=weights)


def exact_gmm_product(*gmms: IsotropicGMM) -> IsotropicGMM:
    """Exact product of K IsotropicGMMs, folded pairwise. Result has
    K_1 * K_2 * ... components and normalized weights."""
    if len(gmms) < 2:
        raise ValueError("Need at least 2 GMMs")
    result = gmms[0]
    for g in gmms[1:]:
        result = _exact_product_pair(result, g)
    return result


def exact_product_ratio_density(numerator_gmms, denominator_gmms, x):
    """Evaluate  log p(x) = sum_i log p_num_i(x) - sum_j log p_den_j(x),
    shifted so max=0 across the input batch (i.e. up to an additive constant)."""
    x = np.atleast_2d(x)
    log_p = np.zeros(len(x))
    for g in numerator_gmms:
        log_p += g.log_prob(x)
    for g in denominator_gmms:
        log_p -= g.log_prob(x)
    log_p -= log_p.max()
    return log_p


def _rejection_bound_shared(gmm_b: IsotropicGMM, gmm_c: IsotropicGMM) -> float:
    """Upper bound M = sup_x B(x)/C(x) for two shared-component
    IsotropicGMMs (same means, same std, possibly different weights).
    Supremum calculated by max_k w_B[k]/w_C[k].
    """
    if not (gmm_b.K == gmm_c.K
            and np.allclose(gmm_b.means, gmm_c.means)
            and np.isclose(gmm_b.std, gmm_c.std)):
        raise ValueError("GMMs must share the same means and std")

    active_b = gmm_b.weights > 0
    active_c = gmm_c.weights > 0
    unsupported = active_b & ~active_c
    if np.any(unsupported):
        raise ValueError(
            f"Components {np.where(unsupported)[0].tolist()} have positive "
            "weight in B but zero weight in C."
        )

    ratios = np.where(active_b, gmm_b.weights / gmm_c.weights, 0.0)
    return float(np.max(ratios))


def _systematic_resample(log_weights: np.ndarray, rng) -> np.ndarray:
    """Systematic resample from unnormalized log weights. Returns (N,) indices."""
    N = len(log_weights)
    w = np.exp(log_weights - logsumexp(log_weights))
    cumw = np.cumsum(w)
    u = (np.arange(N) + rng.random()) / N
    return np.clip(np.searchsorted(cumw, u), 0, N - 1)


def sample_product_ratio_is(
    numerator_gmms, denominator_gmms, n,
    proposal=None, oversample_factor=1000, rng=None,
):
    """Importance-sampling sampler for p(x) propto prod_num(x) / prod_den(x).

    Uses the exact numerator product GMM as the proposal and reweights by
    1/prod_den(x). Then systematic-resamples to n unweighted samples. Up to
    log-prob normalization the only x-dependent factor in the importance
    weight is the denominator, so log_w = -sum_j log p_den_j(x).
    """
    if rng is None:
        rng = np.random.default_rng()
    if proposal is None:
        if len(numerator_gmms) >= 2:
            proposal = exact_gmm_product(*numerator_gmms)
        else:
            proposal = numerator_gmms[0]

    n_proposal = n * oversample_factor
    x = proposal.sample(n_proposal)

    log_w = np.zeros(n_proposal)
    for g in denominator_gmms:
        log_w -= g.log_prob(x)

    indices = _systematic_resample(log_w, rng)
    # Subsample if we have more than n
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False)
    return x[indices]


def sample_product_ratio_rejection(numerator_gmms, denominator_gmms, n, rng=None):
    """Rejection sampler for p(x) propto A(x) * B(x) / C(x).

    The first numerator is the proposal A; the remaining numerators are folded
    into a ratio factor B, and the denominators are folded into C. B and C
    must be shared-component IsotropicGMMs (same means and std) so that
    M = sup_x B(x)/C(x) = max_k w_B[k]/w_C[k] can be computed analytically.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(numerator_gmms) < 2:
        raise ValueError(
            "Rejection sampling needs >=2 numerator GMMs "
            "(first is proposal, rest form the ratio factor B)."
        )
    if len(denominator_gmms) < 1:
        raise ValueError("Need at least 1 denominator GMM.")

    proposal = numerator_gmms[0]
    if len(numerator_gmms) > 2:
        gmm_b = exact_gmm_product(*numerator_gmms[1:])
    else:
        gmm_b = numerator_gmms[1]

    if len(denominator_gmms) > 1:
        gmm_c = exact_gmm_product(*denominator_gmms)
    else:
        gmm_c = denominator_gmms[0]

    # 2x safety factor on the analytical bound.
    M = _rejection_bound_shared(gmm_b, gmm_c) * 2.0

    accepted = []
    total_proposed = 0
    while sum(len(a) for a in accepted) < n:
        remaining = n - sum(len(a) for a in accepted)
        batch_size = max(int(remaining * M * 2), 1024)
        x = proposal.sample(batch_size)
        total_proposed += batch_size

        log_accept = gmm_b.log_prob(x) - gmm_c.log_prob(x) - np.log(M)
        log_accept = np.minimum(log_accept, 0.0)

        log_u = np.log(rng.random(batch_size) + 1e-300)
        mask = log_u < log_accept
        accepted.append(x[mask])

    samples = np.concatenate(accepted, axis=0)[:n]
    info = {
        "acceptance_rate": n / total_proposed if total_proposed > 0 else 0.0,
        "M_bound": M,
        "total_proposed": total_proposed,
    }
    return samples, info


class ProductRatioDistribution:
    """Exact target distribution p(x) propto prod_i p_num_i(x) / prod_j p_den_j(x).

    Implements the Distribution interface (sample, log_prob). The result is
    not assumed to itself be a GMM.

    Args:
        numerator_gmms, denominator_gmms: lists of IsotropicGMM factors.
        sampler: "rejection" or "is". Rejection sampling requires the
            numerators after the first and the denominators to be shared-
            component IsotropicGMMs so the rejection bound is computable
            analytically, importance sampling has no such requirement.
    """

    def __init__(self, numerator_gmms, denominator_gmms, sampler: str = "rejection"):
        if sampler not in ("rejection", "is"):
            raise ValueError(f"sampler must be 'rejection' or 'is', got {sampler!r}")
        self.numerator_gmms = list(numerator_gmms)
        self.denominator_gmms = list(denominator_gmms)
        self.sampler = sampler

    def sample(self, n):
        if self.sampler == "rejection":
            samples, _ = sample_product_ratio_rejection(
                self.numerator_gmms, self.denominator_gmms, n,
            )
            return samples
        return sample_product_ratio_is(
            self.numerator_gmms, self.denominator_gmms, n,
        )

    def log_prob(self, x):
        return exact_product_ratio_density(
            self.numerator_gmms, self.denominator_gmms, x,
        )
