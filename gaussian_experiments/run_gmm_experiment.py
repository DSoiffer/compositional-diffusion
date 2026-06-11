"""Gaussian mixture experiment: composing isotropic-GMM conditions via FKC.

Three conditions (a1, a2, base) share the same component means and std but
have different component weights, analytical target is a1 * a2 / base.

Two entry points:
  python run_gmm_experiment.py [--config PATH]
  python run_gmm_experiment.py [--config PATH] --sweep
"""

import os

import numpy as np
import torch

from gmm_lib import IsotropicGMM, ProductRatioDistribution
from noise_schedule import VPSchedule
from score_model import (
    BoundConditionalModel,
    sample_eps_model,
    train_conditional_score_model,
    train_score_model,
)
from feynman_kac import feynman_kac_sample
from evaluation import compute_distribution_metrics
from sweep import sweep_training_and_particles
from config import GMMConfig, CONFIGS_DIR, load_gmm_config
from plotting import (
    plot_comparison_summary,
    plot_distributions_overview,
    print_metrics,
    save_fig,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_ROOT = os.path.join(SCRIPT_DIR, "figures")
D = 2
DEFAULT_CONFIG = str(CONFIGS_DIR / "gmm.yaml")



def _attr_id(attr_a1, attr_a2) -> str:
    a1 = "-".join(str(i) for i in attr_a1)
    a2 = "-".join(str(i) for i in attr_a2)
    return f"a1-{a1}_a2-{a2}"


def figures_dir_for(attr_a1, attr_a2, mode: str) -> str:
    suffix = "" if mode == "separate" else f"_{mode}"
    return os.path.join(FIGURES_ROOT, f"gmm{suffix}_{_attr_id(attr_a1, attr_a2)}")


def build_gmms(means, attr_a1, attr_a2, component_std, sampler):
    """Return (gmm_a1, gmm_a2, gmm_base, gmm_gt). All three input GMMs share
    the same means and std, differing only in component weights. gmm_gt is
    the exact a1*a2/base ratio target sampled by the requested method."""
    means = np.asarray(means, dtype=np.float64)
    w_a1 = np.zeros(len(means))
    w_a2 = np.zeros(len(means))
    w_a1[list(attr_a1)] = 1.0
    w_a2[list(attr_a2)] = 1.0
    gmm_a1 = IsotropicGMM(means, std=component_std, weights=w_a1)
    gmm_a2 = IsotropicGMM(means, std=component_std, weights=w_a2)
    gmm_base = IsotropicGMM(means, std=component_std)
    gmm_gt = ProductRatioDistribution(
        numerator_gmms=[gmm_a1, gmm_a2],
        denominator_gmms=[gmm_base],
        sampler=sampler,
    )
    return gmm_a1, gmm_a2, gmm_base, gmm_gt


def fixed_pool_data_fn(gmm: IsotropicGMM, n_pool: int):
    """Draw n_pool samples from gmm once, then return a data_fn that resamples
    minibatches with replacement from that fixed pool."""
    pool = torch.from_numpy(gmm.sample(n_pool)).float()

    def fn(bs):
        idx = torch.randint(0, n_pool, (bs,))
        return pool[idx]

    return fn


def eps_model_to_score_fn(model, schedule):
    def score_fn(t, x):
        return -model(t, x) / schedule.sigma(t)
    return score_fn


def analytical_gmm_score_fn(gmm: IsotropicGMM, schedule):
    """Closed-form score of a noised IsotropicGMM at reverse time t. Evaluates
    the responsibility softmax directly on the input tensor's device."""
    means_cpu = torch.from_numpy(gmm.means).float()
    log_weights_cpu = torch.from_numpy(gmm.log_weights).float()
    base_var = float(gmm.var)
    cache: dict = {}

    def score_fn(t, x):
        device = x.device
        if device not in cache:
            cache[device] = (means_cpu.to(device), log_weights_cpu.to(device))
        means, log_weights = cache[device]

        t_scalar = t.reshape(-1)[:1]
        alpha = schedule.alpha(t_scalar)
        sigma = schedule.sigma(t_scalar)
        var_t = alpha ** 2 * base_var + sigma ** 2
        means_t = alpha * means

        diff = x[:, None, :] - means_t[None, :, :]
        log_comp = log_weights - 0.5 * (diff ** 2).sum(-1) / var_t
        resp = torch.softmax(log_comp, dim=-1)
        return (resp @ means_t - x) / var_t

    return score_fn


def sample_from_model(model, schedule, device, n_output, n_steps):
    x_final = sample_eps_model(
        model, schedule, ndim=D,
        n_output=n_output, n_steps=n_steps,
        device=device, verbose=False,
    )
    return x_final.cpu().numpy()


def run_fk(score_fns, schedule, device, *, n_output, n_particles, n_steps):
    x_final = feynman_kac_sample(
        score_fns,
        betas=[1.0, 1.0, -1.0],
        schedule=schedule,
        ndim=D,
        n_output=n_output,
        n_particles=n_particles,
        n_steps=n_steps,
        device=device,
        verbose=False,
    )
    return x_final.cpu().numpy()


def train_separate_models(schedule, gmm_a1, gmm_a2, gmm_base, device,
                          num_iterations, training_size):
    print(f"     Training separate models for a1, a2, base "
          f"(pool size = {training_size})     ")
    return [
        train_score_model(
            fixed_pool_data_fn(g, training_size), schedule, D, device,
            num_iterations=num_iterations,
        )
        for g in (gmm_a1, gmm_a2, gmm_base)
    ]


def train_conditional_models(schedule, gmm_a1, gmm_a2, gmm_base, device,
                             num_iterations, training_size):
    print(f"     Training single conditional model over a1, a2, base "
          f"(pool size = {training_size} per condition)")
    cond_model = train_conditional_score_model(
        data_fns=[
            fixed_pool_data_fn(g, training_size)
            for g in (gmm_a1, gmm_a2, gmm_base)
        ],
        schedule=schedule, x_dim=D, device=device,
        num_iterations=num_iterations,
    )
    return [BoundConditionalModel(cond_model, idx) for idx in range(3)]


def _modes(gmm: IsotropicGMM) -> np.ndarray:
    """Return the active-mode positions of a GMM (where component weight > 0)."""
    return gmm.means[gmm.weights > 0]


def main(config: GMMConfig):
    ss = config.single_shot
    if ss.mode not in ("separate", "conditional"):
        raise ValueError(f"single_shot.mode must be 'separate' or 'conditional', got {ss.mode!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    gmm_a1, gmm_a2, gmm_base, gmm_gt = build_gmms(
        config.means, config.attr_a1, config.attr_a2,
        config.component_std, config.sampler,
    )

    figures_dir = figures_dir_for(config.attr_a1, config.attr_a2, ss.mode)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Device: {device}, seed: {ss.seed}, samples: {ss.n_samples}")
    print(f"a1 modes:   {list(config.attr_a1)}")
    print(f"a2 modes:   {list(config.attr_a2)}")
    print(f"base:       uniform over all {len(config.means)} components")
    print(f"sigma:      {config.component_std}")
    print(f"GT sampler: {config.sampler}")
    print(f"Mode:       {ss.mode}")
    print(f"Pool size:  {ss.training_size} per condition")
    print(f"Target: p_a1 * p_a2 / p_base (exact ratio, not a GMM)")
    print(f"Output dir: {figures_dir}")

    torch.manual_seed(ss.seed)
    np.random.seed(ss.seed)

    trainer = train_separate_models if ss.mode == "separate" else train_conditional_models
    model_a1, model_a2, model_base = trainer(
        schedule, gmm_a1, gmm_a2, gmm_base, device,
        num_iterations=ss.train_iters, training_size=ss.training_size,
    )

    print("\n     FKC with analytical scores")
    analytical_score_fns = [
        analytical_gmm_score_fn(g, schedule) for g in (gmm_a1, gmm_a2, gmm_base)
    ]
    analytical_samples = run_fk(
        analytical_score_fns, schedule, device,
        n_output=ss.n_samples, n_particles=ss.k_particles, n_steps=ss.n_steps,
    )
    analytical_metrics = compute_distribution_metrics(
        analytical_samples, gmm_gt, n_gt_samples=ss.n_samples, seed=ss.seed,
    )
    print_metrics("  ", analytical_metrics)

    print("\n     FKC with learned scores")
    learned_score_fns = [
        eps_model_to_score_fn(m, schedule) for m in (model_a1, model_a2, model_base)
    ]
    learned_samples = run_fk(
        learned_score_fns, schedule, device,
        n_output=ss.n_samples, n_particles=ss.k_particles, n_steps=ss.n_steps,
    )
    learned_metrics = compute_distribution_metrics(
        learned_samples, gmm_gt, n_gt_samples=ss.n_samples, seed=ss.seed,
    )
    print_metrics("  ", learned_metrics)

    print("\n     Sampling from trained models for overview plot")
    gmms = [gmm_a1, gmm_a2, gmm_base]
    true_samples = [g.sample(ss.n_samples) for g in gmms]
    learned_marginals = [
        sample_from_model(m, schedule, device, ss.n_samples, ss.n_steps)
        for m in (model_a1, model_a2, model_base)
    ]
    modes_per_panel = [_modes(g) for g in gmms]

    fig_overview = plot_distributions_overview(
        true_samples=true_samples,
        learned_samples=learned_marginals,
        true_labels=["$P_{a_1}$", "$P_{a_2}$", "$P_{\\mathrm{base}}$"],
        learned_labels=["Learned $a_1$", "Learned $a_2$", "Learned base"],
        colors=["tab:blue", "tab:orange", "tab:gray"],
        grid_range=ss.grid_range,
        suptitle="Gaussian Mixture Composition",
        modes_per_panel=modes_per_panel,
        metric_seed=ss.seed,
    )
    save_fig(fig_overview, os.path.join(figures_dir, f"distributions_overview{ss.fig_ext}"))

    fig_summary = plot_comparison_summary(
        gt_samples=gmm_gt.sample(ss.n_samples),
        analytical_samples=analytical_samples,
        learned_samples=learned_samples,
        analytical_metrics=analytical_metrics,
        learned_metrics=learned_metrics,
        k_particles=ss.k_particles,
        grid_range=ss.grid_range,
        gt_title=f"Ground truth ({config.sampler})",
        overlay_modes_arr=_modes(gmm_base),
    )
    save_fig(fig_summary, os.path.join(figures_dir, f"comparison_summary{ss.fig_ext}"))

    print("\nMETRICS SUMMARY")
    print(f"  Analytical scores: SW2={analytical_metrics['sw2']:.4f}  "
          f"MMD^2={analytical_metrics['mmd2']:.5f}")
    print(f"  Learned scores:    SW2={learned_metrics['sw2']:.4f}  "
          f"MMD^2={learned_metrics['mmd2']:.5f}")


def run_gmm_sweep(config: GMMConfig):
    sw = config.sweep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    gmm_a1, gmm_a2, gmm_base, gmm_gt = build_gmms(
        config.means, config.attr_a1, config.attr_a2,
        config.component_std, config.sampler,
    )

    mode_label = "separate" if sw.separate_models else "conditional"
    figures_dir = figures_dir_for(config.attr_a1, config.attr_a2, mode_label)

    print(f"     GMM {mode_label}-model sweep")
    print(f"  a1 modes: {list(config.attr_a1)}, a2 modes: {list(config.attr_a2)}, "
          f"base = uniform over all {len(config.means)} components, sigma={config.component_std}")
    print(f"  GT target: p_a1 * p_a2 / p_base, sampler={config.sampler}")
    print(f"  training_sizes:  {sw.training_sizes}")
    print(f"  particle_counts: {sw.particle_counts}")
    print(f"  n_runs: {sw.n_runs}, train_iters: {sw.train_iters}")
    print(f"  output dir: {figures_dir}")

    def train_and_get_score_fns(ts):
        gmms = (gmm_a1, gmm_a2, gmm_base)
        if ts == "analytical":
            return [analytical_gmm_score_fn(g, schedule) for g in gmms]
        ts_int = int(ts)
        if sw.separate_models:
            models = [
                train_score_model(
                    fixed_pool_data_fn(g, ts_int),
                    schedule, D, device,
                    num_iterations=sw.train_iters, verbose=False,
                )
                for g in gmms
            ]
        else:
            data_fns = [fixed_pool_data_fn(g, ts_int) for g in gmms]
            cond_model = train_conditional_score_model(
                data_fns=data_fns,
                schedule=schedule, x_dim=D, device=device,
                num_iterations=sw.train_iters, verbose=False,
            )
            models = [BoundConditionalModel(cond_model, idx) for idx in range(3)]
        return [eps_model_to_score_fn(m, schedule) for m in models]

    def ts_label(ts):
        return "analytical" if ts == "analytical" else f"N={ts}"

    def pc_label(pc):
        return f"K={pc}"

    os.makedirs(figures_dir, exist_ok=True)
    base = f"sweep_training_particles_{mode_label}_n{sw.n_runs}"
    out_path = os.path.join(figures_dir, f"{base}{sw.fig_ext}")
    out_tex_path = os.path.join(figures_dir, f"{base}.tex")

    a1_set = ",".join(str(i) for i in config.attr_a1)
    a2_set = ",".join(str(i) for i in config.attr_a2)
    caption_prefix = (
        f"GMM FKC sweep ({mode_label} models, "
        f"$a_1$ modes $= \\{{{a1_set}\\}}$, "
        f"$a_2$ modes $= \\{{{a2_set}\\}}$, "
        f"base $=$ uniform over all {len(config.means)} components, "
        f"$\\sigma_{{\\mathrm{{comp}}}} = {config.component_std}$, "
        f"GT sampler $=$ {config.sampler})"
    )

    return sweep_training_and_particles(
        train_and_get_score_fns,
        betas=[1.0, 1.0, -1.0],
        schedule=schedule,
        gt_dist=gmm_gt,
        device=device,
        training_sizes=list(sw.training_sizes),
        particle_counts=list(sw.particle_counts),
        n_output=sw.n_output,
        n_runs=sw.n_runs,
        out_path=out_path,
        out_tex_path=out_tex_path,
        tex_caption_prefix=caption_prefix,
        grid_range=sw.grid_range,
        ndim=D,
        seed=sw.seed,
        n_steps=sw.n_steps,
        plot_color="purple",
        ts_label=ts_label,
        pc_label=pc_label,
        title=f"GMM FKC ({mode_label}): training samples x K",
        overlay_means=gmm_base.means,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run the (training_size x particle_count) sweep instead of the main pipeline.",
    )
    args = parser.parse_args()

    cfg = load_gmm_config(args.config)
    if args.sweep:
        run_gmm_sweep(cfg)
    else:
        main(cfg)
