"""2D Gaussian FKC experiment.

Three zero-mean diagonal Gaussians in 2D: a1, a2, base. The analytical target
is the product/ratio a1 * a2 / base.

Two entry points:
  python run_gaussian_2d_experiment.py [--config PATH]
  python run_gaussian_2d_experiment.py [--config PATH] --sweep
"""

import os

import numpy as np
import torch

from gmm_lib import DiagonalGaussian, analytical_product_ratio_diagonal
from noise_schedule import VPSchedule
from score_model import (
    train_score_model,
    train_conditional_score_model,
    BoundConditionalModel,
    sample_eps_model,
)
from feynman_kac import feynman_kac_sample
from evaluation import compute_distribution_metrics
from sweep import sweep_training_and_particles
from config import Gaussian2DConfig, CONFIGS_DIR, load_gaussian_2d_config
from plotting import (
    plot_comparison_summary,
    plot_distributions_overview,
    print_metrics,
    save_fig,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_ROOT = os.path.join(SCRIPT_DIR, "figures")
D = 2
DEFAULT_CONFIG = str(CONFIGS_DIR / "gaussian_2d.yaml")


def _fmt_var(v: float) -> str:
    return f"{float(v):g}".replace(".", "p")


def _variances_id(va1, va2, vbase) -> str:
    def part(name, vs):
        return name + "-".join(_fmt_var(v) for v in vs)
    return f"{part('a1', va1)}_{part('a2', va2)}_{part('base', vbase)}"


def _diag_text(vs) -> str:
    vs = [float(v) for v in vs]
    if all(v == vs[0] for v in vs):
        if vs[0] == 1.0:
            return "I"
        return f"{vs[0]:g}I"
    return "diag(" + ", ".join(f"{v:g}" for v in vs) + ")"


def figures_dir_for(va1, va2, vbase, mode: str) -> str:
    vid = _variances_id(va1, va2, vbase)
    suffix = "" if mode == "separate" else f"_{mode}"
    return os.path.join(FIGURES_ROOT, f"gaussian_2d{suffix}_{vid}")


def build_distributions(variances_a1, variances_a2, variances_base):
    dist_a1 = DiagonalGaussian(mean=[0, 0], variances=list(variances_a1))
    dist_a2 = DiagonalGaussian(mean=[0, 0], variances=list(variances_a2))
    dist_base = DiagonalGaussian(mean=[0, 0], variances=list(variances_base))
    dist_gt = analytical_product_ratio_diagonal(
        dist_a1, dist_a2, denominators=[dist_base],
    )
    return dist_a1, dist_a2, dist_base, dist_gt


def fixed_pool_data_fn(dist: DiagonalGaussian, n_pool: int):
    """Draw n_pool samples from dist once, then return a data_fn that resamples
    minibatches with replacement from that fixed pool."""
    pool = torch.from_numpy(dist.sample(n_pool)).float()

    def fn(bs):
        idx = torch.randint(0, n_pool, (bs,))
        return pool[idx]

    return fn


def eps_model_to_score_fn(model, schedule):
    def score_fn(t, x):
        return -model(t, x) / schedule.sigma(t)
    return score_fn


def analytical_score_fn(dist: DiagonalGaussian, schedule):
    """Exact score of the noised diagonal Gaussian at reverse time t.

    At reverse time t, x_t ~ N(alpha(t)*mu, alpha(t)^2 * diag(var) + sigma(t)^2 * I).
    """
    mean_cpu = torch.from_numpy(dist.mean).float()
    var_cpu = torch.from_numpy(dist.variances).float()
    cache: dict = {}

    def score_fn(t, x):
        device = x.device
        if device not in cache:
            cache[device] = (mean_cpu.to(device), var_cpu.to(device))
        mean_d, var_d = cache[device]
        alpha = schedule.alpha(t)
        sigma = schedule.sigma(t)
        mu_t = alpha * mean_d
        noised_var = alpha ** 2 * var_d + sigma ** 2
        return -(x - mu_t) / noised_var

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


def train_separate_models(schedule, dist_a1, dist_a2, dist_base, device,
                          num_iterations, training_size):
    print(f"     Training separate models for a1, a2, base "
          f"(pool size = {training_size})")
    return [
        train_score_model(
            fixed_pool_data_fn(d, training_size), schedule, D, device,
            num_iterations=num_iterations,
        )
        for d in (dist_a1, dist_a2, dist_base)
    ]


def train_conditional_models(schedule, dist_a1, dist_a2, dist_base, device,
                             num_iterations, training_size):
    print(f"     Training single conditional model over a1, a2, base "
          f"(pool size = {training_size} per condition)")
    cond_model = train_conditional_score_model(
        data_fns=[
            fixed_pool_data_fn(d, training_size)
            for d in (dist_a1, dist_a2, dist_base)
        ],
        schedule=schedule, x_dim=D, device=device,
        num_iterations=num_iterations,
    )
    return [BoundConditionalModel(cond_model, idx) for idx in range(3)]


def _condition_labels(dist_a1, dist_a2, dist_base):
    return [
        f"$P_{{a_1}} = \\mathcal{{N}}(0, {_diag_text(dist_a1.variances)})$",
        f"$P_{{a_2}} = \\mathcal{{N}}(0, {_diag_text(dist_a2.variances)})$",
        f"$P_{{\\mathrm{{base}}}} = \\mathcal{{N}}(0, {_diag_text(dist_base.variances)})$",
    ]


def main(config: Gaussian2DConfig):
    ss = config.single_shot
    if ss.mode not in ("separate", "conditional"):
        raise ValueError(f"single_shot.mode must be 'separate' or 'conditional', got {ss.mode!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    dist_a1, dist_a2, dist_base, dist_gt = build_distributions(
        config.variances_a1, config.variances_a2, config.variances_base,
    )

    figures_dir = figures_dir_for(
        config.variances_a1, config.variances_a2, config.variances_base, ss.mode,
    )
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Device: {device}, seed: {ss.seed}, samples: {ss.n_samples}")
    print(f"a1:   N(0, {_diag_text(dist_a1.variances)})")
    print(f"a2:   N(0, {_diag_text(dist_a2.variances)})")
    print(f"base: N(0, {_diag_text(dist_base.variances)})")
    print(f"Target a1*a2/base: N(0, {_diag_text(dist_gt.variances)})")
    print(f"Mode:      {ss.mode}")
    print(f"Pool size: {ss.training_size} per condition")
    print(f"Output dir: {figures_dir}")

    torch.manual_seed(ss.seed)
    np.random.seed(ss.seed)

    trainer = train_separate_models if ss.mode == "separate" else train_conditional_models
    model_a1, model_a2, model_base = trainer(
        schedule, dist_a1, dist_a2, dist_base, device,
        num_iterations=ss.train_iters, training_size=ss.training_size,
    )

    print("\n     FKC with analytical scores")
    analytical_score_fns = [
        analytical_score_fn(d, schedule) for d in (dist_a1, dist_a2, dist_base)
    ]
    analytical_samples = run_fk(
        analytical_score_fns, schedule, device,
        n_output=ss.n_samples, n_particles=ss.k_particles, n_steps=ss.n_steps,
    )
    analytical_metrics = compute_distribution_metrics(
        analytical_samples, dist_gt, n_gt_samples=ss.n_samples, seed=ss.seed,
    )
    print_metrics("  ", analytical_metrics)

    print("\n    FKC with learned scores")
    learned_score_fns = [
        eps_model_to_score_fn(m, schedule) for m in (model_a1, model_a2, model_base)
    ]
    learned_samples = run_fk(
        learned_score_fns, schedule, device,
        n_output=ss.n_samples, n_particles=ss.k_particles, n_steps=ss.n_steps,
    )
    learned_metrics = compute_distribution_metrics(
        learned_samples, dist_gt, n_gt_samples=ss.n_samples, seed=ss.seed,
    )
    print_metrics("  ", learned_metrics)

    print("\n    Sampling from trained models for overview plot")
    dists = [dist_a1, dist_a2, dist_base]
    true_samples = [d.sample(ss.n_samples) for d in dists]
    learned_marginals = [
        sample_from_model(m, schedule, device, ss.n_samples, ss.n_steps)
        for m in (model_a1, model_a2, model_base)
    ]

    fig_overview = plot_distributions_overview(
        true_samples=true_samples,
        learned_samples=learned_marginals,
        true_labels=_condition_labels(dist_a1, dist_a2, dist_base),
        learned_labels=["Learned $a_1$", "Learned $a_2$", "Learned base"],
        colors=["tab:blue", "tab:orange", "tab:gray"],
        grid_range=ss.grid_range,
        suptitle="2D Gaussian Composition",
        metric_seed=ss.seed,
    )
    save_fig(fig_overview, os.path.join(figures_dir, f"distributions_overview{ss.fig_ext}"))

    fig_summary = plot_comparison_summary(
        gt_samples=dist_gt.sample(ss.n_samples),
        analytical_samples=analytical_samples,
        learned_samples=learned_samples,
        analytical_metrics=analytical_metrics,
        learned_metrics=learned_metrics,
        k_particles=ss.k_particles,
        grid_range=ss.grid_range,
        gt_title="Ground truth",
    )
    save_fig(fig_summary, os.path.join(figures_dir, f"comparison_summary{ss.fig_ext}"))

    print("\nMETRICS SUMMARY")
    print(f"  Analytical scores: SW2={analytical_metrics['sw2']:.4f}  "
          f"MMD^2={analytical_metrics['mmd2']:.5f}")
    print(f"  Learned scores:    SW2={learned_metrics['sw2']:.4f}  "
          f"MMD^2={learned_metrics['mmd2']:.5f}")


def run_gaussian_2d_sweep(config: Gaussian2DConfig):
    sw = config.sweep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = VPSchedule(beta_min=0.1, beta_max=20.0)
    dist_a1, dist_a2, dist_base, dist_gt = build_distributions(
        config.variances_a1, config.variances_a2, config.variances_base,
    )

    mode_label = "separate" if sw.separate_models else "conditional"
    figures_dir = figures_dir_for(
        config.variances_a1, config.variances_a2, config.variances_base, mode_label,
    )

    print(f"     2D Gaussian {mode_label}-model sweep")
    print(f"  a1:   N(0, {_diag_text(dist_a1.variances)})")
    print(f"  a2:   N(0, {_diag_text(dist_a2.variances)})")
    print(f"  base: N(0, {_diag_text(dist_base.variances)})")
    print(f"  Target a1*a2/base: N(0, {_diag_text(dist_gt.variances)})")
    print(f"  training_sizes: {sw.training_sizes}")
    print(f"  particle_counts: {sw.particle_counts}")
    print(f"  n_runs: {sw.n_runs}, train_iters: {sw.train_iters}")
    print(f"  output dir: {figures_dir}")

    def train_and_get_score_fns(ts):
        dists = (dist_a1, dist_a2, dist_base)
        if ts == "analytical":
            return [analytical_score_fn(d, schedule) for d in dists]
        ts_int = int(ts)
        if sw.separate_models:
            models = [
                train_score_model(
                    fixed_pool_data_fn(d, ts_int),
                    schedule, D, device,
                    num_iterations=sw.train_iters, verbose=False,
                )
                for d in dists
            ]
        else:
            data_fns = [fixed_pool_data_fn(d, ts_int) for d in dists]
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

    return sweep_training_and_particles(
        train_and_get_score_fns,
        betas=[1.0, 1.0, -1.0],
        schedule=schedule,
        gt_dist=dist_gt,
        device=device,
        training_sizes=list(sw.training_sizes),
        particle_counts=list(sw.particle_counts),
        n_output=sw.n_output,
        n_runs=sw.n_runs,
        out_path=out_path,
        out_tex_path=out_tex_path,
        tex_caption_prefix=(
            f"2D Gaussian FKC sweep ({mode_label} models, "
            f"a1=N(0,{_diag_text(dist_a1.variances)}), "
            f"a2=N(0,{_diag_text(dist_a2.variances)}), "
            f"base=N(0,{_diag_text(dist_base.variances)}), "
            f"target N(0,{_diag_text(dist_gt.variances)}))"
        ),
        grid_range=sw.grid_range,
        ndim=D,
        seed=sw.seed,
        n_steps=sw.n_steps,
        plot_color="red",
        ts_label=ts_label,
        pc_label=pc_label,
        title=(
            f"2D Gaussian FKC ({mode_label}): "
            f"a1=N(0,{_diag_text(dist_a1.variances)}), "
            f"a2=N(0,{_diag_text(dist_a2.variances)}), "
            f"base=N(0,{_diag_text(dist_base.variances)})  --  training samples x K"
        ),
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

    cfg = load_gaussian_2d_config(args.config)
    if args.sweep:
        run_gaussian_2d_sweep(cfg)
    else:
        main(cfg)
