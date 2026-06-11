"""CLI driver for FKC sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler, UNet2DModel
from safetensors.torch import load_file

from fkc_sampling import (
    DiffusersVPSchedule,
    fkc_sample_disc,
    fkc_sample_disc_multimodel,
)


def load_checkpoint(checkpoint_dir: str | Path, device: torch.device, use_base: bool = False):
    """Load the UNet, scheduler, normalization stats, and class list from a
    training checkpoint produced by train.py."""
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    if use_base:
        weights = ckpt / "diffusion_pytorch_model.safetensors"
    else:
        ema_p = ckpt / "ema_model" / "diffusion_pytorch_model.safetensors"
        weights = ema_p if ema_p.exists() else ckpt / "diffusion_pytorch_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    print(f"Loading weights from {weights}")
    state_dict = load_file(str(weights))

    classes_path = ckpt / "classes.json"
    if not classes_path.is_file():
        raise FileNotFoundError(
            f"Missing {classes_path}. train.py must save the class list to the checkpoint."
        )
    with open(classes_path) as f:
        classes = json.load(f)["classes"]
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
        raise ValueError(f"{classes_path} must contain a JSON object with a list of class names.")

    model = UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=len(classes),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    sched_dir = ckpt / "scheduler"
    if not sched_dir.is_dir():
        raise FileNotFoundError(f"Missing scheduler directory: {sched_dir}")
    scheduler = DDPMScheduler.from_pretrained(sched_dir)
    if scheduler.config.prediction_type != "v_prediction":
        raise ValueError(
            f"Expected prediction_type='v_prediction', got {scheduler.config.prediction_type!r}."
        )
    if not getattr(scheduler.config, "rescale_betas_zero_snr", False):
        raise ValueError("Expected rescale_betas_zero_snr=True in the saved scheduler.")

    norm_path = ckpt / "normalize.json"
    with open(norm_path) as f:
        norm = json.load(f)
    data_mean, data_std = norm["mean"], norm["std"]
    print(f"  classes={classes}")
    print(f"  normalize: mean={data_mean}, std={data_std}")

    return model, scheduler, classes, data_mean, data_std


def denormalize(image: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=image.device, dtype=image.dtype).view(3, 1, 1)
    s = torch.tensor(std, device=image.device, dtype=image.dtype).view(3, 1, 1)
    return (image * s + m).clamp(0, 1)


def _scheduler_signature(scheduler):
    """Identifying tuple of a DDPMScheduler config. Used to verify that
    multimodel checkpoints share the same noise schedule; mismatch would
    invalidate the FKC weight."""
    cfg = scheduler.config
    return (
        cfg.prediction_type,
        bool(getattr(cfg, "rescale_betas_zero_snr", False)),
        int(cfg.num_train_timesteps),
        float(cfg.beta_start),
        float(cfg.beta_end),
        cfg.beta_schedule,
    )


def fkc_sample(
    checkpoint_dir: str | Path,
    classes_for_product: list[str],
    betas: list[float],
    *,
    n_output: int = 2,
    n_particles: int = 4,
    n_steps: int = 100,
    device: str | torch.device = "cuda",
    use_base: bool = False,
):
    """Load a single conditional checkpoint, run FKC sampling, return
    denormalized images."""
    device = torch.device(device)
    model, scheduler, all_classes, mean, std = load_checkpoint(
        checkpoint_dir, device, use_base=use_base,
    )
    schedule = DiffusersVPSchedule(scheduler, device)
    image_shape = (3, 256, 256)

    composition_str = " * ".join(
        f"p(x|{cls})^{b:+g}" for cls, b in zip(classes_for_product, betas)
    )
    print(f"FKC composition (single conditional): {composition_str}")
    print(f"  N={n_output}, K={n_particles}, steps={n_steps}, resample=every_step")

    x = fkc_sample_disc(
        model, scheduler, schedule,
        classes_for_product, betas, all_classes,
        image_shape=image_shape,
        n_output=n_output,
        n_particles=n_particles,
        n_steps=n_steps,
        device=device,
    )
    images = x.view(n_output, *image_shape)
    return denormalize(images, mean, std)


def fkc_sample_multimodel(
    checkpoint_dirs: list[str | Path],
    betas: list[float],
    *,
    component_labels: list[str] | None = None,
    n_output: int = 2,
    n_particles: int = 4,
    n_steps: int = 100,
    device: str | torch.device = "cuda",
    use_base: bool = False,
):
    """Load N separate checkpoints, run multi-model FKC sampling, return
    (denormalized images, component_labels). All checkpoints must share the
    same noise schedule (verified via _scheduler_signature). The first
    checkpoint's normalize.json is used for display; a warning is printed if
    the others differ.
    """
    if len(checkpoint_dirs) != len(betas):
        raise ValueError(
            f"checkpoint_dirs ({len(checkpoint_dirs)}) and betas ({len(betas)}) must match"
        )
    if len(checkpoint_dirs) < 2:
        raise ValueError("Multi-model FKC requires at least 2 checkpoints.")

    device = torch.device(device)

    models: list[UNet2DModel] = []
    schedulers: list[DDPMScheduler] = []
    means: list = []
    stds: list = []
    for path in checkpoint_dirs:
        m, sched_, _cls, mu, sd = load_checkpoint(path, device, use_base=use_base)
        models.append(m)
        schedulers.append(sched_)
        means.append(mu)
        stds.append(sd)

    sig0 = _scheduler_signature(schedulers[0])
    for path, s in zip(checkpoint_dirs[1:], schedulers[1:]):
        sig = _scheduler_signature(s)
        if sig != sig0:
            raise ValueError(
                f"Scheduler mismatch between {checkpoint_dirs[0]} and {path}:\n"
                f"  ref  = {sig0}\n"
                f"  this = {sig}\n"
                "Multi-model FKC requires all checkpoints share the same schedule."
            )

    if any(m != means[0] for m in means[1:]) or any(s != stds[0] for s in stds[1:]):
        print(
            "WARNING: normalize.json differs across multimodel checkpoints; "
            f"using stats from {checkpoint_dirs[0]} for display. "
            f"mean={means}, std={stds}"
        )

    scheduler, mean, std = schedulers[0], means[0], stds[0]
    schedule = DiffusersVPSchedule(scheduler, device)
    image_shape = (3, 256, 256)

    if component_labels is None:
        component_labels = [Path(p).name for p in checkpoint_dirs]
    if len(component_labels) != len(checkpoint_dirs):
        raise ValueError(
            f"component_labels ({len(component_labels)}) must match "
            f"checkpoint_dirs ({len(checkpoint_dirs)})"
        )

    composition_str = " * ".join(
        f"p_{lab}(x)^{b:+g}" for lab, b in zip(component_labels, betas)
    )
    print(f"FKC composition (multimodel): {composition_str}")
    print(f"  N={n_output}, K={n_particles}, steps={n_steps}, resample=every_step")

    x = fkc_sample_disc_multimodel(
        models, scheduler, schedule, betas,
        image_shape=image_shape,
        n_output=n_output,
        n_particles=n_particles,
        n_steps=n_steps,
        device=device,
    )
    images = x.view(n_output, *image_shape)
    return denormalize(images, mean, std), component_labels


def _save_grid(images: torch.Tensor, out_path: Path, title: str):
    """Save sampled images as a grid of images to a file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = images.shape[0]
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            ax.imshow(images[i].cpu().permute(1, 2, 0).float().numpy())
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Discrete-DDPM FKC sampling for class-conditional v-prediction UNets. "
        "Pass one --checkpoint + --classes for single-conditional mode, or N --checkpoint "
        "paths (no --classes) for multi-model mode.",
    )
    parser.add_argument(
        "--checkpoint", type=str, nargs="+", required=True,
        help="One checkpoint dir for single-conditional mode, or N>=2 checkpoint "
        "dirs (one per product component) for multi-model mode.",
    )
    parser.add_argument(
        "--classes", type=str, nargs="+", default=None,
        help="Single-conditional mode only: class names to compose. Must match "
        "the names saved in the checkpoint's classes.json.",
    )
    parser.add_argument(
        "--betas", type=float, nargs="+", required=True,
        help="composition weight per component (matches --classes or --checkpoint).",
    )
    parser.add_argument(
        "--component-labels", type=str, nargs="+", default=None,
        help="Multi-model mode only: display labels for plot title "
        "(default: checkpoint dir basenames).",
    )
    parser.add_argument("--n_output", type=int, default=12, help="N: number of output samples.")
    parser.add_argument("--n_particles", type=int, default=1, help="K: particles per swarm.")
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-base", action="store_true", help="Use base weights instead of EMA.")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path.")
    args = parser.parse_args()

    multimodel = len(args.checkpoint) > 1

    if multimodel:
        if args.classes is not None:
            parser.error(
                "--classes is for single-conditional mode only; do not pass it when "
                "you have multiple --checkpoint paths."
            )
        if len(args.checkpoint) != len(args.betas):
            parser.error(
                "--checkpoint and --betas must have the same length in multi-model mode."
            )

        images, labels = fkc_sample_multimodel(
            checkpoint_dirs=args.checkpoint,
            betas=args.betas,
            component_labels=args.component_labels,
            n_output=args.n_output,
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            device=args.device,
            use_base=args.use_base,
        )
        title = " * ".join(f"p_{lab}^{b:+g}" for lab, b in zip(labels, args.betas))
    else:
        if args.classes is None:
            parser.error(
                "--classes is required when --checkpoint has exactly 1 entry "
                "(single-conditional mode)."
            )
        if args.component_labels is not None:
            parser.error("--component-labels is for multi-model mode only.")
        if len(args.classes) != len(args.betas):
            parser.error("--classes and --betas must have the same length")

        images = fkc_sample(
            checkpoint_dir=args.checkpoint[0],
            classes_for_product=args.classes,
            betas=args.betas,
            n_output=args.n_output,
            n_particles=args.n_particles,
            n_steps=args.n_steps,
            device=args.device,
            use_base=args.use_base,
        )
        title = " * ".join(f"{cls}^{b:+g}" for cls, b in zip(args.classes, args.betas))

    _save_grid(
        images, Path(args.out),
        f"FKC: {title}    K={args.n_particles}, steps={args.n_steps}",
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
