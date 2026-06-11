"""YAML-backed configs for the GMM and 2D Gaussian experiments.

Each config has two top-level sections:
  - one experiment-specific section (`gmm` or `gaussian_2d`) holding the
    target-distribution parameters,
  - shared `single_shot:` and `sweep:` sections holding pipeline knobs.

See configs/gmm.yaml and configs/gaussian_2d.yaml for fully-populated
examples. Loaders validate the keys they consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"


@dataclass
class SingleShotConfig:
    seed: int
    n_samples: int
    k_particles: int
    n_steps: int
    train_iters: int
    training_size: int
    grid_range: float
    fig_ext: str
    mode: str  # "separate" or "conditional"


@dataclass
class SweepConfig:
    seed: int
    n_output: int
    n_runs: int
    n_steps: int
    train_iters: int
    grid_range: float
    fig_ext: str
    training_sizes: list[Any]
    particle_counts: list[Any]
    separate_models: bool = False


# 2D Gaussian (anisotropic diagonal Gaussian product)

@dataclass
class Gaussian2DConfig:
    variances_a1: list[float]
    variances_a2: list[float]
    variances_base: list[float]
    single_shot: SingleShotConfig
    sweep: SweepConfig
    raw: dict = field(default_factory=dict)


# GMM (isotropic Gaussian mixture product)

@dataclass
class GMMConfig:
    means: np.ndarray
    attr_a1: tuple[int, ...]
    attr_a2: tuple[int, ...]
    component_std: float
    sampler: str  # "rejection" or "is"
    single_shot: SingleShotConfig
    sweep: SweepConfig
    raw: dict = field(default_factory=dict)


# Loaders

_REQUIRED_SINGLE_SHOT = {
    "seed", "n_samples", "k_particles", "n_steps", "train_iters",
    "training_size", "grid_range", "fig_ext", "mode",
}
_REQUIRED_SWEEP = {
    "seed", "n_output", "n_runs", "n_steps", "train_iters",
    "grid_range", "fig_ext", "training_sizes", "particle_counts",
    "separate_models",
}


def _require_keys(d: dict, required: set[str], section: str) -> None:
    missing = required - set(d.keys())
    if missing:
        raise ValueError(f"Config section {section!r} is missing keys: {sorted(missing)}")


def _build_single_shot(d: dict) -> SingleShotConfig:
    _require_keys(d, _REQUIRED_SINGLE_SHOT, "single_shot")
    return SingleShotConfig(
        seed=int(d["seed"]),
        n_samples=int(d["n_samples"]),
        k_particles=int(d["k_particles"]),
        n_steps=int(d["n_steps"]),
        train_iters=int(d["train_iters"]),
        training_size=int(d["training_size"]),
        grid_range=float(d["grid_range"]),
        fig_ext=str(d["fig_ext"]),
        mode=str(d["mode"]),
    )


def _build_sweep(d: dict) -> SweepConfig:
    _require_keys(d, _REQUIRED_SWEEP, "sweep")
    return SweepConfig(
        seed=int(d["seed"]),
        n_output=int(d["n_output"]),
        n_runs=int(d["n_runs"]),
        n_steps=int(d["n_steps"]),
        train_iters=int(d["train_iters"]),
        grid_range=float(d["grid_range"]),
        fig_ext=str(d["fig_ext"]),
        training_sizes=list(d["training_sizes"]),
        particle_counts=list(d["particle_counts"]),
        separate_models=bool(d["separate_models"]),
    )


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_gaussian_2d_config(path: str | Path) -> Gaussian2DConfig:
    raw = _load_yaml(path)
    if "gaussian_2d" not in raw:
        raise ValueError(f"{path}: missing top-level 'gaussian_2d:' section")
    g2d = raw["gaussian_2d"]
    for key in ("variances_a1", "variances_a2", "variances_base"):
        if key not in g2d:
            raise ValueError(f"{path}: gaussian_2d.{key} is required")
    if "single_shot" not in raw or "sweep" not in raw:
        raise ValueError(f"{path}: both 'single_shot' and 'sweep' sections are required")
    return Gaussian2DConfig(
        variances_a1=[float(v) for v in g2d["variances_a1"]],
        variances_a2=[float(v) for v in g2d["variances_a2"]],
        variances_base=[float(v) for v in g2d["variances_base"]],
        single_shot=_build_single_shot(raw["single_shot"]),
        sweep=_build_sweep(raw["sweep"]),
        raw=raw,
    )


def load_gmm_config(path: str | Path) -> GMMConfig:
    raw = _load_yaml(path)
    if "gmm" not in raw:
        raise ValueError(f"{path}: missing top-level 'gmm:' section")
    gmm = raw["gmm"]
    for key in ("means", "attr_a1", "attr_a2", "component_std", "sampler"):
        if key not in gmm:
            raise ValueError(f"{path}: gmm.{key} is required")
    sampler = str(gmm["sampler"])
    if sampler not in ("rejection", "is"):
        raise ValueError(
            f"{path}: gmm.sampler must be 'rejection' or 'is', got {sampler!r}"
        )
    means = np.asarray(gmm["means"], dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError(
            f"{path}: gmm.means must be an (M, 2) array of component means, got shape {means.shape}"
        )
    if "single_shot" not in raw or "sweep" not in raw:
        raise ValueError(f"{path}: both 'single_shot' and 'sweep' sections are required")
    return GMMConfig(
        means=means,
        attr_a1=tuple(int(i) for i in gmm["attr_a1"]),
        attr_a2=tuple(int(i) for i in gmm["attr_a2"]),
        component_std=float(gmm["component_std"]),
        sampler=sampler,
        single_shot=_build_single_shot(raw["single_shot"]),
        sweep=_build_sweep(raw["sweep"]),
        raw=raw,
    )
