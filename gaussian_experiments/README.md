# GMM and 2D Gaussian Experiments

Two experiments that compose simple distributions via Feynman-Kac correctors
(FKC), corresponding to the 2D Gaussian and Gaussian mixture experiments in the paper.

- The **Gaussian mixture (GMM)** experiment composes three conditions
  (`a1`, `a2`, `base`) built from a shared 2-D isotropic GMM, with target `a1 * a2 / base`. (The target is not itself a GMM in general, so ground-truth samples come from either rejection sampling or importance sampling on the exact ratio density.)
- The **2D Gaussian** experiment composes three diagonal Gaussians (`a1`,
  `a2`, `base`) with different covariance structures, with target `a1 * a2 / base`.(The analytical target is a Gaussian, so ground-truth samples come from closed-form sampling.)


## Running
Before running, ensure you have installed all dependencies with `uv sync` from the root directory. `uv` can be installed [here](https://docs.astral.sh/uv/getting-started/installation/).

Both experiments can be run in one of two modes. Single-shot mode trains and runs a model a single time, producing plots using samples from its learned distributions compared against the ground truth. Sweep mode performs a sweep over different training sizes and number of FKC particles, reproducing the plots in the paper. The `--sweep` argument enables sweep mode. All results are stored under the `figures` directory (will be created if it does not exist).

To run the experiments, `cd` into this directory, and run either

`python run_gaussian_2d_experiment.py --config configs/gaussian_2d.yaml`

or
 
`python run_gmm_experiment.py --config configs/gmm.yaml`


### Configs

Each experiment reads its parameters from a YAML config:

- `configs/gaussian_2d.yaml`: diagonal-covariance variances for `a1`,
  `a2`, `base`, plus additional hyperparameters (described in the config file).
- `configs/gmm.yaml`: component means, condition weights, component std, plus additional hyperparameters. This corresponds to the in-distribution case. You can use `configs/gmm2.yaml` for the out-of-distribution case.

To override either config, copy the YAML, edit, and pass it as `--config`.

Within the config, the `single_shot:` block holds parameters for the per-experiment training + plotting pipeline, the recommended use is for producing plots. To alter settings for the sweep over different training sizes and numbers of particles, change parameters under the `sweep:` block.

Further config details are given in the config files.

Note that the results presented in the paper use a value of 15.0 for a `g_clip` parameter that we implement in FKC sampling, which clips the magnitude of the updates to the incremental `g` correction weight term. In practice, this tends to lower the score estimation error in OOD settings for the learned models by cutting off erroneously high weights. However, this also prevents the analytical score sampling's error from reducing as quickly as it should (using pure vanilla FKC) at high values of K. To disable this feature, set the default value to `None` in `feynman_kac.py`'s FKC sampling.

