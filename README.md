# Code for Catastrophic Compositional Generation: Why Vanilla Diffusion Models Fail to Extrapolate

Experimental code for reproducing our results is provided in `gaussian_experiments` for the Gaussian experiments, and `room_experiments` for the room experiments. Additional instructions are in each of those directories.

## Setup
Before running, ensure you have installed all dependencies with `uv sync` from the root directory. `uv` can be installed [here](https://docs.astral.sh/uv/getting-started/installation/).

Depending on your GPU's CUDA compatibility, you may need to alter all instances of `cu130` in pyproject.toml to an earlier version (e.g., `cu129` for CUDA 12.9), and rerun `uv sync`. Gaussian experiments can also be run on the CPU, but at reduced speed.