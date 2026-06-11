"""MLP noise-prediction models and training for diffusion."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


class MLPScoreModel(nn.Module):
    def __init__(self, x_dim: int, hidden_dim: int = 512, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 1 + x_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, x], dim=1))


class ConditionalMLPScoreModel(nn.Module):
    def __init__(self, x_dim: int, n_conditions: int, hidden_dim: int = 512, n_layers: int = 4):
        super().__init__()
        self.n_conditions = n_conditions
        layers: list[nn.Module] = []
        in_dim = 1 + x_dim + n_conditions
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        c_onehot = F.one_hot(c_idx, num_classes=self.n_conditions).float()
        return self.net(torch.cat([t, x, c_onehot], dim=1))


class BoundConditionalModel(nn.Module):
    """Wrap a ConditionalMLPScoreModel + a fixed condition index to expose
    the (t, x) -> eps interface of MLPScoreModel."""

    def __init__(self, cond_model: ConditionalMLPScoreModel, condition_idx: int):
        super().__init__()
        self.cond_model = cond_model
        self.condition_idx = int(condition_idx)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = torch.full((x.shape[0],), self.condition_idx, dtype=torch.long, device=x.device)
        return self.cond_model(t, x, c)


def train_score_model(
    data_fn,
    schedule,
    x_dim: int,
    device: torch.device,
    *,
    hidden_dim: int = 512,
    n_layers: int = 4,
    lr: float = 2e-4,
    batch_size: int = 512,
    num_iterations: int = 20_000,
    verbose: bool = True,
):
    model = MLPScoreModel(x_dim, hidden_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    iterator = trange(num_iterations) if verbose else range(num_iterations)
    for _ in iterator:
        data = data_fn(batch_size).to(device)
        t = torch.rand(batch_size, 1, device=device)
        eps, x_t = schedule.noise_data(data, t)

        eps_pred = model(t, x_t)
        loss = ((eps_pred - eps) ** 2).sum(1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def train_conditional_score_model(
    data_fns: list,
    schedule,
    x_dim: int,
    device: torch.device,
    *,
    hidden_dim: int = 512,
    n_layers: int = 4,
    lr: float = 2e-4,
    batch_size: int = 512,
    num_iterations: int = 20_000,
    verbose: bool = True,
):
    n_conditions = len(data_fns)
    model = ConditionalMLPScoreModel(x_dim, n_conditions, hidden_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    per_cond = batch_size // n_conditions
    cond_idx = torch.cat(
        [torch.full((per_cond,), i, dtype=torch.long) for i in range(n_conditions)]
    ).to(device)

    iterator = trange(num_iterations) if verbose else range(num_iterations)
    for _ in iterator:
        chunks = [fn(per_cond).to(device) for fn in data_fns]
        data = torch.cat(chunks, dim=0)
        t = torch.rand(data.shape[0], 1, device=device)
        eps, x_t = schedule.noise_data(data, t)

        eps_pred = model(t, x_t, cond_idx)
        loss = ((eps_pred - eps) ** 2).sum(1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


@torch.no_grad()
def sample_eps_model(
    model: nn.Module,
    schedule,
    ndim: int,
    n_output: int,
    *,
    n_steps: int = 500,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> torch.Tensor:
    """Reverse-time SDE sampling from a single eps-prediction model."""
    device = torch.device(device)
    dt = 1.0 / n_steps
    sqrt_dt = math.sqrt(dt)
    x = torch.randn(n_output, ndim, device=device)
    t_buf = torch.empty(n_output, 1, device=device)

    iterator = trange(n_steps) if verbose else range(n_steps)
    for step in iterator:
        t_val = max(min(step * dt, 1.0 - 1e-5), 1e-5)
        t_buf.fill_(t_val)
        eps = model(t_buf, x)
        sigma_t = schedule.sigma(t_buf)
        score = -eps / sigma_t
        a_t = schedule.drift_coeff(t_buf)
        sigma_sde = schedule.diffusion(t_buf)
        drift = a_t * x + sigma_sde**2 * score
        x = x + drift * dt + sigma_sde * sqrt_dt * torch.randn_like(x)

    return x
