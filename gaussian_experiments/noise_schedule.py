"""Variance-Preserving (VP) noise schedule for the diffusion models.

Convention: t in [0, 1] where t=0 is pure noise and t=1 is clean data.
Note: this is reverse from the paper.
"""

import torch


class VPSchedule:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        tau = 1.0 - t
        log_a = -0.5 * (self.beta_min * tau + 0.5 * (self.beta_max - self.beta_min) * tau**2)
        return torch.exp(log_a)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        return torch.sqrt(torch.clamp(1.0 - a**2, min=1e-8))

    def beta_at(self, t: torch.Tensor) -> torch.Tensor:
        tau = 1.0 - t
        return self.beta_min + tau * (self.beta_max - self.beta_min)

    def drift_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_at(t) / 2.0

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta_at(t))

    def noise_data(self, data: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.alpha(t)
        s = self.sigma(t)
        eps = torch.randn_like(data)
        return eps, a * data + s * eps
