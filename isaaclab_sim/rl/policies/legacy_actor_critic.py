from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianTeamActorCritic(nn.Module):
    """Minimal loader for archived Gaussian actor-critic checkpoints.

    The current project no longer trains this PPO/MAPPO-style model. It stays
    here only so old baselines can be evaluated or exported without keeping the
    obsolete training script in the main tree.
    """

    def __init__(
        self,
        obs_dim: int,
        central_obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_mode: str = "shared",
    ):
        super().__init__()
        self.actor_mode = actor_mode
        if actor_mode == "shared":
            self.actor = build_mlp(obs_dim, hidden_dim, action_dim)
        elif actor_mode == "dual":
            self.yellow_actor = build_mlp(obs_dim, hidden_dim, action_dim)
            self.blue_actor = build_mlp(obs_dim, hidden_dim, action_dim)
        else:
            raise ValueError(f"unknown actor_mode: {actor_mode}")
        self.critic = build_mlp(central_obs_dim, hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.90))

    def mean_action(self, obs: torch.Tensor, team_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.actor_mode == "shared":
            return torch.tanh(self.actor(obs))
        if team_ids is None:
            team_ids = (obs[:, -1] < 0.0).long()
        team_ids = team_ids.reshape(-1).to(device=obs.device)
        yellow_mean = self.yellow_actor(obs)
        blue_mean = self.blue_actor(obs)
        selector = (team_ids == 0).reshape(-1, 1)
        return torch.tanh(torch.where(selector, yellow_mean, blue_mean))

    def distribution(self, obs: torch.Tensor, team_ids: torch.Tensor | None = None) -> Normal:
        mean = self.mean_action(obs, team_ids)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)
