from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
TANH_EPS = 1e-6


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(last_dim, hidden_dim), nn.SiLU()])
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class FlowActor(nn.Module):
    """Velocity-reparameterized flow actor for bounded tactical actions.

    This MVP keeps an analytically tractable Gaussian base distribution and
    applies a small deterministic velocity field before tanh squashing. The
    SAC update uses the base log-probability plus tanh correction as an
    approximation, which is enough to replace the old Gaussian actor while
    leaving room for exact CNF likelihood in a later research pass.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        flow_steps: int = 3,
        velocity_scale: float = 0.20,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.flow_steps = int(flow_steps)
        self.velocity_scale = float(velocity_scale)
        self.base = build_mlp(obs_dim, hidden_dim, action_dim * 2)
        self.velocity = build_mlp(obs_dim + action_dim + 1, hidden_dim, action_dim, depth=2)

    def _base_stats(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.base(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def _flow(self, obs: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        value = raw
        steps = max(self.flow_steps, 1)
        for step in range(steps):
            time_feature = torch.full((obs.shape[0], 1), (step + 0.5) / steps, dtype=obs.dtype, device=obs.device)
            velocity = torch.tanh(self.velocity(torch.cat([obs, value, time_feature], dim=-1)))
            value = value + self.velocity_scale * velocity / steps
        return value

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self._base_stats(obs)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        base_raw = mean + std * noise
        flowed_raw = self._flow(obs, base_raw)
        action = torch.tanh(flowed_raw)
        base_log_prob = -0.5 * (((base_raw - mean) / (std + 1e-8)).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        squash_correction = torch.log(1.0 - action.pow(2) + TANH_EPS)
        log_prob = (base_log_prob - squash_correction).sum(dim=-1)
        return action, log_prob, flowed_raw

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _log_std = self._base_stats(obs)
        return torch.tanh(self._flow(obs, mean))


class CentralizedTwinQ(nn.Module):
    def __init__(
        self,
        object_dim: int,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        input_dim = int(object_dim) + int(obs_dim) * num_agents + int(action_dim) * num_agents
        self.q1 = build_mlp(input_dim, hidden_dim, num_agents, depth=3)
        self.q2 = build_mlp(input_dim, hidden_dim, num_agents, depth=3)

    def forward(
        self,
        object_state: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_obs = obs.reshape(obs.shape[0], -1)
        flat_actions = actions.reshape(actions.shape[0], -1)
        x = torch.cat([object_state, flat_obs, flat_actions], dim=-1)
        return self.q1(x), self.q2(x)


class ObjectWorldModel(nn.Module):
    """Auxiliary object-centric dynamics model.

    The SAC critic is still trained from real transitions. This model learns
    one-step object dynamics, rewards and termination so future iterations can
    add TD-MPC2/Dreamer-style imagined rollouts without changing the data path.
    """

    def __init__(
        self,
        object_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.object_dim = int(object_dim)
        self.num_agents = int(num_agents)
        input_dim = int(object_dim) + int(action_dim) * num_agents
        self.trunk = build_mlp(input_dim, hidden_dim, hidden_dim, depth=3)
        self.delta_head = nn.Linear(hidden_dim, object_dim)
        self.reward_head = nn.Linear(hidden_dim, num_agents)
        self.done_head = nn.Linear(hidden_dim, num_agents)

    def forward(self, object_state: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([object_state, actions.reshape(actions.shape[0], -1)], dim=-1)
        hidden = self.trunk(x)
        delta = 0.12 * torch.tanh(self.delta_head(hidden))
        return {
            "next_object_state": object_state + delta,
            "rewards": self.reward_head(hidden),
            "done_logits": self.done_head(hidden),
        }

    def loss(
        self,
        object_state: torch.Tensor,
        actions: torch.Tensor,
        next_object_state: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = self.forward(object_state, actions)
        state_loss = F.smooth_l1_loss(pred["next_object_state"], next_object_state)
        reward_loss = F.mse_loss(pred["rewards"], rewards)
        done_loss = F.binary_cross_entropy_with_logits(pred["done_logits"], dones)
        total = state_loss + 0.40 * reward_loss + 0.20 * done_loss
        metrics = {
            "wm_state_loss": float(state_loss.detach().cpu()),
            "wm_reward_loss": float(reward_loss.detach().cpu()),
            "wm_done_loss": float(done_loss.detach().cpu()),
        }
        return total, metrics
