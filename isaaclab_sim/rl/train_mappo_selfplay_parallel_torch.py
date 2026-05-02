from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from robocup_visionrl_selfplay_env import AGENTS
from robocup_visionrl_selfplay_vec import RoboCupVisionRLSelfPlayVector


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class SharedActorCentralCritic(nn.Module):
    """MAPPO-style shared actor with a centralized critic.

    Each robot acts from its own observation. The critic receives the local
    observation concatenated with the opponent observation, which matches the
    centralized-training/decentralized-execution setup used for self-play.
    """

    def __init__(self, obs_dim: int, central_obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.actor = build_mlp(obs_dim, hidden_dim, action_dim)
        self.critic = build_mlp(central_obs_dim, hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.55))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, central_obs: torch.Tensor):
        dist = self.distribution(obs)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        value = self.critic(central_obs).squeeze(-1)
        return raw_action, raw_action.clamp(-1.0, 1.0), log_prob, value

    def evaluate(self, obs: torch.Tensor, central_obs: torch.Tensor, raw_action: torch.Tensor):
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(central_obs).squeeze(-1)
        return log_prob, entropy, value


@dataclass
class TrainConfig:
    timesteps: int
    num_envs: int
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    learning_rate: float
    seed: int
    hidden_dim: int


def flatten_observations(observations: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    local_rows: list[np.ndarray] = []
    central_rows: list[np.ndarray] = []
    for obs in observations:
        for team in AGENTS:
            opponent = "blue" if team == "yellow" else "yellow"
            local = np.asarray(obs[team], dtype=np.float32)
            local_rows.append(local)
            central_rows.append(np.concatenate([local, np.asarray(obs[opponent], dtype=np.float32)]))
    return np.stack(local_rows), np.stack(central_rows).astype(np.float32)


def actions_to_env(raw_actions: np.ndarray, num_envs: int) -> list[dict[str, np.ndarray]]:
    clipped = np.clip(raw_actions, -1.0, 1.0).astype(np.float32)
    env_actions: list[dict[str, np.ndarray]] = []
    cursor = 0
    for _ in range(num_envs):
        action_dict: dict[str, np.ndarray] = {}
        for team in AGENTS:
            action_dict[team] = clipped[cursor]
            cursor += 1
        env_actions.append(action_dict)
    return env_actions


def rewards_to_array(rewards: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([item[team] for item in rewards for team in AGENTS], dtype=np.float32)


def dones_to_array(terminations: list[dict[str, bool]], truncations: list[dict[str, bool]]) -> np.ndarray:
    return np.asarray(
        [bool(terminations[index][team] or truncations[index][team]) for index in range(len(terminations)) for team in AGENTS],
        dtype=np.float32,
    )


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[step]
        next_value = next_values if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def main():
    parser = argparse.ArgumentParser(description="Train MAPPO self-play policy with parallel RoboCup VisionRL envs.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=str, default="../output/rl/robocup_visionrl_mappo_selfplay")
    args = parser.parse_args()

    cfg = TrainConfig(
        timesteps=args.timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    output_dir = (Path(__file__).resolve().parent / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vec = RoboCupVisionRLSelfPlayVector(num_envs=cfg.num_envs, seed=cfg.seed)
    observations, _ = vec.reset()
    obs_dim = vec.envs[0].observation_spaces["yellow"].shape[0]
    action_dim = vec.envs[0].action_spaces["yellow"].shape[0]
    model = SharedActorCentralCritic(obs_dim, obs_dim * 2, action_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    global_step = 0
    update = 0
    reset_seed = cfg.seed + 10_000
    agents_per_step = cfg.num_envs * len(AGENTS)

    while global_step < cfg.timesteps:
        update += 1
        local_obs_buf: list[np.ndarray] = []
        central_obs_buf: list[np.ndarray] = []
        raw_action_buf: list[np.ndarray] = []
        log_prob_buf: list[np.ndarray] = []
        value_buf: list[np.ndarray] = []
        reward_buf: list[np.ndarray] = []
        done_buf: list[np.ndarray] = []

        for _ in range(cfg.rollout_steps):
            local_obs, central_obs = flatten_observations(observations)
            obs_t = torch.as_tensor(local_obs, dtype=torch.float32, device=device)
            central_t = torch.as_tensor(central_obs, dtype=torch.float32, device=device)
            raw_action, clipped_action, log_prob, value = model.act(obs_t, central_t)

            next_observations, rewards, terminations, truncations, _infos = vec.step(
                actions_to_env(clipped_action.cpu().numpy(), cfg.num_envs)
            )
            dones = dones_to_array(terminations, truncations)

            local_obs_buf.append(local_obs)
            central_obs_buf.append(central_obs)
            raw_action_buf.append(raw_action.cpu().numpy().astype(np.float32))
            log_prob_buf.append(log_prob.cpu().numpy().astype(np.float32))
            value_buf.append(value.cpu().numpy().astype(np.float32))
            reward_buf.append(rewards_to_array(rewards))
            done_buf.append(dones)

            for env_index in range(cfg.num_envs):
                if any(terminations[env_index].values()) or any(truncations[env_index].values()):
                    next_observations[env_index], _ = vec.reset_one(env_index, seed=reset_seed)
                    reset_seed += 1

            observations = next_observations
            global_step += agents_per_step

        final_local, final_central = flatten_observations(observations)
        with torch.no_grad():
            next_values = model.critic(torch.as_tensor(final_central, dtype=torch.float32, device=device)).squeeze(-1)

        rewards_np = np.stack(reward_buf)
        dones_np = np.stack(done_buf)
        values_np = np.stack(value_buf)
        advantages_np, returns_np = compute_gae(
            rewards_np,
            dones_np,
            values_np,
            next_values.cpu().numpy().astype(np.float32),
            cfg.gamma,
            cfg.gae_lambda,
        )

        b_obs = torch.as_tensor(np.concatenate(local_obs_buf), dtype=torch.float32, device=device)
        b_central = torch.as_tensor(np.concatenate(central_obs_buf), dtype=torch.float32, device=device)
        b_actions = torch.as_tensor(np.concatenate(raw_action_buf), dtype=torch.float32, device=device)
        b_old_log_prob = torch.as_tensor(np.concatenate(log_prob_buf), dtype=torch.float32, device=device)
        b_advantages = torch.as_tensor(advantages_np.reshape(-1), dtype=torch.float32, device=device)
        b_returns = torch.as_tensor(returns_np.reshape(-1), dtype=torch.float32, device=device)
        b_values = torch.as_tensor(values_np.reshape(-1), dtype=torch.float32, device=device)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1e-8)
        batch_size = b_obs.shape[0]
        indices = np.arange(batch_size)
        minibatch_size = min(cfg.minibatch_size, batch_size)

        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb = torch.as_tensor(indices[start : start + minibatch_size], dtype=torch.long, device=device)
                new_log_prob, entropy, new_value = model.evaluate(b_obs[mb], b_central[mb], b_actions[mb])
                log_ratio = new_log_prob - b_old_log_prob[mb]
                ratio = log_ratio.exp()
                policy_loss_1 = -b_advantages[mb] * ratio
                policy_loss_2 = -b_advantages[mb] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                value_clipped = b_values[mb] + (new_value - b_values[mb]).clamp(-cfg.clip_coef, cfg.clip_coef)
                value_loss = 0.5 * torch.max((new_value - b_returns[mb]).pow(2), (value_clipped - b_returns[mb]).pow(2)).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

        mean_reward = float(rewards_np.mean())
        done_rate = float(dones_np.mean())
        print(f"[MAPPO]: update={update} steps={global_step} mean_reward={mean_reward:.3f} done_rate={done_rate:.3f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "central_obs_dim": obs_dim * 2,
        "action_dim": action_dim,
        "agents": AGENTS,
    }
    output_path = output_dir / "policy.pt"
    torch.save(checkpoint, output_path)
    print(f"[INFO]: Saved MAPPO self-play policy to {output_path}")


if __name__ == "__main__":
    main()
