from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.distributions import Normal

from expert_policy import batched_actions_to_env
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

    @torch.no_grad()
    def act(self, obs: torch.Tensor, central_obs: torch.Tensor, team_ids: torch.Tensor | None = None):
        dist = self.distribution(obs, team_ids)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        value = self.critic(central_obs).squeeze(-1)
        return raw_action, raw_action.clamp(-1.0, 1.0), log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        central_obs: torch.Tensor,
        raw_action: torch.Tensor,
        team_ids: torch.Tensor | None = None,
    ):
        dist = self.distribution(obs, team_ids)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(central_obs).squeeze(-1)
        return log_prob, entropy, value


def resume_state_for_actor_mode(
    checkpoint: dict[str, object],
    model: SharedActorCentralCritic,
    *,
    checkpoint_mode: str,
    target_mode: str,
) -> tuple[dict[str, torch.Tensor], str]:
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("resume checkpoint model_state_dict must be a state dict")
    if checkpoint_mode == target_mode:
        return state, f"loaded {checkpoint_mode} actor state"
    if checkpoint_mode == "shared" and target_mode == "dual":
        migrated = model.state_dict()
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if key.startswith("actor."):
                suffix = key[len("actor.") :]
                for actor_prefix in ("yellow_actor.", "blue_actor."):
                    target_key = actor_prefix + suffix
                    if target_key in migrated and migrated[target_key].shape == value.shape:
                        migrated[target_key] = value.clone()
                continue
            if key in migrated and migrated[key].shape == value.shape:
                migrated[key] = value.clone()
        return migrated, "migrated shared actor into yellow_actor and blue_actor"
    raise ValueError(
        f"resume checkpoint actor_mode={checkpoint_mode!r} cannot be loaded with --actor-mode={target_mode!r}"
    )


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
    policy_mode: str
    residual_scale: float
    residual_l2_coef: float
    actor_mode: str
    domain_randomization: bool
    action_shield: bool


def load_config_defaults(config_path: Path | None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "timesteps": 500_000,
        "num_envs": 16,
        "rollout_steps": 256,
        "update_epochs": 4,
        "minibatch_size": 1024,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_coef": 0.20,
        "ent_coef": 0.01,
        "vf_coef": 0.50,
        "max_grad_norm": 0.50,
        "learning_rate": 3.0e-4,
        "seed": 7,
        "hidden_dim": 128,
        "device": "auto",
        "output": "../output/rl/robocup_visionrl_mappo_selfplay",
        "policy_mode": "direct",
        "residual_scale": 0.28,
        "residual_l2_coef": 0.0,
        "actor_mode": "shared",
        "domain_randomization": False,
        "action_shield": True,
    }
    if config_path is None:
        return defaults
    resolved = config_path
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parent / resolved
    if not resolved.exists():
        return defaults
    config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    key_map = {
        "entropy_coef": "ent_coef",
        "value_coef": "vf_coef",
    }
    for key, value in config.items():
        mapped = key_map.get(key, key)
        if mapped in defaults:
            defaults[mapped] = value
    return defaults


def flatten_observations(observations: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_rows: list[np.ndarray] = []
    central_rows: list[np.ndarray] = []
    team_rows: list[int] = []
    for obs in observations:
        for team_index, team in enumerate(AGENTS):
            opponent = "blue" if team == "yellow" else "yellow"
            local = np.asarray(obs[team], dtype=np.float32)
            local_rows.append(local)
            central_rows.append(np.concatenate([local, np.asarray(obs[opponent], dtype=np.float32)]))
            team_rows.append(team_index)
    return np.stack(local_rows), np.stack(central_rows).astype(np.float32), np.asarray(team_rows, dtype=np.int64)


def actions_to_env(
    raw_actions: np.ndarray,
    num_envs: int,
    *,
    envs=None,
    policy_mode: str = "direct",
    residual_scale: float = 0.28,
) -> list[dict[str, np.ndarray]]:
    if policy_mode != "direct":
        if envs is None:
            raise ValueError("envs must be supplied when policy_mode is not direct")
        return batched_actions_to_env(
            list(envs),
            raw_actions,
            policy_mode=policy_mode,
            residual_scale=residual_scale,
        )
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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("configs/mappo_selfplay.yaml"))
    pre_args, _unknown = pre_parser.parse_known_args()
    defaults = load_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train MAPPO self-play policy with parallel RoboCup VisionRL envs.",
        parents=[pre_parser],
    )
    parser.add_argument("--timesteps", type=int, default=int(defaults["timesteps"]))
    parser.add_argument("--num-envs", type=int, default=int(defaults["num_envs"]))
    parser.add_argument("--rollout-steps", type=int, default=int(defaults["rollout_steps"]))
    parser.add_argument("--update-epochs", type=int, default=int(defaults["update_epochs"]))
    parser.add_argument("--minibatch-size", type=int, default=int(defaults["minibatch_size"]))
    parser.add_argument("--gamma", type=float, default=float(defaults["gamma"]))
    parser.add_argument("--gae-lambda", type=float, default=float(defaults["gae_lambda"]))
    parser.add_argument("--clip-coef", type=float, default=float(defaults["clip_coef"]))
    parser.add_argument("--ent-coef", type=float, default=float(defaults["ent_coef"]))
    parser.add_argument("--vf-coef", type=float, default=float(defaults["vf_coef"]))
    parser.add_argument("--max-grad-norm", type=float, default=float(defaults["max_grad_norm"]))
    parser.add_argument("--learning-rate", type=float, default=float(defaults["learning_rate"]))
    parser.add_argument("--seed", type=int, default=int(defaults["seed"]))
    parser.add_argument("--hidden-dim", type=int, default=int(defaults["hidden_dim"]))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=str(defaults["device"]))
    parser.add_argument("--output", type=str, default=str(defaults["output"]))
    parser.add_argument("--policy-mode", choices=("direct", "expert", "residual_expert"), default=str(defaults["policy_mode"]))
    parser.add_argument("--residual-scale", type=float, default=float(defaults["residual_scale"]))
    parser.add_argument("--residual-l2-coef", type=float, default=float(defaults["residual_l2_coef"]))
    parser.add_argument("--actor-mode", choices=("shared", "dual"), default=str(defaults["actor_mode"]))
    parser.add_argument("--domain-randomization", action="store_true", default=bool(defaults["domain_randomization"]))
    parser.add_argument("--no-action-shield", action="store_true", help="Disable the rule-aware safety action shield.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional MAPPO policy checkpoint to fine-tune.")
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
        policy_mode=args.policy_mode,
        residual_scale=args.residual_scale,
        residual_l2_coef=args.residual_l2_coef,
        actor_mode=args.actor_mode,
        domain_randomization=bool(args.domain_randomization),
        action_shield=not bool(args.no_action_shield),
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")

    output_dir = (Path(__file__).resolve().parent / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "training_curve.csv"
    jsonl_path = output_dir / "training_curve.jsonl"
    summary_path = output_dir / "training_summary.json"
    started = time.perf_counter()

    vec = RoboCupVisionRLSelfPlayVector(
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        env_kwargs={
            "domain_randomization": cfg.domain_randomization,
            "action_shield": cfg.action_shield,
        },
    )
    observations, _ = vec.reset()
    obs_dim = vec.envs[0].observation_spaces["yellow"].shape[0]
    action_dim = vec.envs[0].action_spaces["yellow"].shape[0]
    model = SharedActorCentralCritic(obs_dim, obs_dim * 2, action_dim, cfg.hidden_dim, cfg.actor_mode).to(device)
    resume_path = args.resume
    if resume_path is not None:
        resolved_resume = resume_path if resume_path.is_absolute() else (Path.cwd() / resume_path).resolve()
        checkpoint = torch.load(resolved_resume, map_location=device)
        if int(checkpoint["obs_dim"]) != obs_dim or int(checkpoint["action_dim"]) != action_dim:
            raise ValueError(
                "resume checkpoint dimensions do not match current environment: "
                f"checkpoint obs/action=({checkpoint['obs_dim']}, {checkpoint['action_dim']}), "
                f"current=({obs_dim}, {action_dim})"
            )
        checkpoint_mode = str(checkpoint.get("actor_mode", checkpoint.get("config", {}).get("actor_mode", "shared")))
        migrated_state, resume_note = resume_state_for_actor_mode(
            checkpoint,
            model,
            checkpoint_mode=checkpoint_mode,
            target_mode=cfg.actor_mode,
        )
        model.load_state_dict(migrated_state)
        print(f"[INFO]: Resumed MAPPO actor/critic from {resolved_resume} ({resume_note})", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    global_step = 0
    update = 0
    reset_seed = cfg.seed + 10_000
    agents_per_step = cfg.num_envs * len(AGENTS)
    curve_rows: list[dict[str, float | int | str]] = []

    curve_file = curve_path.open("w", newline="", encoding="utf-8")
    jsonl_file = jsonl_path.open("w", encoding="utf-8")
    writer = csv.DictWriter(
        curve_file,
        fieldnames=[
            "update",
            "global_step",
            "mean_reward",
            "done_rate",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "explained_variance",
            "steps_per_second",
            "device",
            "policy_mode",
            "residual_scale",
        ],
    )
    writer.writeheader()

    try:
        while global_step < cfg.timesteps:
            update += 1
            update_start = time.perf_counter()
            local_obs_buf: list[np.ndarray] = []
            central_obs_buf: list[np.ndarray] = []
            team_id_buf: list[np.ndarray] = []
            raw_action_buf: list[np.ndarray] = []
            log_prob_buf: list[np.ndarray] = []
            value_buf: list[np.ndarray] = []
            reward_buf: list[np.ndarray] = []
            done_buf: list[np.ndarray] = []

            for _ in range(cfg.rollout_steps):
                local_obs, central_obs, team_ids = flatten_observations(observations)
                obs_t = torch.as_tensor(local_obs, dtype=torch.float32, device=device)
                central_t = torch.as_tensor(central_obs, dtype=torch.float32, device=device)
                team_t = torch.as_tensor(team_ids, dtype=torch.long, device=device)
                raw_action, clipped_action, log_prob, value = model.act(obs_t, central_t, team_t)

                next_observations, rewards, terminations, truncations, _infos = vec.step(
                    actions_to_env(
                        clipped_action.cpu().numpy(),
                        cfg.num_envs,
                        envs=vec.envs,
                        policy_mode=cfg.policy_mode,
                        residual_scale=cfg.residual_scale,
                    )
                )
                dones = dones_to_array(terminations, truncations)

                local_obs_buf.append(local_obs)
                central_obs_buf.append(central_obs)
                team_id_buf.append(team_ids)
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

            final_local, final_central, _final_team_ids = flatten_observations(observations)
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
            b_team_ids = torch.as_tensor(np.concatenate(team_id_buf), dtype=torch.long, device=device)
            b_actions = torch.as_tensor(np.concatenate(raw_action_buf), dtype=torch.float32, device=device)
            b_old_log_prob = torch.as_tensor(np.concatenate(log_prob_buf), dtype=torch.float32, device=device)
            b_advantages = torch.as_tensor(advantages_np.reshape(-1), dtype=torch.float32, device=device)
            b_returns = torch.as_tensor(returns_np.reshape(-1), dtype=torch.float32, device=device)
            b_values = torch.as_tensor(values_np.reshape(-1), dtype=torch.float32, device=device)

            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1e-8)
            batch_size = b_obs.shape[0]
            indices = np.arange(batch_size)
            minibatch_size = min(cfg.minibatch_size, batch_size)
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            approx_kls: list[float] = []

            for _epoch in range(cfg.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, minibatch_size):
                    mb = torch.as_tensor(indices[start : start + minibatch_size], dtype=torch.long, device=device)
                    new_log_prob, entropy, new_value = model.evaluate(b_obs[mb], b_central[mb], b_actions[mb], b_team_ids[mb])
                    log_ratio = new_log_prob - b_old_log_prob[mb]
                    ratio = log_ratio.exp()
                    policy_loss_1 = -b_advantages[mb] * ratio
                    policy_loss_2 = -b_advantages[mb] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                    value_clipped = b_values[mb] + (new_value - b_values[mb]).clamp(-cfg.clip_coef, cfg.clip_coef)
                    value_loss = 0.5 * torch.max((new_value - b_returns[mb]).pow(2), (value_clipped - b_returns[mb]).pow(2)).mean()
                    entropy_loss = entropy.mean()
                    residual_l2 = torch.zeros((), dtype=torch.float32, device=device)
                    if cfg.residual_l2_coef > 0.0 and cfg.policy_mode == "residual_expert":
                        residual_mean = model.mean_action(b_obs[mb], b_team_ids[mb])
                        residual_l2 = residual_mean.pow(2).mean()
                    loss = (
                        policy_loss
                        + cfg.vf_coef * value_loss
                        - cfg.ent_coef * entropy_loss
                        + cfg.residual_l2_coef * residual_l2
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        model.log_std.clamp_(min=-2.20, max=-0.60)

                    policy_losses.append(float(policy_loss.detach().cpu()))
                    value_losses.append(float(value_loss.detach().cpu()))
                    entropies.append(float(entropy_loss.detach().cpu()))
                    with torch.no_grad():
                        approx_kls.append(float(((ratio - 1.0) - log_ratio).mean().detach().cpu()))

            mean_reward = float(rewards_np.mean())
            done_rate = float(dones_np.mean())
            returns_flat = returns_np.reshape(-1)
            values_flat = values_np.reshape(-1)
            return_var = float(np.var(returns_flat))
            explained_variance = 0.0 if return_var <= 1e-9 else 1.0 - float(np.var(returns_flat - values_flat) / return_var)
            elapsed = max(time.perf_counter() - update_start, 1e-9)
            row = {
                "update": update,
                "global_step": global_step,
                "mean_reward": mean_reward,
                "done_rate": done_rate,
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(approx_kls)),
                "explained_variance": explained_variance,
                "steps_per_second": float(cfg.rollout_steps * agents_per_step / elapsed),
                "device": str(device),
                "policy_mode": cfg.policy_mode,
                "residual_scale": cfg.residual_scale,
            }
            writer.writerow(row)
            curve_file.flush()
            jsonl_file.write(json.dumps(row) + "\n")
            jsonl_file.flush()
            curve_rows.append(row)
            print(
                "[MAPPO]: "
                f"update={update} steps={global_step} mean_reward={mean_reward:.3f} "
                f"done_rate={done_rate:.3f} loss_pi={row['policy_loss']:.4f} "
                f"loss_v={row['value_loss']:.4f} entropy={row['entropy']:.3f} device={device}",
                flush=True,
            )
    finally:
        curve_file.close()
        jsonl_file.close()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "central_obs_dim": obs_dim * 2,
        "action_dim": action_dim,
        "agents": AGENTS,
        "actor_mode": cfg.actor_mode,
        "resume": str(resume_path) if resume_path is not None else "",
    }
    output_path = output_dir / "policy.pt"
    torch.save(checkpoint, output_path)
    summary = {
        "algorithm": "MAPPO self-play",
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "central_obs_dim": obs_dim * 2,
        "action_dim": action_dim,
        "agents": AGENTS,
        "actor_mode": cfg.actor_mode,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "final_curve_row": curve_rows[-1] if curve_rows else None,
        "curve_csv": str(curve_path),
        "curve_jsonl": str(jsonl_path),
        "policy_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO]: Saved MAPPO self-play policy to {output_path}", flush=True)
    print(f"[INFO]: Wrote training curve to {curve_path}", flush=True)


if __name__ == "__main__":
    main()
