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

from expert_policy import compose_policy_action
from policies import CentralizedTwinQ, FlowActor, ObjectWorldModel
from replay_buffer import MultiAgentReplayBuffer
from robocup_visionrl_selfplay_env import AGENTS
from robocup_visionrl_selfplay_vec import RoboCupVisionRLSelfPlayVector
from world_model import OBJECT_STATE_DIM, extract_object_state


@dataclass
class TrainConfig:
    timesteps: int
    num_envs: int
    seed: int
    hidden_dim: int
    batch_size: int
    replay_size: int
    learning_starts: int
    gradient_steps: int
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    world_model_lr: float
    alpha_lr: float
    target_entropy: float
    max_grad_norm: float
    flow_steps: int
    flow_velocity_scale: float
    actor_mode: str
    policy_mode: str
    residual_scale: float
    domain_randomization: bool
    action_shield: bool
    world_model_coef: float


def load_defaults(path: Path | None) -> dict[str, object]:
    defaults: dict[str, object] = {
        "timesteps": 100_000,
        "num_envs": 16,
        "seed": 7,
        "hidden_dim": 256,
        "batch_size": 512,
        "replay_size": 200_000,
        "learning_starts": 2048,
        "gradient_steps": 1,
        "gamma": 0.995,
        "tau": 0.01,
        "actor_lr": 3.0e-4,
        "critic_lr": 3.0e-4,
        "world_model_lr": 3.0e-4,
        "alpha_lr": 3.0e-4,
        "target_entropy": -6.0,
        "max_grad_norm": 1.0,
        "flow_steps": 3,
        "flow_velocity_scale": 0.20,
        "actor_mode": "dual",
        "policy_mode": "residual_expert",
        "residual_scale": 0.04,
        "domain_randomization": True,
        "action_shield": True,
        "world_model_coef": 0.25,
        "device": "auto",
        "output": "../output/rl/world_model_sacflow_selfplay",
    }
    if path is None:
        return defaults
    resolved = path if path.is_absolute() else Path(__file__).resolve().parent / path
    if not resolved.exists():
        return defaults
    config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    for key, value in config.items():
        if key in defaults:
            defaults[key] = value
    return defaults


class MultiAgentFlowActors(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        actor_mode: str,
        flow_steps: int,
        velocity_scale: float,
    ):
        super().__init__()
        self.actor_mode = actor_mode
        if actor_mode == "shared":
            self.shared_actor = FlowActor(obs_dim, action_dim, hidden_dim, flow_steps=flow_steps, velocity_scale=velocity_scale)
        elif actor_mode == "dual":
            self.yellow_actor = FlowActor(obs_dim, action_dim, hidden_dim, flow_steps=flow_steps, velocity_scale=velocity_scale)
            self.blue_actor = FlowActor(obs_dim, action_dim, hidden_dim, flow_steps=flow_steps, velocity_scale=velocity_scale)
        else:
            raise ValueError(f"unknown actor_mode: {actor_mode}")

    def _actor(self, index: int) -> FlowActor:
        if self.actor_mode == "shared":
            return self.shared_actor
        return self.yellow_actor if index == 0 else self.blue_actor

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = []
        log_probs = []
        for index in range(obs.shape[1]):
            action, log_prob, _raw = self._actor(index).sample(obs[:, index, :])
            actions.append(action)
            log_probs.append(log_prob)
        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        actions = [self._actor(index).deterministic(obs[:, index, :]) for index in range(obs.shape[1])]
        return torch.stack(actions, dim=1)


def observations_to_array(observations: list[dict[str, np.ndarray]]) -> np.ndarray:
    return np.stack(
        [np.stack([np.asarray(obs[team], dtype=np.float32) for team in AGENTS]) for obs in observations]
    ).astype(np.float32)


def object_states_to_array(vec: RoboCupVisionRLSelfPlayVector) -> np.ndarray:
    return np.stack([extract_object_state(env) for env in vec.envs]).astype(np.float32)


def actions_to_env(
    vec: RoboCupVisionRLSelfPlayVector,
    raw_actions: np.ndarray,
    *,
    policy_mode: str,
    residual_scale: float,
) -> list[dict[str, np.ndarray]]:
    clipped = np.clip(raw_actions, -1.0, 1.0).astype(np.float32)
    action_dicts: list[dict[str, np.ndarray]] = []
    for env_index, env in enumerate(vec.envs):
        item = {}
        for team_index, team in enumerate(AGENTS):
            item[team] = compose_policy_action(
                env,
                team,
                clipped[env_index, team_index],
                policy_mode=policy_mode,
                residual_scale=residual_scale,
            )
        action_dicts.append(item)
    return action_dicts


def random_actions(num_envs: int, action_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(num_envs, len(AGENTS), action_dim)).astype(np.float32)


def rewards_to_array(rewards: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([[item[team] for team in AGENTS] for item in rewards], dtype=np.float32)


def dones_to_array(terminations: list[dict[str, bool]], truncations: list[dict[str, bool]]) -> np.ndarray:
    return np.asarray(
        [
            [bool(terminations[index][team] or truncations[index][team]) for team in AGENTS]
            for index in range(len(terminations))
        ],
        dtype=np.float32,
    )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau).add_(src_param.data, alpha=tau)


def update_step(
    *,
    batch,
    actors: MultiAgentFlowActors,
    critic: CentralizedTwinQ,
    target_critic: CentralizedTwinQ,
    world_model: ObjectWorldModel,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    world_model_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    alpha_optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> dict[str, float]:
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_actions, next_log_probs = actors.sample(batch.next_obs)
        target_q1, target_q2 = target_critic(batch.next_object_state, batch.next_obs, next_actions)
        target_q = torch.minimum(target_q1, target_q2) - alpha * next_log_probs
        backup = batch.rewards + cfg.gamma * (1.0 - batch.dones) * target_q

    current_q1, current_q2 = critic(batch.object_state, batch.obs, batch.actions)
    critic_loss = (current_q1 - backup).pow(2).mean() + (current_q2 - backup).pow(2).mean()
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
    critic_optimizer.step()

    new_actions, log_probs = actors.sample(batch.obs)
    q1_pi, q2_pi = critic(batch.object_state, batch.obs, new_actions)
    q_pi = torch.minimum(q1_pi, q2_pi)
    actor_loss = (alpha * log_probs - q_pi).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actors.parameters(), cfg.max_grad_norm)
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (log_probs.detach() + cfg.target_entropy)).mean()
    alpha_optimizer.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_optimizer.step()

    world_model_loss, world_metrics = world_model.loss(
        batch.object_state,
        batch.actions,
        batch.next_object_state,
        batch.rewards,
        batch.dones,
    )
    weighted_world_model_loss = cfg.world_model_coef * world_model_loss
    world_model_optimizer.zero_grad(set_to_none=True)
    weighted_world_model_loss.backward()
    nn.utils.clip_grad_norm_(world_model.parameters(), cfg.max_grad_norm)
    world_model_optimizer.step()

    return {
        "critic_loss": float(critic_loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "alpha_loss": float(alpha_loss.detach().cpu()),
        "alpha": float(log_alpha.exp().detach().cpu()),
        "q_mean": float(q_pi.detach().mean().cpu()),
        **world_metrics,
    }


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("configs/world_model_flow.yaml"))
    pre_args, _unknown = pre_parser.parse_known_args()
    defaults = load_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Train object-centric world-model SAC Flow self-play policy.",
        parents=[pre_parser],
    )
    for key, value in defaults.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", default=value)
        elif isinstance(value, int):
            parser.add_argument(flag, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(flag, type=float, default=value)
        else:
            parser.add_argument(flag, type=str, default=value)
    args = parser.parse_args()

    cfg = TrainConfig(
        timesteps=int(args.timesteps),
        num_envs=int(args.num_envs),
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        batch_size=int(args.batch_size),
        replay_size=int(args.replay_size),
        learning_starts=int(args.learning_starts),
        gradient_steps=int(args.gradient_steps),
        gamma=float(args.gamma),
        tau=float(args.tau),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        world_model_lr=float(args.world_model_lr),
        alpha_lr=float(args.alpha_lr),
        target_entropy=float(args.target_entropy),
        max_grad_norm=float(args.max_grad_norm),
        flow_steps=int(args.flow_steps),
        flow_velocity_scale=float(args.flow_velocity_scale),
        actor_mode=str(args.actor_mode),
        policy_mode=str(args.policy_mode),
        residual_scale=float(args.residual_scale),
        domain_randomization=bool(args.domain_randomization),
        action_shield=bool(args.action_shield),
        world_model_coef=float(args.world_model_coef),
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")

    output_dir = (Path(__file__).resolve().parent / str(args.output)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "training_curve.csv"
    summary_path = output_dir / "training_summary.json"

    vec = RoboCupVisionRLSelfPlayVector(
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        env_kwargs={
            "domain_randomization": cfg.domain_randomization,
            "action_shield": cfg.action_shield,
        },
    )
    observations, _infos = vec.reset()
    obs_array = observations_to_array(observations)
    object_array = object_states_to_array(vec)
    obs_dim = obs_array.shape[-1]
    action_dim = vec.envs[0].action_spaces["yellow"].shape[0]

    actors = MultiAgentFlowActors(
        obs_dim,
        action_dim,
        cfg.hidden_dim,
        actor_mode=cfg.actor_mode,
        flow_steps=cfg.flow_steps,
        velocity_scale=cfg.flow_velocity_scale,
    ).to(device)
    critic = CentralizedTwinQ(OBJECT_STATE_DIM, obs_dim, action_dim, len(AGENTS), cfg.hidden_dim).to(device)
    target_critic = CentralizedTwinQ(OBJECT_STATE_DIM, obs_dim, action_dim, len(AGENTS), cfg.hidden_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    world_model = ObjectWorldModel(OBJECT_STATE_DIM, action_dim, len(AGENTS), cfg.hidden_dim).to(device)
    actor_optimizer = torch.optim.Adam(actors.parameters(), lr=cfg.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model_lr)
    log_alpha = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)

    replay = MultiAgentReplayBuffer(
        cfg.replay_size,
        len(AGENTS),
        obs_dim,
        OBJECT_STATE_DIM,
        action_dim,
        seed=cfg.seed,
    )

    curve_file = curve_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        curve_file,
        fieldnames=[
            "env_step",
            "buffer_size",
            "mean_reward",
            "done_rate",
            "critic_loss",
            "actor_loss",
            "alpha",
            "q_mean",
            "wm_state_loss",
            "wm_reward_loss",
            "wm_done_loss",
            "steps_per_second",
        ],
    )
    writer.writeheader()
    started = time.perf_counter()
    update_metrics: dict[str, float] = {}
    reset_seed = cfg.seed + 100_000

    try:
        env_step = 0
        while env_step < cfg.timesteps:
            if len(replay) < cfg.learning_starts:
                raw_actions = random_actions(cfg.num_envs, action_dim, rng)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
                    raw_actions = actors.sample(obs_t)[0].detach().cpu().numpy().astype(np.float32)

            next_observations, rewards, terminations, truncations, infos = vec.step(
                actions_to_env(vec, raw_actions, policy_mode=cfg.policy_mode, residual_scale=cfg.residual_scale)
            )
            reward_array = rewards_to_array(rewards)
            done_array = dones_to_array(terminations, truncations)
            next_obs_array_before_reset = observations_to_array(next_observations)
            next_object_array_before_reset = object_states_to_array(vec)

            for index in range(cfg.num_envs):
                replay.add(
                    obs_array[index],
                    object_array[index],
                    raw_actions[index],
                    reward_array[index],
                    next_obs_array_before_reset[index],
                    next_object_array_before_reset[index],
                    done_array[index],
                )

            for index in range(cfg.num_envs):
                if bool(done_array[index].max() > 0.0):
                    next_observations[index], _ = vec.reset_one(index, seed=reset_seed)
                    reset_seed += 1

            observations = next_observations
            obs_array = observations_to_array(observations)
            object_array = object_states_to_array(vec)
            env_step += cfg.num_envs

            if len(replay) >= cfg.learning_starts:
                for _ in range(cfg.gradient_steps):
                    batch = replay.sample(cfg.batch_size, device)
                    update_metrics = update_step(
                        batch=batch,
                        actors=actors,
                        critic=critic,
                        target_critic=target_critic,
                        world_model=world_model,
                        actor_optimizer=actor_optimizer,
                        critic_optimizer=critic_optimizer,
                        world_model_optimizer=world_model_optimizer,
                        log_alpha=log_alpha,
                        alpha_optimizer=alpha_optimizer,
                        cfg=cfg,
                    )
                    soft_update(critic, target_critic, cfg.tau)

            if env_step % max(cfg.num_envs * 10, 1) == 0 or env_step >= cfg.timesteps:
                elapsed = max(time.perf_counter() - started, 1e-9)
                row = {
                    "env_step": env_step,
                    "buffer_size": len(replay),
                    "mean_reward": float(reward_array.mean()),
                    "done_rate": float(done_array.mean()),
                    "critic_loss": update_metrics.get("critic_loss", 0.0),
                    "actor_loss": update_metrics.get("actor_loss", 0.0),
                    "alpha": update_metrics.get("alpha", float(log_alpha.exp().detach().cpu())),
                    "q_mean": update_metrics.get("q_mean", 0.0),
                    "wm_state_loss": update_metrics.get("wm_state_loss", 0.0),
                    "wm_reward_loss": update_metrics.get("wm_reward_loss", 0.0),
                    "wm_done_loss": update_metrics.get("wm_done_loss", 0.0),
                    "steps_per_second": float(env_step / elapsed),
                }
                writer.writerow(row)
                curve_file.flush()
                print(
                    "[WM-SACFLOW]: "
                    f"steps={env_step} buffer={len(replay)} reward={row['mean_reward']:.3f} "
                    f"done={row['done_rate']:.3f} q={row['q_mean']:.3f} alpha={row['alpha']:.3f}",
                    flush=True,
                )
    finally:
        curve_file.close()

    checkpoint = {
        "algorithm": "object_centric_world_model_sac_flow_selfplay",
        "actor_state_dict": actors.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "target_critic_state_dict": target_critic.state_dict(),
        "world_model_state_dict": world_model.state_dict(),
        "log_alpha": float(log_alpha.detach().cpu()),
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "object_state_dim": OBJECT_STATE_DIM,
        "action_dim": action_dim,
        "agents": list(AGENTS),
        "actor_mode": cfg.actor_mode,
    }
    policy_path = output_dir / "policy.pt"
    torch.save(checkpoint, policy_path)
    summary = {
        "algorithm": checkpoint["algorithm"],
        "policy_path": str(policy_path),
        "curve_csv": str(curve_path),
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "object_state_dim": OBJECT_STATE_DIM,
        "action_dim": action_dim,
        "agents": list(AGENTS),
        "device": str(device),
        "torch_version": torch.__version__,
        "wall_time_s": round(time.perf_counter() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO]: Saved object-centric SAC Flow policy to {policy_path}", flush=True)


if __name__ == "__main__":
    main()
