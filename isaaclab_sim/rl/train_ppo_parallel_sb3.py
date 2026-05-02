from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from robocup_visionrl_gym_env import RoboCupVisionRLGymEnv


def make_env(seed: int, rank: int):
    def _init():
        env = RoboCupVisionRLGymEnv()
        env.reset(seed=seed + rank)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO with parallel RoboCup VisionRL rule environments.")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default="../output/rl/robocup_visionrl_ppo_parallel")
    args = parser.parse_args()

    output_dir = (Path(__file__).resolve().parent / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = VecMonitor(SubprocVecEnv([make_env(args.seed, rank) for rank in range(args.num_envs)]))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3.0e-4,
        n_steps=512,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=str(output_dir / "tb"),
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(str(output_dir / "policy"))
    env.close()
    print(f"[INFO]: Saved parallel PPO policy to {output_dir / 'policy.zip'}")


if __name__ == "__main__":
    main()
