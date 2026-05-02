from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from robocup_visionrl_gym_env import RoboCupVisionRLGymEnv


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the RoboCup VisionRL rule environment.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--output", type=str, default="../output/rl/robocup_visionrl_ppo")
    args = parser.parse_args()

    output_dir = (Path(__file__).resolve().parent / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(RoboCupVisionRLGymEnv())
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3.0e-4,
        n_steps=1024,
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
    print(f"[INFO]: Saved PPO policy to {output_dir / 'policy.zip'}")


if __name__ == "__main__":
    main()
