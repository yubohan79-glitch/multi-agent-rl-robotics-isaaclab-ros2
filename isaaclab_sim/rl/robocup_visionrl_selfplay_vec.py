from __future__ import annotations

import numpy as np

from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv


class RoboCupVisionRLSelfPlayVector:
    """Simple vectorized self-play runner for MAPPO rollout collection.

    It keeps environments in-process so debugging is easy. A future MAPPO
    trainer can replace this with a multiprocessing collector when policies
    and replay buffers are finalized.
    """

    def __init__(self, num_envs: int = 16, seed: int = 7):
        self.envs = [RoboCupVisionRLSelfPlayEnv() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.seed = seed

    def reset(self):
        observations = []
        infos = []
        for index, env in enumerate(self.envs):
            obs, info = env.reset(seed=self.seed + index)
            observations.append(obs)
            infos.append(info)
        return observations, infos

    def reset_one(self, index: int, seed: int | None = None):
        return self.envs[index].reset(seed=self.seed + index if seed is None else seed)

    def step(self, actions: list[dict[str, np.ndarray]]):
        outputs = []
        for env, action in zip(self.envs, actions):
            outputs.append(env.step(action))
        observations, rewards, terminations, truncations, infos = zip(*outputs)
        return list(observations), list(rewards), list(terminations), list(truncations), list(infos)


if __name__ == "__main__":
    vec = RoboCupVisionRLSelfPlayVector(num_envs=8)
    observations, _ = vec.reset()
    for _ in range(8):
        actions = [
            {team: vec.envs[index].action_spaces[team].sample() for team in AGENTS}
            for index in range(vec.num_envs)
        ]
        observations, rewards, terminations, truncations, infos = vec.step(actions)
        mean_yellow = sum(item["yellow"] for item in rewards) / vec.num_envs
        mean_blue = sum(item["blue"] for item in rewards) / vec.num_envs
        print(f"mean_reward yellow={mean_yellow:.3f} blue={mean_blue:.3f}")
