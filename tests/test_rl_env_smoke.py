from __future__ import annotations

import numpy as np

from robocup_visionrl_gym_env import RoboCupVisionRLGymEnv
from robocup_visionrl_selfplay_env import AGENTS, RoboCupVisionRLSelfPlayEnv
from robocup_visionrl_selfplay_vec import RoboCupVisionRLSelfPlayVector


def test_single_agent_rule_env_step():
    env = RoboCupVisionRLGymEnv()
    obs, info = env.reset(seed=7)

    assert obs.shape == env.observation_space.shape
    assert info == {}

    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_selfplay_env_step_contract():
    env = RoboCupVisionRLSelfPlayEnv()
    observations, infos = env.reset(seed=11)

    assert set(observations) == set(AGENTS)
    assert set(infos) == set(AGENTS)

    actions = {team: np.zeros(3, dtype=np.float32) for team in AGENTS}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    assert set(observations) == set(AGENTS)
    assert set(rewards) == set(AGENTS)
    assert set(terminations) == set(AGENTS)
    assert set(truncations) == set(AGENTS)
    assert set(infos) == set(AGENTS)


def test_vectorized_selfplay_smoke():
    vec = RoboCupVisionRLSelfPlayVector(num_envs=4, seed=3)
    observations, infos = vec.reset()

    assert len(observations) == 4
    assert len(infos) == 4

    actions = [
        {team: vec.envs[index].action_spaces[team].sample() for team in AGENTS}
        for index in range(vec.num_envs)
    ]
    outputs = vec.step(actions)
    assert all(len(item) == 4 for item in outputs)
