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


def test_selfplay_observation_exposes_opponent_bearing():
    env = RoboCupVisionRLSelfPlayEnv()
    observations, _ = env.reset(seed=11)

    yellow_obs = observations["yellow"]
    blue_obs = observations["blue"]

    np.testing.assert_allclose(yellow_obs[4:6], -blue_obs[4:6], atol=1e-6)
    assert np.isfinite(yellow_obs[6:11]).all()
    assert yellow_obs[6] > 0.0
    assert -1.0 <= yellow_obs[7] <= 1.0
    assert -1.0 <= yellow_obs[8] <= 1.0
    assert yellow_obs[9] in (0.0, 1.0)
    assert 0.0 <= yellow_obs[10] <= 1.0


def test_single_agent_observation_exposes_blue_bearing():
    env = RoboCupVisionRLGymEnv()
    obs, _ = env.reset(seed=7)

    opponent_track = obs[8:13]
    assert np.isfinite(opponent_track).all()
    assert opponent_track[0] > 0.0
    assert -1.0 <= opponent_track[1] <= 1.0
    assert -1.0 <= opponent_track[2] <= 1.0
    assert opponent_track[3] in (0.0, 1.0)
    assert 0.0 <= opponent_track[4] <= 1.0
