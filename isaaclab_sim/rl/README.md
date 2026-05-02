# RoboCup VisionRL RL Interface

This folder provides the reinforcement-learning bridge for the RoboCup IsaacLab scene.

Chosen algorithm path:

- PPO for the first single-agent yellow-side smoke test.
- MAPPO-style self-play for the final two-robot elimination strategy.

Why MAPPO/self-play:

- Two robots learn against changing opponents instead of following a fixed script.
- The actor uses only local robot observations, so it can be deployed through ROS2.
- The centralized critic can see both sides during training, which stabilizes sparse events such as shooting, armor removal, base knockdown, and contact tactics.
- The environment is vectorized so many independent matches can train in parallel before transferring policies into the full IsaacLab scene.

Fast single-agent smoke test:

```powershell
cd <repo-root>\isaaclab_sim\rl
<isaaclab-python> robocup_visionrl_gym_env.py
```

Train a PPO baseline:

```powershell
cd <repo-root>\isaaclab_sim\rl
<isaaclab-python> train_ppo_sb3.py --timesteps 200000
```

Train a parallel PPO baseline:

```powershell
cd <repo-root>\isaaclab_sim\rl
<isaaclab-python> train_ppo_parallel_sb3.py --num-envs 8 --timesteps 1000000
```

Run the two-agent self-play interface smoke test:

```powershell
cd <repo-root>\isaaclab_sim\rl
<isaaclab-python> robocup_visionrl_selfplay_env.py
```

Run the vectorized self-play rollout smoke test:

```powershell
cd <repo-root>\isaaclab_sim\rl
<isaaclab-python> robocup_visionrl_selfplay_vec.py
```

Train MAPPO-style self-play with parallel environments:

```powershell
cd <repo-root>\isaaclab_sim\rl
<isaaclab-python> train_mappo_selfplay_parallel_torch.py --num-envs 16 --rollout-steps 256 --timesteps 500000
```

Action space:

- `linear_velocity`: normalized forward/backward velocity in `[-1, 1]`
- `angular_velocity`: normalized yaw rate in `[-1, 1]`
- `fire_gate`: fire when greater than `0.25`

Rule logic included:

- Yellow and blue robots have collision footprints.
- Wall and obstacle collision is checked before accepting motion.
- Robots can collide with each other and with targets.
- Collision-knocked targets give the non-contact team the rule score.
- Robots only gain attack reward on opponent-owned targets; own-target and own-base hits are punished.
- Eight normal targets remove one opponent armor plate when knocked by a valid shot.
- Opponent base targets can be attacked directly when the shot line is clear.
- Collision or blocked motion reduces localization confidence.
- Spinning in place restores localization confidence and is rewarded only when confidence is poor.
- Episode time limit is 180 seconds.

Strategy layer:

- Attack target selection: nearest/high-confidence normal target vs. direct base rush.
- Defense selection: block the opponent only when leading or when the opponent threatens the own base lane.
- Collision reward is contextual; contact near the own base is punished because it can create accidental own-target risk.
- Obstacles are static by default. Pushing them should be added only after real-world repeatability tests.
- Keep Nav2, AprilTag detection, and shooter services as the real-robot interface instead of deploying a policy that depends on simulator-only state.
