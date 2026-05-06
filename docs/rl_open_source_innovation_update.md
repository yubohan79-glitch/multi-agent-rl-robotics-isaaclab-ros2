# Open-Source Innovation Update

This update reviewed practical ideas from robot-learning papers, Isaac Lab documentation and PPO open-source tooling, then integrated the two changes that fit this project without weakening the rule model.

## Sources Reviewed

- OpenAI dynamics/domain randomization for Sim2Real robot control: <https://openai.com/index/sim-to-real-transfer-of-robotic-control-with-dynamics-randomization/>
- Isaac Lab GitHub repository: <https://github.com/isaac-sim/IsaacLab>
- Isaac Lab event/randomization manager documentation and source docs: <https://isaac-sim.github.io/IsaacLab/main/index.html>
- Stable-Baselines3 Contrib GitHub repository: <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>
- Stable-Baselines3 Contrib Maskable PPO documentation: <https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html>
- Invalid action masking paper: <https://arxiv.org/abs/2006.14171>

## Selected Innovations

1. Reset-time Sim2Real domain randomization

   Each episode now samples bounded perturbations for drive scale, turn scale, push response, shot accuracy, drift loss and sensor noise. This trains the high-level policy against the kinds of mismatch that will appear between IsaacLab/rule simulation and the ROS2 robot.

2. Geometry-aware action shield

   The environment now suppresses unsafe contact/fire commands before they become impossible physical actions or hard rule violations. The shield does not award fake success; it records `action_shield_contact` and `action_shield_fire` in `info`, so unsafe policy tendencies remain measurable.

## Implementation Scope

- `isaaclab_sim/rl/robocup_visionrl_selfplay_env.py`
- `isaaclab_sim/rl/robocup_visionrl_selfplay_vec.py`
- `isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py`
- `isaaclab_sim/rl/configs/mappo_selfplay.yaml`
- `tests/test_rl_strategy_contract.py`

## Final Training And Evidence

Checkpoint:

`isaaclab_sim/output/rl/mappo_drshield_recessed_base_shared_gpu_seed419/policy.pt`

Data snapshot:

`docs/rl_data/drshield_recessed_base_shared/`

Evaluation:

| Episodes | Yellow win | Blue win | Draw/timeout | Base wins/episode | Own-target penalties |
|---:|---:|---:|---:|---:|---:|
| 128 | 50.00% | 48.44% | 1.56% | 0.9844 | 0.0 |

Strict audit:

| Episodes | Hard violations | Warnings | Own-target penalties | Base wins/episode |
|---:|---:|---:|---:|---:|
| 16 | 0 | 0 | 0.0 | 1.0 |

Three synchronized IsaacLab MP4s were regenerated from strict episode 1:

- `docs/media/final_training_replay_overview.mp4`
- `docs/media/final_training_replay_yellow_pov.mp4`
- `docs/media/final_training_replay_blue_pov.mp4`
