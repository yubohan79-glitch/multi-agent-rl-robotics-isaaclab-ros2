# Capability Boundaries and Measured Evidence

本文档解决“能力边界与实测数据不明确”的问题。项目 README 中的 multi-agent、Sim2Real、IsaacLab、ROS2 等描述应按下面的证据边界理解，不能把未公开验证的能力当成已验证结果。

## 1. What Is Validated

当前公开验证的正式场景是 RoboCup-style 双车视觉对抗：

- agent count: 2
- sides: yellow vs blue
- robot model: differential-drive ground robot
- interaction: cooperative/competitive self-play in a two-team adversarial match
- objects: normal targets, recessed base targets, blue base blockers, red pushable boxes, walls, start partitions
- perception contract: target owner, target yaw, target visibility, line-of-sight, shooting range, hit dwell
- policy: object-centric world model + SAC Flow / PolicyFlow-style self-play with residual expert prior and action shield
- runtime interface: ROS2/Nav2/shooter/perception contract and IsaacLab replay

Validated public metrics are stored in:

```text
docs/rl_data/world_model_sacflow_final/training_summary.json
docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json
docs/rl_data/world_model_sacflow_final/strict_replay_summary.json
```

## 2. Measured Results

### 128-Episode Multi-Seed Evaluation

| Metric | Value |
| --- | ---: |
| Episodes | 128 |
| Yellow win rate | 49.22% |
| Blue win rate | 50.78% |
| Draw rate | 0.00% |
| Mean episode time | 30.8148 s |
| Mean yellow score | 40.8984 |
| Mean blue score | 41.7188 |
| Mean normal hits, yellow | 2.2734 |
| Mean normal hits, blue | 2.2500 |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Robot contacts/game | 0.00 |
| Repeat target order events | 0 |

Normal target hit distribution:

| Side | 1 hit | 2 hits | 3 hits | 4 hits |
| --- | ---: | ---: | ---: | ---: |
| Yellow | 0.78% | 71.88% | 26.56% | 0.78% |
| Blue | 2.34% | 70.31% | 27.34% | 0.00% |

Base success by number of normal targets:

| Side | 1 hit | 2 hits | 3 hits | 4 hits |
| --- | ---: | ---: | ---: | ---: |
| Yellow | 0 attempts | 36.22% | 45.71% | 100.00% from 1 attempt |
| Blue | 0 attempts | 42.40% | 34.29% | 0 attempts |

Pushable box evidence:

| Metric | Yellow | Blue |
| --- | ---: | ---: |
| Push events/episode | 2.0938 | 0.4219 |

| Box | Mean final displacement |
| --- | ---: |
| `box_ne` | 0.1193 m |
| `box_sw` | 0.0253 m |

### 8-Episode Strict Replay Audit

| Metric | Value |
| --- | ---: |
| Episodes | 8 |
| Yellow win rate | 37.50% |
| Blue win rate | 62.50% |
| Draw/timeout rate | 0.00% |
| Hard violations | 0 |
| Warnings | 0 |
| Normal hits/episode | 3.75 |
| Base wins/episode | 1.0000 |
| Own-target penalties/episode | 0.0 |
| Target contact events/episode | 0.0 |
| Robot contacts/episode | 0.0 |
| Recovery events/episode | 0.0 |

Strict replay media:

```text
docs/media/最终回放_顶视角.mp4
docs/media/最终回放_黄车第一视角.mp4
docs/media/最终回放_蓝车第一视角.mp4
```

## 3. Capability Matrix

| Capability | Status | Evidence |
| --- | --- | --- |
| Two-agent adversarial match | Validated | 128-episode evaluation and 8-episode strict replay |
| Yellow/blue asymmetric tactics | Validated | win-rate balance, target order, base rush and push events |
| Object-centric world model | Implemented and evaluated | training config, policy path, world-model coefficient, object state dimension |
| SAC Flow / PolicyFlow-style actor | Implemented and evaluated | `flow_steps`, `flow_velocity_scale`, residual expert policy mode |
| Rule-aware action shield | Implemented and evaluated | zero static/box penetrations in public eval |
| Pushable red box | Validated in sim/replay | push events and nonzero final displacement |
| Base blocker line-of-sight | Validated in rule/replay contract | strict replay hard violations = 0 |
| ROS2 runtime contract | Implemented | `crc_robocup_vision_ws/` packages and dry-run launch |
| IsaacLab replay | Implemented | final three-view MP4/GIF and strict replay summary |
| Large-scale multi-agent, more than 2 robots | Not publicly validated | no published benchmark or scaling curve in this repository |
| Distributed multi-node training | Not provided as a public feature | current released path is single-machine vectorized training |
| Real-robot success-rate benchmark | Not publicly provided | Sim2Real contract and calibration ladder exist, but no public quantified deployment dataset is included |

## 4. Multi-Agent Scope

In this repository, “multi-agent” means a two-agent adversarial self-play setup:

- yellow policy and blue policy act in the same environment.
- Both agents observe object-centric state.
- Both agents interact through target race, route blocking, base-rush timing, collision risk and pushable obstacles.
- The critic/world model can use centralized object state during training.
- Runtime behavior remains compatible with per-robot ROS2 interfaces.

The code structure is designed so more agents could be added by extending observation packing, action layout, collision checks and evaluation metrics. However, this repository does not publish evidence that the method scales to large swarms or many-team settings. Any claim about more than two simultaneously trained robots should be treated as future work unless accompanied by new evaluation data.

## 5. Cooperation and Competition Mechanism

The validated setting is competitive, not cooperative swarm control:

- Each side tries to hit opponent targets and opponent base.
- Own-side targets are illegal attack targets.
- Red boxes can create route opportunities.
- Base-rush timing creates strategic pressure.
- Blocking is possible but constrained by collision and penetration contracts.

There is no public benchmark here for cooperative formation, shared-goal navigation, auction/task allocation or communication learning among many robots.

## 6. Distributed Training Boundary

The released training path uses vectorized local environments and CUDA updates:

```bash
python3 isaaclab_sim/rl/train_world_model_sacflow_selfplay.py \
  --timesteps 200000 \
  --num-envs 32 \
  --batch-size 1024 \
  --gradient-steps 2 \
  --device cuda
```

Supported in the public workflow:

- single-machine training
- vectorized environment rollout
- GPU actor/critic/world-model updates
- JSON/CSV evaluation output

Not provided as a documented public feature:

- multi-node distributed rollout workers
- distributed replay buffer service
- parameter-server training
- multi-GPU synchronized training
- cluster scheduler integration

## 7. Sim2Real Boundary

The Sim2Real contribution in this repository is an interface and validation ladder:

- ROS2 topics/services/actions are separated from simulation internals.
- Nav2, AprilTag detection, EKF, shooter services and `/cmd_vel` remain the deployment contract.
- `docs/sim2real.md` defines calibration order, domain randomization, verification ladder and required logs.
- IsaacLab replay checks physical plausibility before real-world deployment.

What is not publicly claimed:

- no public real-robot win-rate benchmark is included.
- no public real-robot migration success percentage is included.
- no public long-horizon real arena dataset is included.
- no public statistical comparison between sim-only and real-robot matches is included.

Therefore, the correct wording is:

```text
The project provides a ROS2/IsaacLab Sim2Real contract and validation procedure, with simulation and replay evidence. Public quantified real-robot deployment metrics are not included in this repository.
```

## 8. How to Extend the Evidence

To claim a new capability, add a new evidence package:

1. define the new scenario and agent count.
2. document layout/rules in `config/`.
3. add tests for collision, line-of-sight, scoring and done reasons.
4. run at least 64 evaluation episodes, preferably 128+.
5. publish JSON/CSV metrics.
6. generate full replay media.
7. document failure cases and residual risks.
8. update this file with exact metrics and paths.

For real-robot Sim2Real claims, add:

- hardware configuration.
- calibration logs.
- rosbag2 recordings.
- number of real runs.
- success/failure definition.
- win rate or task success rate.
- collision/stuck/relocalization statistics.
- lighting and arena condition notes.
