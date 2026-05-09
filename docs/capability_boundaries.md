# Capability Boundaries and Measured Evidence

This file defines exactly what this repository validates and what it does not yet claim. It is intended to prevent over-claiming around multi-agent scale, IsaacLab physics, distributed training and Sim2Real deployment.

## 1. Validated Main Scenario

The primary validated scenario is a RoboCup-style two-robot yellow-vs-blue adversarial match. This 1v1 line has progressed beyond simulation: after the IsaacLab/rule-environment evidence package, real-robot 1v1 experiments were also performed. The repository still separates that fact from a fully published statistical hardware benchmark, because detailed rosbag logs, run counts and success-rate tables are not included here yet.

Validated elements:

- two differential-drive robots;
- object-centric world-model + SAC Flow / PolicyFlow-style self-play;
- opponent-only target shooting;
- legal target yaw and line-of-sight checks;
- 0.80 s laser dwell gate;
- normal-target and recessed-base shooting range gates;
- pushable red boxes;
- base blockers that must be removed before base hit;
- collision, penetration, stuck and replay audits;
- IsaacLab three-view MP4/GIF replay.
- 1v1 real-robot experiment coverage through the ROS2 deployment stack.

Primary evidence files:

```text
docs/rl_data/world_model_sacflow_final/training_summary.json
docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json
docs/rl_data/world_model_sacflow_final/strict_replay_summary.json
docs/media/ (final three-view MP4/GIF files with Chinese filenames)
```

## 2. Two-Robot Measured Results

128-episode multi-seed evaluation:

| Metric | Value |
| --- | ---: |
| Episodes | 128 |
| Yellow win rate | 49.22% |
| Blue win rate | 50.78% |
| Draw rate | 0.00% |
| Mean episode time | 30.8148 s |
| Mean yellow score | 40.8984 |
| Mean blue score | 41.7188 |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Robot contacts/game | 0.00 |
| Repeat target order events | 0 |

8-episode strict replay audit:

| Metric | Value |
| --- | ---: |
| Episodes | 8 |
| Yellow win rate | 37.50% |
| Blue win rate | 62.50% |
| Draw/timeout rate | 0.00% |
| Hard violations | 0 |
| Warnings | 0 |
| Base wins/episode | 1.0000 |
| Own-target penalties/episode | 0.0 |
| Robot contacts/episode | 0.0 |
| Recovery events/episode | 0.0 |

## 3. Large-Scale 50v50 Evidence

The repository now also includes a large-scale 50-vs-50 benchmark. This benchmark is designed for scalable strategy research and tactical visualization.

Validated in the current 50v50 evidence package:

- two teams with 50 vehicles each;
- three control zones;
- base shield opening via zone control;
- legal base damage only after shield opening;
- line-of-sight fire and fire cooldowns;
- vehicle elimination;
- score/winner closure;
- 256-game evaluation;
- IsaacLab tactical replay with 100 vehicle-shaped actors.

Current 256-game metrics:

| Metric | Value |
| --- | ---: |
| Episodes | 256 |
| Yellow win rate | 36.72% |
| Blue win rate | 42.19% |
| Draw rate | 21.09% |
| Mean yellow base damage | 44.90 |
| Mean blue base damage | 44.89 |
| Mean yellow base open rate | 18.37% |
| Mean blue base open rate | 18.39% |
| Mean robot contacts | 0.00 |
| P95 robot contacts | 0.00 |
| Mean obstacle contacts | 0.00 |

Large-scale evidence files:

```text
docs/large_scale_50v50_plan.md
docs/large_scale_50v50_curriculum_plan.md
docs/large_scale_50v50_report.md
docs/rl_data/large_scale_50v50/
docs/media/large_scale_50v50_isaaclab_replay.mp4
docs/media/large_scale_50v50_replay.mp4
docs/figures/large_scale_50v50/
```

Boundary for the 50v50 result:

- It is validated as rule-level large-scale training and evaluation.
- It is validated as an IsaacLab visual/tactical replay of the accepted trace.
- It is still a simulation-stage large-scale result.
- It is not yet validated as full rigid-body IsaacLab reinforcement learning with all 100 vehicles physically simulated during training.
- It is not real-hardware 100-robot validation.

## 4. Capability Matrix

| Capability | Status | Evidence |
| --- | --- | --- |
| Two-agent adversarial match | Validated | 128-episode eval and 8-episode strict replay |
| Object-centric world model | Implemented and evaluated | world-model/SAC Flow training artifacts |
| SAC Flow / PolicyFlow-style actor | Implemented and evaluated | training config, checkpoint and eval summaries |
| Rule-aware action shield | Validated in main scenario | zero static/box penetrations |
| Pushable red boxes | Validated in main scenario | push events and changing box poses |
| Base blocker line-of-sight | Validated in main scenario | strict replay hard violations = 0 |
| ROS2 runtime contract | Implemented and used for 1v1 real-robot experiments | `crc_robocup_vision_ws/` packages |
| IsaacLab two-robot replay | Validated | final three-view MP4/GIF |
| 1v1 real-robot experiment | Performed | public docs state coverage; full statistical hardware benchmark is not yet packaged |
| Large-scale 50v50 rule benchmark | Implemented and evaluated | 256-game eval and 50v50 artifacts |
| Large-scale IsaacLab replay | Implemented for simulation-stage 50v50 | `large_scale_50v50_isaaclab_replay.mp4` and generated figures |
| Full 100-robot rigid-body IsaacLab RL | Not yet validated | future physics-scaling milestone |
| Distributed multi-node training | Not provided | no released worker/replay-server stack |
| 50v50 real-robot deployment | Not validated | current 50v50 result remains simulation-stage only |
| Public quantified real-robot success-rate table | Not yet packaged | no public hardware run statistics table or rosbag bundle |

## 5. Multi-Agent Scope

The repository uses "multi-agent" in two explicit scopes:

1. Primary scope: a validated two-robot adversarial RoboCup-style match with subsequent 1v1 real-robot experiments.
2. Large-scale extension: a simulation-stage 50-vs-50 rule-level benchmark with IsaacLab replay evidence.

It does not yet claim:

- cooperative formation learning;
- learned inter-agent communication;
- 100-robot hardware deployment;
- 50v50 real-robot deployment;
- multi-node distributed self-play;
- full-physics 100-robot RL in IsaacLab.

## 6. Distributed Training Boundary

Supported in the public workflow:

- single-machine training;
- vectorized local environment rollout;
- CUDA actor/critic/world-model updates for the main two-robot policy;
- population-based local policy search for the 50v50 benchmark;
- JSON/CSV evaluation output.

Not provided as public features:

- multi-node rollout workers;
- distributed replay buffer service;
- parameter-server training;
- synchronized multi-GPU training;
- cluster scheduler integration.

## 7. Sim2Real Boundary

The Sim2Real contribution is an interface and validation ladder. The 1v1 line has been exercised on real robots after the simulation/replay validation. The 50v50 line has not; it remains simulation-stage evidence.

- ROS2 topics, services and actions are separated from simulation internals;
- Nav2, AprilTag detection, EKF, shooter services and `/cmd_vel` remain the deployment contract;
- `docs/sim2real.md` defines calibration order, domain randomization, validation steps and required logs;
- IsaacLab replay checks physical plausibility before real-world deployment.

Not publicly claimed:

- no public full statistical 1v1 hardware win-rate table yet;
- no public 1v1 migration success percentage yet;
- no public long-horizon real arena dataset;
- no public statistical comparison between sim-only and real-robot matches.
- no 50v50 real-robot deployment.

Correct public wording:

```text
The project provides a ROS2/IsaacLab Sim2Real contract and validation procedure. The 1v1 scenario has real-robot experiment coverage, while public quantified hardware statistics are not yet packaged. The 50v50 scenario remains simulation-stage only.
```

## 8. How to Extend the Evidence

To claim a new capability:

1. define the scenario and agent count;
2. document layout, rules and done reasons;
3. add tests for collision, line-of-sight, scoring and replay invariants;
4. run at least 64 evaluation episodes, preferably 128+;
5. publish JSON/CSV metrics;
6. generate full replay media;
7. document failure cases and residual risks;
8. update this file with exact metrics and paths.

For real-robot Sim2Real claims, add:

- hardware configuration;
- calibration logs;
- rosbag2 recordings;
- number of real runs;
- success/failure definition;
- win rate or task success rate;
- collision/stuck/relocalization statistics;
- lighting and arena condition notes.
