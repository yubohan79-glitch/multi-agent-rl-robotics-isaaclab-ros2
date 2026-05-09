# Admissions Project Brief

## Project Summary

Object-Centric World-Model Flow RL for Multi-Agent Robotics is a ROS2 + IsaacLab project built around a RoboCup-style adversarial visual navigation task. The system combines real robot software infrastructure, simulation replay, rule-level reinforcement learning, object-centric state modeling, and auditable evaluation artifacts.

The main validated scenario is a two-robot yellow-vs-blue match. Each robot must navigate, identify opponent targets, push physical-style red boxes, respect base blockers, satisfy laser range and dwell-time constraints, and avoid illegal hits or penetration. The learning stack uses an object-centric world model with SAC Flow / PolicyFlow-style self-play and rule-aware action shielding.

## Why It Is Relevant for Graduate Applications

This project demonstrates work across several research and engineering layers:

| Layer | Demonstrated Work |
| --- | --- |
| Robotics systems | ROS2 Jazzy workspace, Nav2 integration, launch files, robot descriptions, behavior orchestration and shooter services |
| Simulation | IsaacLab replay scene, differential-drive robot behavior, pushable obstacles, base blockers and target interaction |
| Learning | Object-centric state representation, world-model-assisted SAC Flow / PolicyFlow-style self-play and residual expert behavior |
| Safety and rules | Action shielding, legal target ownership, line-of-sight, laser dwell time, range gates, collision and penetration audits |
| Evaluation | Multi-seed evaluation, strict replay audit, JSON/CSV metrics, replay videos, GIFs and generated figures |
| Sim2Real | ROS2 deployment contract and 1v1 real-robot experiment coverage, with remaining hardware evidence boundaries documented clearly |

## Key Public Results

### Two-Robot Main Scenario

The public 128-episode stochastic evaluation reports:

| Metric | Value |
| --- | ---: |
| Yellow win rate | 49.22% |
| Blue win rate | 50.78% |
| Draw rate | 0.00% |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Robot contacts per game | 0.00 |

The strict replay audit reports zero hard violations and zero own-target penalties for the selected replay set.

Primary evidence:

- `docs/rl_data/world_model_sacflow_final/training_summary.json`
- `docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json`
- `docs/rl_data/world_model_sacflow_final/strict_replay_summary.json`
- `docs/media/最终回放_三视角同步拼接版.gif`

### Large-Scale 50v50 Extension

The repository also includes a simulation-stage 50v50 benchmark for scalable rule-level multi-agent strategy. It uses a staged 5v5 -> 10v10 -> 25v25 -> 50v50 curriculum and evaluates the final stage over 256 games.

This extension is intentionally documented as simulation-stage evidence. It is not claimed as 100-robot hardware deployment or full rigid-body IsaacLab reinforcement learning for all 100 vehicles.

Primary evidence:

- `docs/large_scale_50v50_plan.md`
- `docs/large_scale_50v50_curriculum_plan.md`
- `docs/large_scale_50v50_report.md`
- `docs/rl_data/large_scale_50v50/`
- `docs/media/large_scale_50v50_isaaclab_replay.mp4`

## What I Would Emphasize in an Application

The strongest application framing is:

> I built a reproducible ROS2 + IsaacLab multi-agent robotics learning stack that connects rule-aware simulation, object-centric reinforcement learning, strict evaluation, replay visualization and real-robot deployment contracts. The project is not only a trained policy; it is an end-to-end robotics research artifact with explicit evidence boundaries.

Recommended emphasis:

- real robotics system integration rather than only algorithm scripting;
- explicit rule modeling and safety checks;
- evaluation discipline: metrics, replay audits and failure boundaries;
- Sim2Real thinking through ROS2 contracts, even where public hardware statistics are still incomplete;
- scalability exploration through the 50v50 simulation-stage benchmark.

## Current Limitations

The repository intentionally does not overclaim:

- No public statistical 1v1 hardware table or rosbag bundle is packaged yet.
- The 50v50 result is simulation-stage and rule-level, not hardware validation.
- The 50v50 replay is tactical visualization of the accepted trace, not full 100-vehicle rigid-body RL training evidence.
- The current results are portfolio-level research engineering evidence, not a peer-reviewed SOTA benchmark.

These limitations are documented in `docs/capability_boundaries.md`.

## Reviewer Reading Path

For a fast review:

1. Read the README top section and watch the first GIF.
2. Open `docs/capability_boundaries.md` to check what is and is not claimed.
3. Inspect `docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json`.
4. Watch the three-view replay GIFs or MP4s under `docs/media/`.
5. If interested in scaling, read `docs/large_scale_50v50_report.md`.

## Suggested Application Description

Short version:

> Built a ROS2 + IsaacLab multi-agent robotic learning system for a RoboCup-style visual adversarial task, combining object-centric world models, SAC Flow / PolicyFlow self-play, rule-aware action shielding, strict replay audits, and Sim2Real deployment contracts.

Long version:

> This project implements an end-to-end robotics learning stack for adversarial visual navigation. I designed the rule environment, ROS2 runtime contracts, IsaacLab replay pipeline, object-centric learning representation, SAC Flow / PolicyFlow-style self-play, evaluation scripts, replay audits, figures and documentation. The validated main setting is a two-robot match with balanced 128-episode evaluation and zero reported penetration/contact violations; the repository also includes a clearly bounded 50v50 simulation-stage benchmark for scalable multi-agent strategy research.
