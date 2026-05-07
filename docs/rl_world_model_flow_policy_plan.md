# Object-Centric World Model + SAC Flow Policy Plan

Date: 2026-05-07

This document defines the current formal learning architecture for
`RoboCupVisionRL_IsaacLab_ROS2`. The project uses an object-centric world model
with SAC Flow / PolicyFlow-style self-play for two-robot visual target combat.
All figures, evaluations and replay media in the README are generated from this
current architecture.

## 1. Algorithm Positioning

The formal method is:

`object-centric world model + SAC Flow tactical actor + centralized twin-Q critic + strict rule replay`

The method is designed for:

- long-horizon target ordering
- pushable-box route selection
- early base-rush timing
- yellow/blue asymmetric tempo
- legal line-of-sight shooting with dwell time
- strict collision and penetration auditing
- Sim2Real export into ROS2 behavior nodes

## 2. Core Modules

### 2.1 Object-Centric State Encoder

The object state explicitly tracks:

- yellow and blue robot poses, velocities and confidence
- normal targets, owners, yaw and knocked state
- base targets and removable armor blockers
- pushable red boxes and their persistent map poses
- start partitions, walls and static blockers
- laser line-of-sight, shooting range and dwell state

The world model learns transitions over these objects instead of compressing the
entire match into a single unstructured vector. This makes push-box motion,
armor removal and base-rush timing visible to the learning system.

### 2.2 SAC Flow Tactical Actor

The actor is a velocity-reparameterized flow policy over six high-level tactical
controls:

- target selection
- base-rush gate
- block/interference gate
- recovery gate
- fire gate
- risk preference

The flow actor can represent multi-modal choices such as clearing another
normal target, nudging a box route, holding fire pose, or switching to a base
attack window.

### 2.3 Centralized Twin-Q Critic

The critic uses local observations plus object-centric state during training.
Runtime execution remains decentralized: each robot exports its local tactical
actor and leaves safety gates to the rule-aware ROS2 behavior layer.

### 2.4 Rule-Aware Residual Expert

The learned actor adjusts a rule-aware prior rather than relearning basic
navigation and target legality from scratch. The prior enforces:

- opponent-owned targets only
- no own-base hit
- legal shooting distance
- line-of-sight raycast
- 0.8 second dwell gate
- base armor blocker gating
- pushable-box collision consistency

### 2.5 Strict Replay Contract

Every formal result must pass strict replay checks:

- no static obstacle penetration
- no pushable box penetration
- no illegal own-target score
- no base hit before armor removal
- no teleporting or impossible differential-drive step
- score and armor changes must match rule events
- three IsaacLab replay views must be generated from the same trace

## 3. Training Pipeline

1. Run geometry and rule prechecks.
2. Generate expert-guided rollouts for yellow and blue.
3. Train the world-model SAC Flow self-play policy.
4. Evaluate with stochastic multi-seed games.
5. Run strict replay audit.
6. Record IsaacLab MP4 from top, yellow first-person and blue first-person views.
7. Convert the three replay videos to README GIF previews.
8. Regenerate publication-style figures and editable PPTX master.

## 4. Current Checkpoint And Data

Checkpoint:

`isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun/policy.pt`

Mirrored README data:

`docs/rl_data/world_model_sacflow_final/`

Generated figure sources:

- `docs/figures/paper/fig01_project_overview.png`
- `docs/figures/paper/fig02_method_architecture.png`
- `docs/figures/paper/fig03_training_and_results.png`
- `docs/figures/paper/fig04_ablation_and_safety.png`
- `docs/figures/paper/fig05_sim2real_replay_pipeline.png`
- `docs/figures/paper/world_model_sacflow_paper_figures_master.pptx`
- `docs/figures/rl/rl_training_curve_gpu.svg`
- `docs/figures/rl/rl_strategy_event_metrics.svg`
- `docs/figures/rl/rl_target_base_metrics.svg`
- `docs/figures/rl/rl_box_push_metrics.svg`

## 5. Behavior Improvements

The current policy layer includes:

- safe micro-aim scanning at normal and base fire poses
- centimeter-level side/radial base fire candidates after armor removal
- current-position hold when fire geometry is already ready
- route-aware target selection
- pushable-box contact and recoil modeling
- stuck recovery through conservative reverse-turn escape

These changes reduce cases where a robot reaches a good shooting pose but
freezes just outside the laser hit radius.

## 6. Result Contract

Formal reports must include:

- yellow win rate
- blue win rate
- draw rate
- average score
- average match time
- normal target count distribution
- base hit success grouped by normal-target count
- push count and final box displacement
- static and pushable penetration counts
- robot contact and abnormal spin counts
- strict replay hard violations and warnings
- three replay GIFs in README, with top view first

## 7. Future Work

- Increase route diversity for four-normal-target base attacks.
- Strengthen blue-side push-box exploration so both teams show box-route skill.
- Add a compact world-model imagination report for target order and base timing.
- Keep README figures regenerated from mirrored result data.
