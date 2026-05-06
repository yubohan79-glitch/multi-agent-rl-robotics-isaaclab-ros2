# Full GPU Reinforcement-Learning Strategy Report

This document records the completed MAPPO/self-play training run, the tactical policy contract, and the data-driven figures generated from the finished experiment.

Note: this report describes the archived full run before the precision shooter-outlet update. The newer run uses a `0.05 m` to `0.50 m` shooter-outlet range gate, 38-dimensional local observations, pushable obstacle vectors and non-scoring target contact. See `docs/rl_precision_shooting_model.md`.

## Executive Summary

The reinforcement-learning module trains a high-level two-robot competition strategy for the RoboCup VisionRL scenario. The policy does not directly output wheel speeds. Instead, it selects tactical commands that are executed by the ROS2/Nav2/vision/shooter stack:

- which opponent target to attack
- when to rush the opponent base target
- when to block or interfere with the opponent
- when to request localization recovery after collision or stuck behavior
- when the shooter gate is allowed to fire
- how much tactical risk to accept

The complete GPU run used MAPPO-style centralized training and decentralized execution. The critic sees both robots' observations during training, while each deployed robot only needs its own local observation.

## Competition Rule Encoding

The rule environment follows the project competition contract:

- two robots compete from opposite sides of a 3 m x 3 m arena
- each robot must enter the opponent field to attack opponent targets
- own-side targets are forbidden
- normal targets award score and remove opponent armor
- the base target is the terminal win objective
- obstacles, armor plates, fences and robot-robot contact affect motion and localization confidence
- collision or stuck events reduce localization confidence and can trigger a spin-scan recovery tactic

This is intentionally a high-level strategy environment. Real wheel dynamics, camera frames, lidar, IMU and shooter actuation are handled by IsaacLab/ROS2 modules and the Sim2Real contract.

## Policy Interface

Local observation dimension: `34`

Centralized critic observation dimension: `68`

Action dimension: `6`

| Action | Meaning | Deployment consumer |
| --- | --- | --- |
| `target_selector` | chooses normal target vs base target intent | `rcvrl_behavior` target planner |
| `base_rush_gate` | permits high-risk base attack | competition behavior state machine |
| `block_interference_gate` | chooses blocking/interference posture | Nav2 goal generator |
| `recovery_gate` | requests spin-scan localization recovery | EKF/Nav2 recovery branch |
| `fire_gate` | permits shooter-service request | shooter safety gate |
| `risk_preference` | adjusts route standoff and blocking aggression | behavior planner |

The shooter remains behind an opponent-target safety gate, so the learned actor cannot bypass the rule that own targets must not be fired upon.

## Completed GPU Training Run

Command:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py --config configs/mappo_selfplay.yaml
```

Equivalent explicit command:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --timesteps 500000 \
  --num-envs 32 \
  --rollout-steps 256 \
  --update-epochs 4 \
  --minibatch-size 2048 \
  --hidden-dim 256 \
  --device cuda \
  --output ../output/rl/mappo_selfplay_full_gpu
```

Hardware and software:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| PyTorch | 2.11.0+cu128 |
| CUDA available | true |
| Parallel environments | 32 |
| Agent steps | 507,904 |
| Wall time | 474.243 s |
| Final throughput | 1127 steps/s |
| Final mean reward | 0.01635 |
| Final critic explained variance | 0.2958 |

Training artifacts are stored locally under `isaaclab_sim/output/rl/mappo_selfplay_full_gpu/`. Git-tracked data copies are stored under `docs/rl_data/mappo_selfplay_full_gpu/`.

## Evaluation Results

| Evaluation | Episodes | Yellow win | Blue win | Timeout/draw | Mean yellow score | Mean blue score | Normal hits/episode | Base wins/episode | Own-target penalties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MAPPO deterministic | 64 | 0.0% | 0.0% | 100.0% | 10.00 | 0.00 | 2.00 | 0.00 | 0.00 |
| MAPPO stochastic | 64 | 0.0% | 40.6% | 59.4% | 11.17 | 38.44 | 4.86 | 0.41 | 0.00 |
| Scripted baseline | 64 | 0.0% | 0.0% | 100.0% | n/a | n/a | 5.00 | 0.00 | 0.00 |

The deterministic actor is conservative at this training budget. The stochastic actor shows the intended tactical behavior: it sometimes accepts risk, blocks/interferes, clears enough normal targets to open the base window, and achieves base-hit terminal wins without own-target penalties.

## Strict Replay Audit

After training, the stochastic MAPPO actor was replayed for 32 episodes with step-by-step rule and motion checks. The audit checks action bounds, finite poses, arena-boundary penetration, static-obstacle penetration, differential-drive step limits, opponent-only target selection/fire, own-base safety, score monotonicity and armor-state consistency.

Result: `PASS`

- hard violations: `0`
- warnings: `0`
- own-target penalties per episode: `0.0`
- target-contact knockdowns per episode: `0.0`
- base wins per episode: `0.3438`
- normal hits per episode: `4.9688`

Full report: `docs/rl_strict_replay_audit.md`

## Representative Learned Episode

A representative stochastic MAPPO episode ended with a blue base-hit win:

- duration: `157.7 s`
- final score: yellow `10`, blue `75`
- final armor: yellow `1`, blue `2`
- normal hits: `5`
- base-hit wins: `1`
- own-target penalties: `0`
- blue target order: `T06_EastBelowGate -> T04_WestBelowGate -> T07_SouthWest -> YellowBaseTarget`
- yellow target order: `T05_EastAboveGate -> T03_WestAboveGate`

This episode demonstrates the intended strategy: the policy does not hard-code a fixed number of targets. It attacks normal targets when needed, then switches to the opponent base when the risk/visibility tradeoff becomes favorable.

## Generated Figures

The following SVG figures are generated from the completed training and evaluation data:

- `docs/figures/rl/rl_training_curve_gpu.svg`
- `docs/figures/rl/rl_strategy_event_metrics.svg`
- `docs/figures/rl/rl_policy_episode_trace.svg`
- `docs/figures/rl/rl_tactical_contract.svg`

Source data:

- `docs/rl_data/mappo_selfplay_full_gpu/training_curve.csv`
- `docs/rl_data/mappo_selfplay_full_gpu/training_curve.jsonl`
- `docs/rl_data/mappo_selfplay_full_gpu/training_summary.json`
- `docs/rl_data/mappo_selfplay_full_gpu/mappo_full_gpu_eval.json`
- `docs/rl_data/mappo_selfplay_full_gpu/mappo_full_gpu_eval_stochastic.json`
- `docs/rl_data/mappo_selfplay_full_gpu/scripted_rules_baseline_eval.json`

Regenerate figures with:

```bash
python isaaclab_sim/rl/generate_rl_figures.py
```

## Sim2Real Interpretation

The learned policy is intended to sit above the robot runtime rather than replace it:

- IsaacLab/RL actor outputs high-level tactical commands
- `rcvrl_behavior` checks rule legality and converts tactics into navigation/search/align/fire states
- Nav2 executes route following and obstacle avoidance
- AprilTag perception confirms target identity and relative pose
- EKF fuses wheel odometry, IMU and lidar-derived localization
- shooter services execute only after opponent-target safety approval

This keeps the learned component portable: the real robot can keep deterministic low-level controllers while the RL layer decides tactical timing and risk.

## Policy Export

The trained decentralized actor can be exported as TorchScript:

```bash
python3 isaaclab_sim/rl/export_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_selfplay_full_gpu/policy.pt \
  --format torchscript \
  --device cpu \
  --output-dir isaaclab_sim/output/policy_export/mappo_selfplay_full_gpu
```

The export contains only the local actor, not the centralized critic. The output action remains the six-dimensional tactical vector documented above, and the opponent-target safety gate remains outside the learned actor in `rcvrl_behavior`.

## Limitations and Next Training Pass

The current result is useful and data-backed, but not final competition-grade policy convergence. Recommended next steps:

1. Increase training to 2 to 5 million agent steps.
2. Add asymmetric self-play sampling so yellow/blue do not collapse into one dominant side.
3. Add reward ablations for base-rush, blocking, recovery and own-target safety.
4. Export policy to ONNX/TorchScript for ROS2 deployment testing.
5. Evaluate deterministic actors with temperature or action-noise scheduling instead of raw mean action.
6. Add IsaacLab physics-in-the-loop episodes after rule-level strategy convergence.

The important current milestone is complete: the project now has a GPU-trained MAPPO strategy, archived run data, reproducible evaluation, and figures generated from real experiment outputs.
