# RoboCup VisionRL RL Interface

This folder provides the reinforcement-learning bridge for the RoboCup IsaacLab scene. The new research training path is an object-centric world-model + SAC Flow self-play actor over the fast Python rule environment, then audited replay in IsaacLab. The previous MAPPO-style centralized-training/decentralized-execution actor remains as the stable baseline.

## Current Policy

- Recommended research algorithm: object-centric world-model SAC Flow self-play.
- Stable baseline algorithm: MAPPO-style self-play with a centralized critic and local actors.
- Final actor mode: dual actor, with a yellow actor and a blue actor trained from team-specific expert priors.
- Policy mode: residual expert, so the learned actor adjusts a rule-aware tactical prior instead of relearning basic navigation from scratch.
- Observation dimension: 46 local features per robot.
- MAPPO critic input: local observation concatenated with opponent observation.
- SAC Flow critic/world model input: explicit object-centric state with robot, target, pushable box and base-armor blocker tokens.
- Action dimension: 6 high-level tactical controls.

Current dual-expert checkpoint after local training/evaluation:

`isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507/policy.pt`

`isaaclab_sim/output/` is a generated runtime directory and is not committed.
Clone users can run the smoke tests without a checkpoint, then run the training
command below to regenerate the checkpoint before MAPPO evaluation/export.

Final closure report:

`docs/rl_dual_experts_contact_hull_seed260507_report.md`

## Open-Source-Inspired Innovation

The latest run adds two practical ideas selected from current robot-learning/open-source practice:

- Reset-time Sim2Real domain randomization, following the same robustness principle used by dynamics/domain-randomization work and Isaac Lab's event/randomization model. This project randomizes drive scale, turn scale, push response, shot accuracy, drift loss and sensor noise per episode.
- A geometry-aware action shield, adapted from invalid-action/action-masking practice in PPO ecosystems. Instead of silently letting the policy fire through blockers or brush its own critical assets, the environment suppresses unsafe contact/fire commands and records the shield intervention in `info`.

Reference material reviewed for this update:

- OpenAI dynamics randomization: <https://openai.com/index/sim-to-real-transfer-of-robotic-control-with-dynamics-randomization/>
- Isaac Lab GitHub repository: <https://github.com/isaac-sim/IsaacLab>
- Isaac Lab domain randomization/event APIs: <https://isaac-sim.github.io/IsaacLab/main/index.html>
- Isaac Lab event manager source docs: <https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/event_manager.html>
- Stable-Baselines3 Contrib GitHub repository: <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>
- Stable-Baselines3 Contrib Maskable PPO: <https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html>
- Invalid action masking paper: <https://arxiv.org/abs/2006.14171>

## Action Contract

The actor output is clipped to `[-1, 1]` and mapped to:

- `target_selector`: choose among visible/reachable opponent targets.
- `base_rush_gate`: decide when to stop normal-target cleanup and attempt the base.
- `block_interference_gate`: decide whether opponent contact/blocking is worth the time risk.
- `recovery_gate`: request spin/relocalization only when confidence is low.
- `fire_gate`: decide when the laser should stay enabled for the dwell window.
- `risk_preference`: controls closer base angles, push attempts and contact tolerance.

## Observation Contract

The observation includes pose, score/armor state, target progress, opponent relative pose, route/fire geometry and compact multi-sensor fusion features:

- wheel/IMU motion consistency
- EKF localization confidence
- scan/costmap clearance
- front-left and front-right ToF clearance
- bumper/contact flags
- camera target visibility
- pushable obstacle relative poses and contact state

The ROS2-side contract is documented in `isaaclab_sim/rl/configs/mappo_selfplay.yaml` and maps to `/wheel/odom`, `/imu/data_raw`, `/scan`, `/odometry/filtered`, `/target_detection`, `/cmd_vel`, ToF/bumper topics and shooter enable/fire services.

## Rule Model

- Robots attack opponent-owned targets only.
- Normal targets are placed about 45 degrees to the surrounding wall/corner geometry.
- Base targets are smaller and recessed inside the base.
- Ground-touching blue armor blocks navigation and laser line of sight until removed.
- Target contact is non-scoring and does not knock targets down.
- Laser shots require a clear line of sight and at least 0.80 s dwell; normal targets use 5-50 cm shooter-outlet range and recessed base targets use 20-80 cm.
- Hit probability increases with better distance/lateral alignment and longer dwell up to the capped confidence.
- Pushable boxes are dynamic rigid obstacles; robots may push them, but jammed boxes can block the route.
- Robot-robot contact is allowed as a tactical event and does not trigger relocalization by itself.
- Static-wall, armor, target and jammed-box penetration are strict replay failures.

## Team Experts

`expert_policy.py` now exposes separate priors:

- `yellow_expert_action`: yellow-side target order, push timing, side-gate window and recessed blue-base attack rhythm.
- `blue_expert_action`: blue-side target order, push timing, side-gate window and recessed yellow-base attack rhythm.
- `scripted_action`: compatibility wrapper that dispatches to the correct team expert.

The MAPPO trainer supports `--actor-mode dual`; when resuming from an older shared checkpoint it migrates `actor.*` weights into both `yellow_actor.*` and `blue_actor.*`, then continues training them independently.

Dual-actor checkpoints can be exported either as an auto-dispatch actor or as
fixed yellow/blue experts:

```bash
python3 isaaclab_sim/rl/export_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507/policy.pt \
  --team yellow \
  --output-dir isaaclab_sim/output/policy_export/yellow_expert

python3 isaaclab_sim/rl/export_policy.py \
  --checkpoint isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507/policy.pt \
  --team blue \
  --output-dir isaaclab_sim/output/policy_export/blue_expert
```

## Training

Recommended object-centric world-model SAC Flow run:

```bash
python3 isaaclab_sim/rl/train_world_model_sacflow_selfplay.py \
  --config isaaclab_sim/rl/configs/world_model_flow.yaml \
  --timesteps 200000 \
  --num-envs 32 \
  --batch-size 1024 \
  --gradient-steps 2 \
  --device cuda \
  --seed 260707 \
  --output ../output/rl/world_model_sacflow_seed260707
```

This path replaces the Gaussian PPO/MAPPO actor with a velocity-reparameterized flow actor, trains a centralized twin-Q critic from replay, and jointly learns an auxiliary object-centric dynamics model. The object state explicitly tracks both robots, all targets, red pushable boxes and active base armor blockers so later TD-MPC2/Dreamer-style imagined rollouts can be added without changing the replay contract.

Evaluate and export the resulting checkpoint with:

```bash
python3 isaaclab_sim/rl/evaluate_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 64 \
  --stochastic \
  --output isaaclab_sim/output/eval/world_model_sacflow_eval64.json

python3 isaaclab_sim/rl/evaluate_strategy_contract.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 64 \
  --stochastic \
  --output-json isaaclab_sim/output/eval/world_model_sacflow_contract_eval64.json \
  --output-csv isaaclab_sim/output/eval/world_model_sacflow_contract_eval64.csv

python3 isaaclab_sim/rl/export_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --format torchscript \
  --output-dir isaaclab_sim/output/policy_export/world_model_sacflow_seed260707
```

Fast single-agent smoke test:

```bash
cd isaaclab_sim/rl
python3 robocup_visionrl_gym_env.py
```

Two-agent self-play smoke test:

```bash
cd isaaclab_sim/rl
python3 robocup_visionrl_selfplay_env.py
```

Dual-expert continuation run:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config isaaclab_sim/rl/configs/mappo_selfplay.yaml \
  --timesteps 16384 \
  --num-envs 32 \
  --rollout-steps 64 \
  --update-epochs 2 \
  --minibatch-size 2048 \
  --hidden-dim 256 \
  --device cuda \
  --seed 260507 \
  --policy-mode residual_expert \
  --residual-scale 0.04 \
  --residual-l2-coef 0.0018 \
  --actor-mode dual \
  --domain-randomization \
  --resume isaaclab_sim/output/rl/mappo_dual_experts_recovery_cooldown_blend052_seed260505/policy.pt \
  --output isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507
```

## Evaluation And Audit

Latest stochastic 64-episode dual-expert evaluation:

| Episodes | Yellow Win | Blue Win | Draw/Timeout | Base Wins/Episode | Own-Target Penalties |
|---:|---:|---:|---:|---:|---:|
| 64 | 50.00% | 43.75% | 6.25% | 0.9375 | 0.0 |

Evaluation files:

- `isaaclab_sim/output/eval/mappo_dual_experts_contact_hull_seed260507_eval64.json`
- `isaaclab_sim/output/eval/mappo_dual_experts_contact_hull_seed260507_eval64.csv`

The latest strict replay audit is:

`docs/rl_dual_experts_contact_hull_seed260507_strict8.md`

Strict audit summary:

- episodes: 8
- yellow win rate: 37.50%
- blue win rate: 62.50%
- hard violations: 0
- warnings: 0
- own-target penalties per episode: 0.0
- base wins per episode: 1.000

The strict split is a small legality audit sample; side balance is reported from the 64 stochastic evaluation episodes above.

## IsaacLab Replay

The audited trace can be rendered by `isaaclab_sim/robocup_visionrl_arena_sim.py` with `--replay_trace`, `--replay_events`, `--record_video` and `--record_view`.

Current tracked MP4s:

- `docs/media/isaaclab_contact_hull_top.mp4`
- `docs/media/isaaclab_contact_hull_yellow_pov.mp4`
- `docs/media/isaaclab_contact_hull_blue_pov.mp4`
