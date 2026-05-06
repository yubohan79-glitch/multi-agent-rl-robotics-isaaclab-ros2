# Dual-Expert Recovery Cooldown And Contact-Fix Report

Date: 2026-05-06

## What Changed

- Added a relocalization cooldown (`RECOVERY_COOLDOWN_S=2.50`) and lowered the recovery trigger threshold to avoid repeated spin-relocalization loops.
- Made expert recovery more conservative: experts recover on hard contact, or on critical localization confidence with bad scan/odometry, rather than every mild confidence dip.
- Slightly adjusted `blue_expert` two-hit base-rush timing so blue still learns the opened-armor side window instead of giving up early.
- Fixed robot-robot contact separation so contact resolution cannot push a robot into walls, armor blockers, targets or pushable boxes.
- Added tests for relocalization cooldown and contact separation against static blockers.
- Extended `build_balanced_dual_actor_policy.py` with `--blend-alpha`, allowing partial residual-actor symmetrization while keeping yellow/blue expert priors separate.

## Why

The previous dual-expert checkpoint had good geometry legality but blue win rate stayed at `42.19%`, and blue relocalization was still high. Full residual symmetrization improved blue but caused static/box penetration. The real bug was contact separation: after robot-robot collision the solver separated the robots without rechecking static blockers. The final selected checkpoint uses a partial residual blend (`alpha=0.52`) plus the contact solver fix.

## Verification

- Unit/contract tests: `61 passed`.
- Rule geometry audit: `target_count=10`, `physics_checks=3`, `failures=0`.
- Geometry audit table: `docs/rl_rule_geometry_audit.md`.
- IsaacLab video frame check: all three MP4s are readable, non-empty and non-blank.

## Training And Checkpoints

Continuation checkpoint:

`isaaclab_sim/output/rl/mappo_dual_experts_recovery_cooldown_seed260505/policy.pt`

Selected deployment checkpoint:

`isaaclab_sim/output/rl/mappo_dual_experts_recovery_cooldown_blend052_seed260505/policy.pt`

Training curve:

`isaaclab_sim/output/rl/mappo_dual_experts_recovery_cooldown_seed260505/training_curve.csv`

Selection notes:

- `blend_alpha=0.45`: no penetration, but blue stayed at `42.19%`.
- `blend_alpha=0.60`: reached `48.44% / 45.31%` before contact fix but exposed the robot-contact penetration bug; after contact fix it became blue-heavy.
- `blend_alpha=1.00`: rejected because static and box penetration appeared.
- `blend_alpha=0.52`: selected because it has zero penetration, blue at `50.00%`, and the smallest overall safety regression among the balanced candidates.

## 64-Episode Evaluation

Evaluation files:

- `isaaclab_sim/output/eval/mappo_dual_experts_recovery_cooldown_blend052_seed260505_eval64.json`
- `isaaclab_sim/output/eval/mappo_dual_experts_recovery_cooldown_blend052_seed260505_eval64.csv`

| Metric | Value |
|---|---:|
| Episodes | 64 |
| Yellow win rate | 42.19% |
| Blue win rate | 50.00% |
| Draw/timeout rate | 7.81% |
| Mean episode time | 41.1484 s |
| Mean yellow score | 37.3438 |
| Mean blue score | 42.8125 |
| Mean yellow normal hits | 2.0312 |
| Mean blue normal hits | 2.1875 |
| Robot contacts / episode | 0.0938 |
| Yellow relocalization / episode | 0.0781 |
| Blue relocalization / episode | 0.7812 |
| Yellow abnormal spin steps / episode | 10.2031 |
| Blue abnormal spin steps / episode | 9.4375 |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Repeat target-order events | 0 |

Ordinary-target count distribution:

| Team | 1 target | 2 targets | 3 targets | 4 targets |
|---|---:|---:|---:|---:|
| Yellow | 4.69% | 87.50% | 7.81% | 0.00% |
| Blue | 1.56% | 78.12% | 20.31% | 0.00% |

Base success by ordinary-target count:

| Team | 1 hit | 2 hits | 3 hits | 4 hits |
|---|---:|---:|---:|---:|
| Yellow | 0 / 0 | 24 / 61 = 39.34% | 3 / 5 = 60.00% | 0 / 0 |
| Blue | 0 / 0 | 25 / 63 = 39.68% | 7 / 12 = 58.33% | 0 / 0 |

Pushable-box metrics:

| Metric | box_ne | box_sw |
|---|---:|---:|
| Mean final displacement | 0.5338 m | 0.5567 m |
| Mean max displacement | 0.5338 m | 0.5567 m |

## Strict Replay

Strict replay files:

- Summary: `isaaclab_sim/output/replay/mappo_dual_experts_recovery_cooldown_blend052_seed260505_strict8/strict_replay_summary.json`
- Trace: `isaaclab_sim/output/replay/mappo_dual_experts_recovery_cooldown_blend052_seed260505_strict8/strict_replay_trace.csv`
- Events: `isaaclab_sim/output/replay/mappo_dual_experts_recovery_cooldown_blend052_seed260505_strict8/strict_replay_events.jsonl`
- Report: `docs/rl_dual_experts_recovery_cooldown_blend052_strict8.md`

Strict result:

- Episodes: 8
- Yellow win rate: 50.00%
- Blue win rate: 37.50%
- Draw/timeout: 12.50%
- Hard violations: 0
- Warnings: 0
- Own-target penalties: 0.0 / episode
- Robot contacts: 0.25 / episode
- Recovery events: 0.0 / episode
- Base wins: 0.875 / episode

## IsaacLab Three-View Replay

Rendered strict episode: `3` (`28.1 s`, blue base win, zero hard violations).

| File | View | Resolution | Frames | Duration | Size |
|---|---|---:|---:|---:|---:|
| `docs/media/isaaclab_blend052_contactfix_overview.mp4` | overview | 1280x720 | 373 | 31.08 s | 4.38 MB |
| `docs/media/isaaclab_blend052_contactfix_yellow_pov.mp4` | yellow POV | 1280x720 | 373 | 31.08 s | 4.29 MB |
| `docs/media/isaaclab_blend052_contactfix_blue_pov.mp4` | blue POV | 1280x720 | 373 | 31.08 s | 4.69 MB |

Video check conclusions:

- Both robots start together and attack opponent targets.
- Targets are present, not embedded in walls, and keep the 45-degree plane audit pass.
- Both pushable red boxes move to persistent new positions in the IsaacLab logs.
- Base armor is removed before the final base hit; the base target is not scored through intact armor.
- No static or box penetration appears in the 64-episode contract evaluation for the selected checkpoint.

## Remaining Risks

- Yellow win rate is `42.19%`, still slightly below the target band. This is the remaining bottleneck after eliminating penetration and reducing recovery loops.
- Abnormal spin steps are still visible in aggregate metrics; strict video episode 3 was selected because it has a clean, complete match flow, but longer training should reduce these metrics further.
- The policy still strongly prefers 2-target base rushes; this matches the user's observed real-match tendency, but it means 4-target statistics remain sparse.

## Figures To Draw

- System architecture diagram.
- ROS2 + IsaacLab + RL closed-loop diagram.
- Multi-sensor fusion structure diagram.
- Yellow/blue win-rate bar chart.
- MAPPO training curve from `training_curve.csv`.
- Ordinary-target hit-count distribution.
- Base success rate vs ordinary-target count.
- Pushable-box displacement statistics.
- Three-view replay screenshot collage.
