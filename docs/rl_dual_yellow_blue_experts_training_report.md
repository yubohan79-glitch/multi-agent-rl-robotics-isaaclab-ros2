# Yellow/Blue Dual-Expert MAPPO Update

Date: 2026-05-05

## What Changed

- Split the single scripted expert into `yellow_expert_action` and `blue_expert_action`.
- Added team-specific target order, side-gate preference, push-risk timing and base-rush tempo in `isaaclab_sim/rl/expert_policy.py`.
- Kept `scripted_action(env, team)` as a compatibility wrapper that dispatches to the correct team expert.
- Changed MAPPO defaults to `policy_mode=residual_expert`, `actor_mode=dual`, `residual_scale=0.04`.
- Added shared-to-dual checkpoint migration in `train_mappo_selfplay_parallel_torch.py`.
- Removed the previous temporary `base_rush_priority_team="blue"` compensation from the environment.
- Updated evaluation accounting so base-rush attempts are counted per episode/hit-count window, not only after a non-dwell shot event.

## Why

The previous shared expert made both cars learn nearly the same rhythm. That hid side-specific problems: one side could overfit a base-rush route, push boxes at the wrong time, or repeatedly choose a target that was not actually selectable by the tactical target sorter. The dual-expert setup gives each side its own prior while still allowing MAPPO to learn residual corrections.

## Verification

- Unit and contract tests: `59 passed`.
- Geometry audit: `target_count=10`, `physics_checks=3`, `failures=0`.
- Geometry audit table: `docs/rl_rule_geometry_audit.md`.
- Expert smoke evaluation:
  - JSON: `isaaclab_sim/output/eval/yellow_blue_experts_smoke8.json`
  - Yellow win: 37.5%, Blue win: 37.5%, Draw: 25.0%
  - Static penetrations: 0
  - Box penetrations: 0

## Training

Dual continuation checkpoint:

`isaaclab_sim/output/rl/mappo_dual_yellow_blue_experts_seed260506_rs004/policy.pt`

Source short training checkpoint:

`isaaclab_sim/output/rl/mappo_dual_yellow_blue_experts_seed260506_short/policy.pt`

Training curve:

`isaaclab_sim/output/rl/mappo_dual_yellow_blue_experts_seed260506_short/training_curve.csv`

Notes:

- The trainer resumed from `mappo_sidegate_rigidbody_shared_seed506_rulefix/policy.pt`.
- Shared actor weights were migrated into both `yellow_actor` and `blue_actor`.
- A longer 65k run was attempted, but the GPU was concurrently occupied and timed out before final checkpoint save. The saved model is the 16k short continuation plus deployment residual-scale override.

## 64-Episode Evaluation

Evaluation files:

- `isaaclab_sim/output/eval/mappo_dual_yellow_blue_experts_seed260506_rs004_eval64.json`
- `isaaclab_sim/output/eval/mappo_dual_yellow_blue_experts_seed260506_rs004_eval64.csv`

Summary:

| Metric | Value |
|---|---:|
| Episodes | 64 |
| Yellow win rate | 51.56% |
| Blue win rate | 42.19% |
| Draw/timeout rate | 6.25% |
| Mean episode time | 37.0453 s |
| Mean yellow score | 41.3281 |
| Mean blue score | 35.5469 |
| Mean yellow normal hits | 2.0781 |
| Mean blue normal hits | 2.0469 |
| Robot contacts / episode | 0.0625 |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Repeat target-order events | 0 |

Base success by ordinary-target count:

| Team | 1 hit | 2 hits | 3 hits | 4 hits |
|---|---:|---:|---:|---:|
| Yellow | 0 / 0 | 29 / 63 = 46.03% | 4 / 5 = 80.00% | 0 / 0 |
| Blue | 0 / 0 | 21 / 60 = 35.00% | 6 / 9 = 66.67% | 0 / 0 |

Ordinary-target count distribution:

| Team | 1 target | 2 targets | 3 targets | 4 targets |
|---|---:|---:|---:|---:|
| Yellow | 0.00% | 92.19% | 7.81% | 0.00% |
| Blue | 6.25% | 79.69% | 14.06% | 0.00% |

Pushable-box metrics:

| Metric | box_ne | box_sw |
|---|---:|---:|
| Mean final displacement | 0.5576 m | 0.5327 m |
| Mean max displacement | 0.5576 m | 0.5327 m |

Contact, drift and safety metrics:

| Metric | Yellow | Blue | Total/Notes |
|---|---:|---:|---|
| Push events / episode | 13.6094 | 12.8906 | persistent box poses in trace |
| Robot contacts / episode | - | - | 0.0625 |
| Relocalization events / episode | 0.0156 | 2.6250 | blue still too high |
| Abnormal spin steps / episode | 4.3594 | 4.3594 | symmetric residual issue |
| Static penetrations | - | - | 0 |
| Box penetrations | - | - | 0 |
| Repeat target-order events | - | - | 0 |

## Strict Replay

Strict replay files:

- Summary: `isaaclab_sim/output/replay/mappo_dual_yellow_blue_experts_seed260506_rs004_strict8/strict_replay_summary.json`
- Trace: `isaaclab_sim/output/replay/mappo_dual_yellow_blue_experts_seed260506_rs004_strict8/strict_replay_trace.csv`
- Events: `isaaclab_sim/output/replay/mappo_dual_yellow_blue_experts_seed260506_rs004_strict8/strict_replay_events.jsonl`
- Report: `docs/rl_dual_yellow_blue_experts_strict8.md`
- Lightweight overview video: `docs/media/dual_yellow_blue_experts_strict_episode0_overview.mp4`
- IsaacLab overview video: `docs/media/isaaclab_dual_experts_overview.mp4`
- IsaacLab yellow POV video: `docs/media/isaaclab_dual_experts_yellow_pov.mp4`
- IsaacLab blue POV video: `docs/media/isaaclab_dual_experts_blue_pov.mp4`

Strict replay result:

- Episodes: 8
- Hard violations: 0
- Warnings: 0
- Own-target penalties: 0.0 / episode
- Robot contacts: 0.0 / episode
- Base wins: 0.875 / episode

IsaacLab three-view video check:

| File | View | Resolution | Frames | Duration | Size |
|---|---|---:|---:|---:|---:|
| `docs/media/isaaclab_dual_experts_overview.mp4` | overview | 1280x720 | 337 | 28.08 s | 4.07 MB |
| `docs/media/isaaclab_dual_experts_yellow_pov.mp4` | yellow POV | 1280x720 | 337 | 28.08 s | 4.42 MB |
| `docs/media/isaaclab_dual_experts_blue_pov.mp4` | blue POV | 1280x720 | 337 | 28.08 s | 4.37 MB |

Video replay conclusions:

- The three videos were rendered from the same strict episode (`episode=0`, trace duration `26.2 s`) so the tactical events line up across views.
- IsaacLab render logs show `RandomObstacleSouthWest` being pushed across multiple new coordinates, confirming persistent box motion in the replay instead of a one-frame contact reset.
- Replay events show armor panels are removed before the base hit: yellow removes two blue armor panels, blue removes two yellow armor panels, then yellow legally wins on `BlueBaseTarget`.
- Strict replay checks report zero hard violations, zero warnings, zero own-target penalties, zero robot contacts and zero blocked/penetration steps for the audited eight-episode batch.
- Frame sampling with OpenCV confirmed all three MP4s are readable, non-empty and non-blank.

## Remaining Risks

- Blue win rate is 42.19%, slightly below the 45%-55% target band; it is close but not fully solved.
- The 64-episode run still has blue relocalization events at 2.625 / episode and symmetric abnormal-spin steps at 4.3594 / episode.
- The model was only short-continued for 16k steps because another CUDA job slowed training. The three-view export is complete, but a longer dedicated-GPU continuation is still recommended before claiming final competition-level balance.

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
