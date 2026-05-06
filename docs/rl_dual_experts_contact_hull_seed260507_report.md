# Dual-Expert Contact-Hull RL Closure Report

Verdict: **PASS with one training-risk note**

This report closes the current `mappo_dual_experts_contact_hull_seed260507` loop: target/physics audit, code fixes, continuation training, 64-episode evaluation, strict replay, IsaacLab three-view recording and video inspection.

## What Changed

- Target geometry remains rule-checked at 10 total targets: 8 normal targets plus 2 recessed base targets. Every target has a 45-degree target plane, front-face probe outside walls/armor, and no wall, armor or start-fence overlap.
- The pushable-box contact hull was enlarged in both the rule environment and strict replay. Robot-box separation now uses `ROBOT_PUSHABLE_CLEARANCE_RADIUS = ROBOT_RADIUS + 0.030`.
- IsaacLab replay uses an additional visual safety hull, `ROBOT_PUSHABLE_RENDER_CLEARANCE_RADIUS = ROBOT_PUSHABLE_CLEARANCE_RADIUS + 0.050`, so rendered robots do not visually pass through red pushable boxes.
- Dynamic red boxes are included in IsaacLab costmap recovery and push correction, so box poses persist after being pushed instead of snapping back.
- The yellow expert two-hit base gate was restored to allow a legal early-base attempt from the opened armor side when the geometric window is good.
- The IsaacLab recorder now supports a true `top` view in addition to `overview`, `yellow_pov` and `blue_pov`.
- First-person replay cameras were lifted and widened so pushing a box shows the box top and target window instead of filling the whole frame with the red box face.

## Why

The previous replay could look like robot-box penetration in IsaacLab even when the 2D rule trace was legal. The fix makes the rule collision radius, strict replay validator and visual renderer agree on a conservative clearance envelope. The target audit also addresses the repeated failure mode where a target could look correct in yaw but be placed too close to wall or armor geometry.

## Verification

Geometry audit:

```bash
PYTHONPATH=isaaclab_sim/rl python3 isaaclab_sim/rl/audit_rule_geometry.py
```

Result:

- target_count: `10`
- physics_checks: `3`
- failures: `0`
- report: `docs/rl_rule_geometry_audit.md`

Full rule/test suite:

```bash
PYTHONPATH=isaaclab_sim/rl python3 -m pytest tests -q
```

Result: `66 passed`.

Strict replay audit:

- report: `docs/rl_dual_experts_contact_hull_seed260507_strict8.md`
- JSON: `isaaclab_sim/output/replay/mappo_dual_experts_contact_hull_seed260507_strict8/strict_replay_summary.json`
- trace: `isaaclab_sim/output/replay/mappo_dual_experts_contact_hull_seed260507_strict8/strict_replay_trace.csv`
- events: `isaaclab_sim/output/replay/mappo_dual_experts_contact_hull_seed260507_strict8/strict_replay_events.jsonl`

Strict summary:

| Metric | Value |
| --- | ---: |
| Episodes | 8 |
| Yellow win rate | 37.50% |
| Blue win rate | 62.50% |
| Draw/timeout rate | 0.00% |
| Hard violations | 0 |
| Warnings | 0 |
| Own-target penalties/episode | 0.0 |
| Blocked steps/episode | 0.0 |
| Target contact events/episode | 0.0 |
| Robot contacts/episode | 5.5 |
| Recovery events/episode | 0.0 |
| Base wins/episode | 1.0 |

## Training

Checkpoint:

`isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507/policy.pt`

Training curve:

`isaaclab_sim/output/rl/mappo_dual_experts_contact_hull_seed260507/training_curve.csv`

Configuration:

| Field | Value |
| --- | ---: |
| seed | 260507 |
| timesteps | 16,384 |
| num_envs | 32 |
| rollout_steps | 64 |
| update_epochs | 2 |
| hidden_dim | 256 |
| actor_mode | dual |
| policy_mode | residual_expert |
| residual_scale | 0.04 |
| residual_l2_coef | 0.0018 |
| domain_randomization | true |
| action_shield | true |

The run is a continuation from the previous dual-expert recovery-cooldown policy, with the contact-hull and yellow two-hit base-gate fixes active.

## Evaluation

Evaluation files:

- JSON: `isaaclab_sim/output/eval/mappo_dual_experts_contact_hull_seed260507_eval64.json`
- CSV: `isaaclab_sim/output/eval/mappo_dual_experts_contact_hull_seed260507_eval64.csv`

64-episode stochastic evaluation:

| Metric | Value |
| --- | ---: |
| yellow_win_rate | 50.00% |
| blue_win_rate | 43.75% |
| draw_rate | 6.25% |
| mean_episode_time_s | 42.1969 |
| mean_yellow_score | 40.6250 |
| mean_blue_score | 36.7188 |
| mean_normal_hits_yellow | 2.1250 |
| mean_normal_hits_blue | 2.0938 |
| static_penetrations_total | 0 |
| box_penetrations_total | 0 |
| repeat_target_order_events_total | 0 |
| robot_contacts_per_episode | 5.0625 |

Normal-target count distribution:

| Team | 1 hit | 2 hits | 3 hits | 4 hits |
| --- | ---: | ---: | ---: | ---: |
| Yellow | 0.00% | 87.50% | 12.50% | 0.00% |
| Blue | 1.56% | 87.50% | 10.94% | 0.00% |

Base success by prior normal hits:

| Team | 1 hit | 2 hits | 3 hits | 4 hits |
| --- | ---: | ---: | ---: | ---: |
| Yellow | 0/0 | 27/64 = 42.19% | 5/7 = 71.43% | 0/0 |
| Blue | 0/0 | 27/62 = 43.55% | 1/7 = 14.29% | 0/0 |

Box and contact statistics:

| Metric | Value |
| --- | ---: |
| yellow push_events_per_episode | 12.2812 |
| blue push_events_per_episode | 12.4062 |
| mean_final_box_displacement_m box_ne | 0.5091 |
| mean_final_box_displacement_m box_sw | 0.5104 |
| yellow relocalization_events_per_episode | 1.5625 |
| blue relocalization_events_per_episode | 1.6094 |
| yellow abnormal_spin_steps_per_episode | 0.1094 |
| blue abnormal_spin_steps_per_episode | 19.5156 |

## IsaacLab Replay

Selected strict episode: `5` with seed `940005`. It ends in a legal yellow base win at `29.8 s`, with 4 normal-target hits, 1 base win, 0 hard violations and 0 warnings.

Three-view videos:

| File | View | Resolution | Frames | FPS | Size |
| --- | --- | ---: | ---: | ---: | ---: |
| `docs/media/isaaclab_contact_hull_top.mp4` | top | 1280x720 | 385 | 12 | 4.85 MB |
| `docs/media/isaaclab_contact_hull_yellow_pov.mp4` | yellow POV | 1280x720 | 385 | 12 | 4.43 MB |
| `docs/media/isaaclab_contact_hull_blue_pov.mp4` | blue POV | 1280x720 | 385 | 12 | 4.40 MB |

Checked frames:

- `docs/media/isaaclab_contact_hull_top_frame180.png`
- `docs/media/isaaclab_contact_hull_yellow_pov_frame180.png`
- `docs/media/isaaclab_contact_hull_blue_pov_frame180.png`

Video inspection conclusion:

- Targets are present, not embedded in walls, and normal target faces are at about 45 degrees to the adjacent wall/corner geometry.
- Recessed base targets remain behind ground-touching blue armor blockers until the corresponding armor route is opened.
- The red boxes move persistently in the top-view replay; the strict trace reports mean final displacement of about `0.51 m` for both boxes.
- Robots do not pass through static blockers, armor, targets or pushable boxes in the strict trace; strict penetration counters are zero.
- Both robots leave at `t=0` and attack opponent-owned targets only.
- Robot-robot contact is counted as a tactical event and does not directly trigger a collision recovery reset.
- No repeated invisible/failed target loop is reported in the 64-episode evaluation.

## Remaining Risks

- Blue's 64-episode win rate is `43.75%`, slightly below the requested 45%-55% band. The split is close, but another continuation pass can target blue recovery/spin behavior.
- Blue abnormal spin is still high at `19.5156` steps/episode, concentrated in longer or losing episodes. This is the main next tuning target.
- The 3-hit base-success sample is small: only 7 attempts per side. Yellow's 3-hit success is strong, while blue's is weak; this needs more episodes or a curriculum pass that forces more 3-hit/4-hit base attempts.
- The latest closure is IsaacLab/rule-env verified. No new live WSL2 ROS topic bag was collected in this closure report.

## Figures To Draw

- System architecture diagram.
- ROS2 + IsaacLab + RL closed-loop diagram.
- Multi-sensor fusion structure diagram.
- Win-rate bar chart.
- Training curve.
- Normal target-hit count distribution.
- Base success rate versus normal-target count.
- Box push displacement statistics.
- Three-view replay screenshot collage.
