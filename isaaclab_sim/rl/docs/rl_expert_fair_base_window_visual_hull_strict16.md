# Strict MAPPO Replay Audit

Verdict: **PASS**

This report replays the trained MAPPO tactical actor and audits each step against strict rule and physics invariants.

## Replay Setup

- checkpoint: `isaaclab_sim/output/rl/mappo_dual_experts_post_retry_bluetempo_rs005_seed261020/policy.pt`
- deterministic: `True`
- device: `cuda`
- episodes: `16`
- max step translation: `0.12 m`
- max step yaw delta: `0.27 rad`
- static blocker tolerance: `0.012 m`

## Strict Checks

- action shape is exactly 6D and bounded to [-1, 1]
- robot pose is finite, inside the arena boundary, and outside static blockers
- per-step translation/yaw changes stay within differential-drive limits
- selected targets and fired targets must belong to the opponent
- own-base hit/collision is an immediate hard violation
- scores and armor only change in rule-compatible directions
- target contact is recorded as a warning unless it actually knocks down a target

## Summary

| Metric | Value |
| --- | ---: |
| `episodes` | 16 |
| `yellow_win_rate` | 0.5625 |
| `blue_win_rate` | 0.4375 |
| `draw_or_timeout_rate` | 0.0 |
| `hard_violations` | 0 |
| `warnings` | 0 |
| `normal_hits_per_episode` | 4.4375 |
| `base_wins_per_episode` | 1.0 |
| `own_target_penalties_per_episode` | 0.0 |
| `blocked_steps_per_episode` | 0.0 |
| `target_contact_events_per_episode` | 0.0 |
| `robot_contacts_per_episode` | 0.0 |
| `recovery_events_per_episode` | 0.0 |
| `block_steps_per_episode` | 0.0 |
| `base_rush_steps_per_episode` | 171.5 |
| `wall_time_s` | 81.483 |

## Output Files

- JSON summary: `isaaclab_sim/rl/isaaclab_sim/output/replay/expert_fair_base_window_visual_hull_strict16/strict_replay_summary.json`
- CSV trace: `isaaclab_sim/rl/isaaclab_sim/output/replay/expert_fair_base_window_visual_hull_strict16/strict_replay_trace.csv`
- JSONL event log: `isaaclab_sim/rl/isaaclab_sim/output/replay/expert_fair_base_window_visual_hull_strict16/strict_replay_events.jsonl`

## Notes

Blocked steps are not counted as hard violations because the costmap/barrier logic prevented penetration. Actual penetration after integration is a hard violation.
A pose that only touches the inflated costmap boundary within the static-blocker tolerance is counted as a warning, not as physical wall penetration.
Pushable-box contact is allowed only within the tolerance; robot-box penetration is a hard violation.
Target contact is allowed only as a non-scoring brush/contact event; any contact-induced knockdown remains a hard violation.
Robot-robot contact is allowed as a tactical event, but it is counted so future training can penalize unsafe or wasteful contact.
