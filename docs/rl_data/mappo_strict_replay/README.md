# Strict MAPPO Replay Data

This directory contains the Git-tracked summary and event log from the earlier strict post-training replay audit.

For the latest dual-expert contact-hull strict audit, use:

`docs/rl_dual_experts_contact_hull_seed260507_strict8.md`

The latest closure report is:

`docs/rl_dual_experts_contact_hull_seed260507_report.md`

## Files

| File | Content |
| --- | --- |
| `strict_replay_summary.json` | 32-episode replay summary, per-episode outcomes, violations and warnings |
| `strict_replay_events.jsonl` | event-level log for target hits, base wins, contacts and notable rule events |

The full step-by-step CSV trace for this archived run is generated under `isaaclab_sim/output/replay/mappo_strict_replay_full/strict_replay_trace.csv` and is intentionally not committed because it is a large runtime artifact.

## Replay Command

```bash
python3 isaaclab_sim/rl/replay_mappo_policy_strict.py \
  --checkpoint isaaclab_sim/output/rl/mappo_selfplay_full_gpu/policy.pt \
  --episodes 32 \
  --seed 901 \
  --device cpu \
  --stochastic \
  --output-dir ../output/replay/mappo_strict_replay_full \
  --report ../../docs/rl_strict_replay_audit.md
```

This archived run predates the final smaller recessed base targets, ground-touching armor blockers and dynamic pushable-box trace used by the current MP4s.
