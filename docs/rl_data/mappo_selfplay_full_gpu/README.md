# MAPPO Self-Play Full GPU Run Data

This directory contains the Git-tracked data copy for the earlier completed CUDA MAPPO/self-play run. The runtime output directory `isaaclab_sim/output/` is ignored by Git, so these files are archived here for paper figures, README figures and portfolio review.

For the latest rule-accurate dual-expert contact-hull run, use:

`docs/rl_dual_experts_contact_hull_seed260507_report.md`

## Files

| File | Content |
| --- | --- |
| `training_curve.csv` | one row per MAPPO update, used for training-curve plots |
| `training_curve.jsonl` | JSONL copy of the same update-level training diagnostics |
| `training_summary.json` | run configuration, GPU metadata, final metrics and policy path |
| `mappo_full_gpu_eval.json` | deterministic policy evaluation over 64 episodes |
| `mappo_full_gpu_eval_stochastic.json` | stochastic policy evaluation over 64 episodes |
| `scripted_rules_baseline_eval.json` | scripted baseline evaluation over 64 episodes |

## Figure Regeneration

```bash
python isaaclab_sim/rl/generate_rl_figures.py
```

Generated SVGs are written to `docs/figures/rl/`.

This archived run is still useful for trend figures and baseline comparison, but it predates the final recessed base target geometry, 0.80 s dwell-fire audit and dynamic pushable-box replay.
