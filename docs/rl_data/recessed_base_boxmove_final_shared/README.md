# Archived Recessed-Base Moving-Box Shared-Actor Data

This directory archives the small data files from the previous recessed-base moving-box shared-actor run. Runtime outputs under `isaaclab_sim/output/` are ignored by Git, so these copies keep the archived experiment tables reproducible.

For the current dual-expert contact-hull final run, use:

`docs/rl_dual_experts_contact_hull_seed260507_report.md`

Source checkpoint:

`isaaclab_sim/output/rl/mappo_recessed_base_boxmove_final_shared_gpu_seed371/policy.pt`

Training summary:

- CUDA device: NVIDIA GeForce RTX 4090
- Torch: 2.11.0+cu128
- Timesteps: 220,000 requested / 221,184 collected
- Parallel environments: 24
- Actor mode: shared
- Policy mode: residual expert
- Observation dim: 46
- Central critic dim: 92
- Wall time: 679.051 s

Evaluation summary:

| Seed | Episodes | Yellow win | Blue win | Draw/timeout | Base wins/episode | Own-target penalties |
|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 64 | 46.88% | 53.12% | 0.00% | 1.0000 | 0.0 |
| 2600 | 64 | 45.31% | 51.56% | 3.12% | 0.9688 | 0.0 |
| Combined | 128 | 46.09% | 52.34% | 1.56% | 0.9844 | 0.0 |

Strict replay audit:

- Episodes: 16
- Yellow win rate: 50.00%
- Blue win rate: 50.00%
- Hard violations: 0
- Warnings: 0
- Own-target penalties/episode: 0.0
- Base wins/episode: 1.0

Media in `docs/media/final_training_replay_*.mp4` was regenerated from the later `drshield_recessed_base_shared` strict trace. The table below describes this archived run's original episode 8 media metadata:

| File | Duration | FPS | Resolution |
|---|---:|---:|---:|
| `docs/media/final_training_replay_overview.mp4` | 23.04 s | 24 | 1280x720 |
| `docs/media/final_training_replay_yellow_pov.mp4` | 23.04 s | 24 | 1280x720 |
| `docs/media/final_training_replay_blue_pov.mp4` | 23.04 s | 24 | 1280x720 |

Selected replay motion:

- `box_ne` moves about 0.165 m.
- `box_sw` moves about 0.122 m.
- Both robots start at `t=0`, attack opponent-side targets only, and finish with a legal base-target win.

Archived files:

- `training_curve.csv`
- `training_summary.json`
- `eval_seed2500_stochastic.json`
- `eval_seed2600_stochastic.json`
- `strict_replay_summary.json`
