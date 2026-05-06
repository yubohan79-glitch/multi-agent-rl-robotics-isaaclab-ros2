# Domain-Randomized Shielded Recessed-Base Data

This directory archives the small data files used by the previous domain-randomized, shielded recessed-base README metrics and three-view IsaacLab replay notes. Runtime outputs under `isaaclab_sim/output/` are ignored by Git, so these copies keep the archived experiment tables reproducible.

The latest contact-hull dual-expert closure is documented in `docs/rl_dual_experts_contact_hull_seed260507_report.md`.

Source checkpoint:

`isaaclab_sim/output/rl/mappo_drshield_recessed_base_shared_gpu_seed419/policy.pt`

Training summary:

- CUDA device: NVIDIA GeForce RTX 4090
- Torch: 2.11.0+cu128
- Timesteps: 220,000 requested / 221,184 collected
- Parallel environments: 24
- Actor mode: shared
- Policy mode: residual expert
- Observation dim: 46
- Central critic dim: 92
- Domain randomization: enabled
- Action shield: enabled
- Wall time: 905.130 s

Evaluation summary:

| Seed | Episodes | Yellow win | Blue win | Draw/timeout | Base wins/episode | Own-target penalties |
|---:|---:|---:|---:|---:|---:|---:|
| 3100 | 64 | 50.00% | 48.44% | 1.56% | 0.9844 | 0.0 |
| 3200 | 64 | 50.00% | 48.44% | 1.56% | 0.9844 | 0.0 |
| Combined | 128 | 50.00% | 48.44% | 1.56% | 0.9844 | 0.0 |

Strict replay audit:

- Episodes: 16
- Yellow win rate: 62.50%
- Blue win rate: 37.50%
- Hard violations: 0
- Warnings: 0
- Own-target penalties/episode: 0.0
- Base wins/episode: 1.0

The strict audit is a legality sample; side balance is reported from the combined stochastic evaluation.

Media generated from strict episode 1:

| File | Duration | FPS | Resolution |
|---|---:|---:|---:|
| `docs/media/final_training_replay_overview.mp4` | 28.04 s | 24 | 1280x720 |
| `docs/media/final_training_replay_yellow_pov.mp4` | 28.04 s | 24 | 1280x720 |
| `docs/media/final_training_replay_blue_pov.mp4` | 28.04 s | 24 | 1280x720 |

Selected replay motion:

- `box_ne` moves about 0.164 m.
- `box_sw` moves about 0.123 m.
- Both robots start at `t=0`, attack opponent-side targets only, and finish with a legal base-target win.

Archived files:

- `training_curve.csv`
- `training_summary.json`
- `eval_seed3100_stochastic.json`
- `eval_seed3200_stochastic.json`
- `strict_replay_summary.json`
