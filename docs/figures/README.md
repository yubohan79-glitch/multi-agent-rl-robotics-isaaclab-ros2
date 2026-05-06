# Figure Assets

This directory keeps generated visual assets grouped by use:

- `portfolio/`: README and portfolio overview figures.
- `paper/`: paper-style method and system figures.
- `rl/`: reinforcement-learning architecture, Sim2Real figures and data-driven GPU MAPPO result plots.

SVG files are editable sources. PNG files are exported previews for GitHub rendering.

The archived full-GPU result figures in `rl/` are generated from `docs/rl_data/mappo_selfplay_full_gpu/` by `isaaclab_sim/rl/generate_rl_figures.py`.

The latest dual yellow/blue expert experiment data is under `isaaclab_sim/output/`, with the write-up in `docs/rl_dual_experts_contact_hull_seed260507_report.md`. Use it for current tables showing:

- 64 stochastic evaluation episodes: yellow 50.00%, blue 43.75%, draw/timeout 6.25%.
- 8 strict replay episodes: hard violations 0, warnings 0, own-target penalties 0, base wins/episode 1.000.
- Current IsaacLab media: `docs/media/isaaclab_contact_hull_*.mp4`.

The sensor-fusion architecture and evaluation SVGs in `rl/` remain editable assets; update their source labels if regenerating them against the final recessed-base data snapshot.
