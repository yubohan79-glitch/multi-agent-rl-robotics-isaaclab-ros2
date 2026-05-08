# Large-Scale 50v50 Multi-Agent Battle Report

This report documents the first formal large-scale rule-level extension for the repository. It is not a replacement for the two-robot IsaacLab result; it is a new 100-agent benchmark contract used to study scalable multi-agent decision making before expensive full-physics replay.

## Scenario

- Two teams: yellow and blue.
- Agents per team: 50.
- Arena: 80 m x 50 m.
- Objectives: capture three middle control zones, open the enemy base shield, eliminate opponents, then damage the enemy base.
- Obstacles: three static cover/barrier regions.
- Safety metrics: robot contacts, obstacle contacts, shielded base shots, survivors and base health.

## Training

- Algorithm: population-based swarm flow policy search.
- Generations: 100.
- Population: 16.
- Candidate episodes: 2.
- Total training episodes sampled: 6400.
- Best fitness: 511.6296.
- Wall time: 1526.74 s.

## Evaluation

- Episodes: 256.
- Yellow win rate: 57.03%.
- Blue win rate: 42.97%.
- Draw rate: 0.00%.
- Mean yellow score: 215.63.
- Mean blue score: 150.11.
- Mean yellow survivors: 45.58 / 50.
- Mean blue survivors: 46.39 / 50.
- Mean yellow base damage: 41.72.
- Mean blue base damage: 27.74.
- Mean yellow base open rate: 19.77%.
- Mean blue base open rate: 43.42%.
- Mean robot contacts: 85.68.
- P95 robot contacts: 105.00.
- Mean obstacle contacts: 0.00.
- Mean final zone state: [-1.0, -0.8315, 1.0].

## Artifacts

- Checkpoint: `docs/rl_data/large_scale_50v50/policy_checkpoint.json`
- Training curve: `docs/rl_data/large_scale_50v50/training_curve.csv`
- Evaluation JSON: `docs/rl_data/large_scale_50v50/eval_summary.json`
- Evaluation CSV: `docs/rl_data/large_scale_50v50/eval_episodes.csv`
- IsaacLab replay MP4: `docs/media/large_scale_50v50_isaaclab_replay.mp4`
- IsaacLab replay GIF: `docs/media/large_scale_50v50_isaaclab_replay.gif`
- Figures: `docs/figures/large_scale_50v50/`

## Boundary

This benchmark validates scalable rule-level 50v50 mechanics and a trained swarm policy baseline. It does not claim IsaacLab rigid-body validation for all 100 robots and does not claim real-robot deployment. Those require a separate physics scaling and Sim2Real evidence package.
