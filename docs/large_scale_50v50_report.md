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
- Generations: 150.
- Population: 20.
- Candidate episodes: 2.
- Total training episodes sampled: 12000.
- Best fitness: 368.7113.
- Wall time: 2561.32 s.

## Evaluation

- Episodes: 256.
- Yellow win rate: 36.72%.
- Blue win rate: 42.19%.
- Draw rate: 21.09%.
- Mean yellow score: 227.67.
- Mean blue score: 227.55.
- Mean yellow survivors: 48.71 / 50.
- Mean blue survivors: 48.77 / 50.
- Mean yellow base damage: 44.90.
- Mean blue base damage: 44.89.
- Mean yellow base open rate: 18.37%.
- Mean blue base open rate: 18.39%.
- Mean robot contacts: 0.00.
- P95 robot contacts: 0.00.
- Mean obstacle contacts: 0.00.
- Mean final zone state: [-1.0, 0.0011, 1.0].

## Artifacts

- Checkpoint: `docs/rl_data/large_scale_50v50/policy_checkpoint.json`
- Training curve: `docs/rl_data/large_scale_50v50/training_curve.csv`
- Evaluation JSON: `docs/rl_data/large_scale_50v50/eval_summary.json`
- Evaluation CSV: `docs/rl_data/large_scale_50v50/eval_episodes.csv`
- Rule-level preview MP4: `docs/media/large_scale_50v50_replay.mp4`
- Rule-level preview GIF: `docs/media/large_scale_50v50_replay.gif`
- IsaacLab tactical replay MP4: `docs/media/large_scale_50v50_isaaclab_replay.mp4`
- IsaacLab tactical replay GIF: `docs/media/large_scale_50v50_isaaclab_replay.gif`
- Figures: `docs/figures/large_scale_50v50/`

## Boundary

This benchmark validates scalable rule-level 50v50 mechanics and a trained swarm policy baseline. It does not claim IsaacLab rigid-body validation for all 100 robots and does not claim real-robot deployment. Those require a separate physics scaling and Sim2Real evidence package.
