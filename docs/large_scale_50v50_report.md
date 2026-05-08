# Large-Scale 50v50 Multi-Agent Battle Report

This report records the accepted 50-vs-50 benchmark run. The requirement was that the strategy produce a complete rule-scoring loop and visible base offense/defense, and that the replay be tactically readable rather than a cloud of small points.

## Status

Accepted as a large-scale rule-level training benchmark with IsaacLab tactical replay evidence.

Not claimed as:

- 100-robot real-hardware validation;
- full rigid-body IsaacLab reinforcement learning for all 100 vehicles;
- distributed multi-node training.

## Implemented Rules

- Two teams: yellow and blue.
- Vehicles per team: 50.
- Arena: `80 m x 50 m`.
- Objective chain: contest zones -> open base shield -> assault enemy base -> destroy base or win by score.
- Combat: line-of-sight fire, range limits, cooldowns and hit probability.
- Base protection: shielded base shots are counted but cannot damage HP.
- Obstacles: three static barrier regions produce obstacle-contact metrics.
- Collision telemetry: robot contacts are counted and published.

## Training

| Item | Value |
| --- | ---: |
| Algorithm | population-based swarm-flow policy search |
| Generations | 40 |
| Population | 12 |
| Candidate episodes | 1 |
| Probe episodes | 4 |
| Selection episodes per candidate | 24 |
| Total training episodes sampled | 960 |
| Best training fitness | 509.5003 |
| Wall time | 296.44 s |

Checkpoint:

```text
docs/rl_data/large_scale_50v50/policy_checkpoint.json
```

## Evaluation

Formal stability evaluation:

| Metric | Value |
| --- | ---: |
| Episodes | 256 |
| Yellow win rate | 59.77% |
| Blue win rate | 40.23% |
| Draw rate | 0.00% |
| Mean elapsed time | 33.19 s |
| Mean yellow score | 215.20 |
| Mean blue score | 134.25 |
| Mean yellow survivors | 45.60 / 50 |
| Mean blue survivors | 46.33 / 50 |
| Mean yellow base HP | 21.14 |
| Mean blue base HP | 3.91 |
| Mean yellow base damage | 41.63 |
| Mean blue base damage | 24.71 |
| Mean yellow base open rate | 19.83% |
| Mean blue base open rate | 43.02% |
| Mean robot contacts | 83.88 |
| P95 robot contacts | 102.00 |
| Mean obstacle contacts | 0.00 |

Interpretation:

- The policy does create base-assault behavior on both sides.
- Both bases are sometimes opened; base damage is nonzero for both teams.
- Yellow has an advantage but blue remains above 40% win rate, so the baseline is usable for a first public large-scale benchmark.
- Robot contacts are high, which is expected in this dense battle setting and is published as a remaining optimization target.
- Obstacle-contact count is zero in the accepted run.

Evaluation files:

```text
docs/rl_data/large_scale_50v50/eval_summary.json
docs/rl_data/large_scale_50v50/eval_episodes.csv
docs/rl_data/large_scale_50v50/training_summary.json
docs/rl_data/large_scale_50v50/training_curve.csv
docs/rl_data/large_scale_50v50/policy_selection.csv
```

## IsaacLab Replay

The accepted policy trace was exported and replayed in IsaacLab with 100 vehicle-shaped actors, visible headings, bases, zones, barriers, tactical lanes and a side telemetry panel.

Trace:

```text
docs/rl_data/large_scale_50v50/isaaclab_replay_trace.npz
```

Replay script:

```text
isaaclab_sim/large_scale_50v50_isaaclab_replay.py
```

Replay media:

```text
docs/media/large_scale_50v50_isaaclab_replay.mp4
docs/media/large_scale_50v50_isaaclab_replay.gif
```

Video QA result:

- MP4: 1920x1080, 30 fps, 900 frames, 30 seconds.
- The replay shows team formations, lane movement, zone contesting, base HP changes and winner state.
- The vehicles are rendered as readable car-shaped actors with noses rather than tiny points.
- The side panel reports HP, alive counts, zone ownership and shield-open status.
- No blank or black-frame issue was detected in sampled frames.

## Figures

```text
docs/figures/large_scale_50v50/large_scale_50v50_rule_layout.png
docs/figures/large_scale_50v50/large_scale_50v50_rule_closure.png
docs/figures/large_scale_50v50/large_scale_50v50_training.png
docs/figures/large_scale_50v50/large_scale_50v50_eval.png
```

## Remaining Risks

- The current 50v50 training is rule-level, not full rigid-body IsaacLab RL.
- Robot contact rate is high and should be reduced in the next curriculum.
- The baseline is slightly yellow-favored; future work should tune symmetry, lane assignment and base-damage scaling.
- Full Sim2Real claims require hardware logs, calibration records and real-run statistics that are not part of this benchmark.
