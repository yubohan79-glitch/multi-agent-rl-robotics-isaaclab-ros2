# Large-Scale Curriculum Plan: 5v5 -> 10v10 -> 25v25 -> 50v50

The previous direct 50v50 population-search run produced the rule-scoring loop, but it did not learn a sufficiently structured team strategy. The most important warning signs were:

- the selected final policy fell back to the baseline theta;
- probe games became one-sided late in training;
- robot contact counts remained high;
- the video showed enough base assault to pass a benchmark smoke test, but not enough coordinated tactics.

The new plan uses a staged curriculum. Each stage preserves the same rule loop:

```text
zone control -> shield opening -> base assault -> score/win closure
```

The curriculum changes only scale and difficulty. It does not relax the final 50v50 rule contract.

## Stage Schedule

| Stage | Team Size | Purpose | Training Budget | Eval Games |
| --- | ---: | --- | ---: | ---: |
| stage01_05v05 | 5v5 | learn lane assignment, first shield opening and base attack | 80 generations, population 14, 4 episodes/candidate | 128 |
| stage02_10v10 | 10v10 | add tactical interference and small-squad spreading | 90 generations, population 16, 3 episodes/candidate | 160 |
| stage03_25v25 | 25v25 | learn multi-lane crowd control and contact reduction | 110 generations, population 18, 2 episodes/candidate | 192 |
| stage04_50v50 | 50v50 | final benchmark scale | 150 generations, population 20, 2 episodes/candidate | 256 |

## Difficulty Schedule

| Stage | Base HP | Shield Progress | Capture Rate |
| --- | ---: | ---: | ---: |
| 5v5 | 8 | 1.2 | 0.105 |
| 10v10 | 14 | 2.2 | 0.090 |
| 25v25 | 28 | 5.0 | 0.070 |
| 50v50 | 45 | 8.0 | 0.060 |

Lower stages intentionally make shield opening and base assault easier so the policy first learns the task sequence. The final 50v50 stage returns to the published benchmark difficulty.

## Promotion Gates

Each stage records:

- yellow and blue win rate;
- draw rate;
- yellow and blue base damage;
- yellow and blue shield-open rate;
- mean and p95 robot contacts;
- obstacle contacts;
- checkpoint path.

A stage is considered promotable when:

- both sides have nonzero, nontrivial win rate;
- both sides produce base damage;
- both shield-open rates are nonzero;
- obstacle contacts are zero;
- p95 robot contacts are below the stage-specific contact cap.

If a stage fails, its checkpoint is not considered a reliable final result. The script can either continue for diagnosis or stop immediately with `--stop-on-failure`.

## Command

```powershell
& "C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe" `
  isaaclab_sim\rl\run_large_scale_curriculum.py `
  --seed 607050
```

Outputs:

```text
docs/rl_data/large_scale_curriculum/
docs/rl_data/large_scale_50v50/
docs/media/large_scale_50v50_replay.mp4
docs/media/large_scale_50v50_replay.gif
docs/figures/large_scale_50v50/
```

The final IsaacLab MP4/GIF should be regenerated only after the final 50v50 stage passes inspection.

## Accepted Run

The accepted run used seed `607051` and passed every stage:

| Stage | Yellow Win | Blue Win | Draw | Yellow Base Damage | Blue Base Damage | Robot Contacts Mean/P95 | Obstacle Contacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage01_05v05 | 43.75% | 35.16% | 21.09% | 8.32 | 8.38 | 0.00 / 0.00 | 0.00 |
| stage02_10v10 | 40.00% | 44.38% | 15.62% | 13.74 | 13.87 | 0.00 / 0.00 | 0.00 |
| stage03_25v25 | 40.63% | 44.27% | 15.10% | 28.12 | 28.20 | 0.00 / 0.00 | 0.00 |
| stage04_50v50 | 36.72% | 42.19% | 21.09% | 44.90 | 44.89 | 0.00 / 0.00 | 0.00 |

The 5v5 failure found before this accepted run was caused by a role-assignment bug: smaller teams were all assigned to squad `0`, so no vehicle could enter the base-assault role. The current implementation scales five tactical roles across any team size, removes the old one-sided blue damage multiplier, and stops immediately if any stage fails.
