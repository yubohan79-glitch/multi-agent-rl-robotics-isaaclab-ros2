# Large-Scale 50v50 Multi-Agent Battle Plan

This document defines the repository's large-scale adversarial simulation extension. The goal is to test whether the object-centric world-model / swarm-flow policy idea can organize 100 vehicles into readable team tactics before attempting expensive full rigid-body training for every vehicle.

The current accepted evidence package contains:

- a 50-vs-50 rule-scoring environment;
- formal population-based swarm-flow policy training;
- 256-game evaluation;
- a trace exported from the trained policy;
- an IsaacLab replay using 100 vehicle-shaped actors, bases, zones, walls and obstacles;
- GIF/MP4 media and figures for review.

This is not claimed as 100-robot real-hardware validation, and it is not claimed as full rigid-body IsaacLab reinforcement learning for all contacts. It is a reproducible large-scale benchmark and an IsaacLab visual replay of the accepted tactical trace.

## 1. Scenario Contract

Arena:

- size: `80 m x 50 m`;
- teams: yellow and blue;
- agents: `50` differential-drive vehicles per team;
- starting formations: each team starts near its own base in five tactical squads;
- bases: yellow base on the left, blue base on the right;
- control zones: three middle zones, one per lane;
- obstacles: three static cover/barrier regions that split the attack lanes.

Episode termination:

1. one base reaches zero HP;
2. one team loses all vehicles;
3. the maximum step budget is reached.

Winner selection:

1. the side that destroys the opponent base wins;
2. otherwise the side with the higher composite score wins;
3. exact score ties are draws.

Composite score includes:

- opponent vehicle eliminations;
- opponent base damage;
- accumulated control-zone ownership;
- surviving own vehicles;
- penalties for shielded base shots, obstacle contacts and excessive robot contacts.

## 2. Rule-Scoring Closure

The rule loop is intentionally closed: tactical behavior must move through the whole chain rather than exploiting a single reward term.

1. Vehicles contest the three control zones.
2. Zone advantage accumulates shield-opening progress.
3. A base remains protected until the opponent has opened the shield window.
4. Shielded base shots are logged but do not damage the base.
5. Once a shield window is open, attack squads move to legal base-assault positions.
6. Base damage reduces HP and can terminate the match.
7. Evaluation reports win rate, base damage, base HP, zone state, shield-open rates, contacts and survivors.

This prevents fake success such as shooting through a closed shield, ignoring zone control, or winning only by vehicle elimination while never attacking the base.

## 3. Policy Design

The current baseline uses a shared parameterized swarm-flow policy. Each vehicle receives a team-side transform and squad identity, so the same policy can express yellow and blue behavior while still producing different local actions.

Policy inputs:

- own position, health and cooldown;
- squad assignment;
- nearest enemy direction and distance;
- local squad centroid;
- zone ownership state;
- own and enemy base positions;
- obstacle repulsion;
- base shield status.

Policy actions are not individual neural actions. They are emergent from weighted flow fields:

- move toward control zones;
- move toward enemy base after the shield opens;
- engage nearby enemies;
- maintain local cohesion;
- keep separation within dense squads;
- flank by squad lane;
- retreat or defend when low-health or base pressure is high.

Optimized parameters:

- `zone_weight`
- `base_weight`
- `enemy_weight`
- `cohesion_weight`
- `separation_weight`
- `flank_bias_m`
- `defense_weight`
- `aggression`
- `spread_m`
- `retreat_health`

## 4. Training Plan

Current implemented training uses population-based swarm-flow policy search:

1. initialize a tactical baseline policy;
2. sample candidate parameter vectors;
3. evaluate each candidate from both yellow and blue sides against archived opponents;
4. score candidates with win/loss, score gap, base damage, shield-open rate, survival, contact penalties and shielded-shot penalties;
5. promote elites into a running archive;
6. perform a validation selection pass so a one-sided exploit is not selected as the final policy.

Formal training command used for the current artifact:

```powershell
& "C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe" isaaclab_sim\rl\large_scale_50v50_battle.py all `
  --generations 40 `
  --population 12 `
  --episodes-per-candidate 1 `
  --probe-episodes 4 `
  --selection-episodes 24 `
  --eval-episodes 128 `
  --max-steps 720 `
  --video-seconds 30 `
  --gif-seconds 12 `
  --fps 30 `
  --gif-fps 8 `
  --width 1920 `
  --height 1080 `
  --verbose
```

Additional stability evaluation:

```powershell
& "C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe" isaaclab_sim\rl\large_scale_50v50_battle.py eval `
  --episodes 256 `
  --max-steps 720 `
  --seed 508500
```

## 5. Evaluation Gate

The current promotion gate requires:

- at least 128 games, with 256 preferred for the public table;
- both teams must have nonzero win rate;
- draw rate should stay near zero unless the scenario intentionally studies stalemate;
- both teams must produce base damage;
- shield-open rates must be nonzero for both bases;
- obstacle contact must remain zero in the accepted benchmark;
- robot contacts must be reported, not hidden;
- replay video must show vehicles, lanes, zones, bases and HP changes clearly.

Current 256-game evaluation:

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
| Mean yellow base damage | 41.63 |
| Mean blue base damage | 24.71 |
| Mean yellow base open rate | 19.83% |
| Mean blue base open rate | 43.02% |
| Mean robot contacts | 83.88 |
| P95 robot contacts | 102.00 |
| Mean obstacle contacts | 0.00 |

## 6. IsaacLab Replay Plan

The accepted trace is exported to:

```text
docs/rl_data/large_scale_50v50/isaaclab_replay_trace.npz
```

The IsaacLab replay script loads that trace and creates:

- 100 vehicle-shaped actors;
- vehicle noses indicating heading;
- yellow and blue bases;
- three control zones;
- static barriers;
- three tactical lanes;
- a side panel with time, HP, alive count, zone ownership and shield-open status.

Replay command:

```powershell
& "C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe" isaaclab_sim\large_scale_50v50_isaaclab_replay.py `
  --headless `
  --device cpu `
  --duration 30 `
  --record_fps 30 `
  --record_width 1920 `
  --record_height 1080 `
  --record_video docs/media/large_scale_50v50_isaaclab_replay.mp4
```

Published replay artifacts:

```text
docs/media/large_scale_50v50_isaaclab_replay.mp4
docs/media/large_scale_50v50_isaaclab_replay.gif
```

## 7. Figures

Published figure artifacts:

```text
docs/figures/large_scale_50v50/large_scale_50v50_rule_layout.png
docs/figures/large_scale_50v50/large_scale_50v50_rule_closure.png
docs/figures/large_scale_50v50/large_scale_50v50_training.png
docs/figures/large_scale_50v50/large_scale_50v50_eval.png
```

## 8. Next Research Step

The next strict milestone is full IsaacLab physics scaling:

1. convert the 100 visual actors into rigid differential-drive assets;
2. add contact sensors and per-vehicle collision logs;
3. validate FPS and memory use under 100 vehicles;
4. train a smaller physical curriculum first, such as 5v5 -> 10v10 -> 25v25 -> 50v50;
5. rerun the same evaluation table and video QA without changing the success definition.

Until that milestone is complete, this repository should describe the current result as a trained large-scale rule-level benchmark with IsaacLab tactical replay evidence, not as completed 100-robot full-physics RL.
