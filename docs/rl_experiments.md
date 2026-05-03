# Reinforcement-Learning Experiments

## Algorithms

| Algorithm | Role | Notes |
| --- | --- | --- |
| Scripted baseline | CI and rule regression | Deterministic, interpretable, not final strategy |
| PPO | Single-agent smoke training | Fast attack baseline for the yellow side |
| MAPPO self-play | Final high-level policy | CTDE training with local actor deployment |

## Reward Ablation Plan

| Ablation | Change | Expected observation |
| --- | --- | --- |
| No own-target penalty | Remove own-target negative reward | More unsafe blind-fire behavior |
| No recovery reward | Remove spin recovery reward | Longer stuck periods after collision |
| No collision context | Flat collision penalty | Less tactical blocking near opponent routes |
| No base-rush reward | Reduce terminal base reward | Over-clears ordinary targets and wastes time |

## Policy Export Contract

The deployable actor must output high-level tactical commands only:

- target selection
- route mode
- base-rush gate
- blocking gate
- recovery gate
- fire gate

The real robot keeps Nav2, AprilTag detection, EKF localization and shooter services in the control loop. This avoids deploying a simulator-only end-to-end velocity policy.

## Deterministic Replay

Use:

```bash
cd isaaclab_sim/rl
python evaluate_selfplay.py --episodes 32 --seed 11
```

The output JSON is written under `isaaclab_sim/output/`, which is ignored by Git.
