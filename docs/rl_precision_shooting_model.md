# Precision Shooting and Pushable-Obstacle Training Update

This note records the latest rule-model update for the RoboCup VisionRL self-play environment and IsaacLab replay.

## Updated Shooting Contract

The laser rule now uses the shooter outlet as the distance origin instead of the robot center:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `SHOOT_MIN_RANGE` | `0.05 m` | shots closer than 5 cm from the shooter outlet are rejected |
| `SHOOT_RANGE` | `0.50 m` | shots beyond 50 cm from the shooter outlet are rejected |
| `SHOOT_IDEAL_DISTANCE` | `0.30 m` | nominal precision/throughput balance point |
| `SHOOTER_FORWARD_OFFSET` | `0.20 m` | fixed laser module offset from `base_link` |

The rule gate now checks:

- shooter outlet to target projection is between `0.05 m` and `0.50 m`
- lateral error is within the normal/base target hit radius
- the line of fire is not blocked by walls, armor, fences, or moved pushable boxes
- robot center remains outside the target contact clearance, so contact does not become a scoring shortcut

## Accuracy and Time Tradeoff

Shot probability is geometry-based:

- closer outlet-to-target distance gives higher hit probability
- centered shots give higher hit probability
- base targets are slightly harder than normal targets
- very close shooting can cost more time because the policy must drive deeper into a valid standoff pose

The MAPPO reward uses this estimate in two places:

- rewards high-confidence fire-ready poses
- adds a small close-range time cost so the learned policy can trade speed against precision

## Collision and Target Contact

Target contact no longer knocks down a target. If a robot brushes a target:

- the target remains standing
- localization confidence drops
- the robot receives a penalty
- recovery behavior can trigger a spin-scan relocalization branch

This keeps scoring tied to legal laser hits instead of accidental target contact.

## Pushable vs Static Objects

The two 0.3 m cube obstacles are modeled as pushable boxes. Fences, armor plates, walls and targets remain static blockers. The policy observation includes pushable obstacle vectors, and the high-risk route can learn when pushing a box is useful after or around a shooting action.

## Verification

Validated locally:

```text
python -m py_compile isaaclab_sim/rl/robocup_visionrl_gym_env.py \
  isaaclab_sim/rl/robocup_visionrl_selfplay_env.py \
  isaaclab_sim/robocup_visionrl_arena_sim.py

python3 -m pytest tests/test_rl_env_smoke.py tests/test_rl_strategy_contract.py -q -p no:cacheprovider
14 passed
```

CUDA smoke training also ran successfully with `torch.cuda.is_available() == True`.

## Current Full Training Run

The full precision run is running under:

```text
isaaclab_sim/output/rl/mappo_precision_005_050_outlet_gpu/
```

Command:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config isaaclab_sim/rl/configs/mappo_selfplay.yaml \
  --timesteps 1000000 \
  --num-envs 32 \
  --rollout-steps 256 \
  --device cuda \
  --output ../output/rl/mappo_precision_005_050_outlet_gpu
```

After completion, evaluate with `evaluate_mappo_policy.py`, run strict replay audit, then regenerate the RL figures from the new run data.
