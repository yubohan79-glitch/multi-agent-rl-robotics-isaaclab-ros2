# Rule-Accurate RL Update: 45-Degree Targets, Dwell Fire, and Pushable Obstacles

Date: 2026-05-04

## Why This Update Was Needed

The previous replay was too optimistic for sim2real: some targets were rendered too close to the wall, target knockdown could happen at unrealistic range, and the robot could pass through pushable obstacles in the simplified replay logic. This update makes the rule environment stricter before continuing RL training.

## Rule Notes Checked From Local Competition Files

- The arena contains 8 normal targets, 2 base targets, and two 0.3 m cube obstacles.
- Normal target AprilTag ID is `1`, yellow base target ID is `2`, and blue base target ID is `3`.
- Elimination matches require each robot to enter the opponent field and attack opponent targets only.
- A normal target gives 5 points and removes one opponent base armor plate in order.
- The first robot to knock down the opponent base target wins; knocking down the own base target loses.
- The official target-making document describes a 650 nm red point laser trigger pattern of 100 ms on, 100 ms off, then 100 ms on. For sim2real conservatism, this project now models a stricter continuous aiming dwell of `0.8 s`.

## Implemented Fixes

- Normal targets are inset to `1.36 m` from arena center and angled about 45 degrees instead of being visually embedded in the wall.
- Base targets keep diagonal 45-degree placement near their base corner.
- Valid shooter-outlet range is `0.05 m` to `0.50 m`.
- Laser firing requires the policy `fire_gate > 0.55`, opponent-target safety, short range, line of sight, yaw alignment, and `0.8 s` dwell.
- The robot now holds its firing pose once legal shooting geometry is reached, instead of driving past the target before dwell completes.
- Target contact no longer knocks targets down; it is penalized and audited.
- Static walls, fences, armor, and targets remain non-pushable blockers.
- 0.3 m boxes are pushable rigid obstacles in the rule environment; if a box is jammed, the robot is blocked rather than allowed to pass through.
- IsaacLab demo-flow timed target knockdowns were disabled so target falls only come from the same range/line/dwell rule gate.
- The strict replay renderer was updated to use the 45-degree, inset target layout.

## Training Run

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config configs/mappo_selfplay.yaml \
  --timesteps 500000 \
  --num-envs 32 \
  --rollout-steps 256 \
  --device cuda \
  --policy-mode residual_expert \
  --residual-scale 0.22 \
  --residual-l2-coef 0.004 \
  --actor-mode dual \
  --output ../output/rl/mappo_rule45_dwell_push_residual_gpu
```

Final checkpoint:

```text
isaaclab_sim/output/rl/mappo_rule45_dwell_push_residual_gpu/policy.pt
```

Balanced deployment checkpoint:

```text
isaaclab_sim/output/rl/mappo_rule45_dwell_push_residual_balanced/policy.pt
```

## Evaluation Summary

Expert baseline, 128 episodes:

- yellow win rate: `0.438`
- blue win rate: `0.461`
- draw/timeout rate: `0.102`
- normal hits per episode: `4.0`
- own-target penalties: `0.0`

Balanced residual MAPPO, stochastic 128 episodes:

- yellow win rate: `0.4688`
- blue win rate: `0.4531`
- draw/timeout rate: `0.0781`
- normal hits per episode: `4.0`
- base hit wins per episode: `0.9219`
- own-target penalties: `0.0`

Strict audit, stochastic 64 episodes:

- verdict: `PASS`
- hard violations: `0`
- warnings: `0`
- target contact events per episode: `0.0`
- robot contacts per episode: `0.0`
- base wins per episode: `0.9531`

## Replay Artifacts

- Strict audit report: `docs/rl_rule45_dwell_push_strict_audit.md`
- Strict replay trace: `isaaclab_sim/output/replay/mappo_rule45_dwell_push_balanced_strict64/strict_replay_trace.csv`
- Strict replay MP4: `docs/media/rule45_dwell_push_strict_replay_episode0.mp4`

## Current Recommendation

Use the balanced residual MAPPO policy with `policy_mode=residual_expert` and `residual_scale=0.08` for replay and evaluation. It keeps the learned tactical layer active while preserving the engineered safety gate, short-range dwell shooting, and no-own-target contract.

The deterministic policy still shows seed-dependent side bias, so final reported fairness should use multi-seed stochastic evaluation or paired color-swap evaluation.
