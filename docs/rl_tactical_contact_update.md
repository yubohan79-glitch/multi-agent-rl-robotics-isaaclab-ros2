# Tactical Contact and Target-Retry Update

Date: 2026-05-04

## Purpose

This update separates legal robot-robot tactical contact from localization failure. The high-level policy can now choose blocking or contact when it improves win probability, while static collisions with walls, armor, targets, and jammed obstacles still remain costly and can trigger recovery.

## Implemented Changes

- Robot-robot contact no longer reduces localization confidence.
- Robot-robot contact no longer enters `RECOVER_LOCALIZATION`.
- Both robots keep explicit opponent pose features in the observation:
  - relative position
  - relative bearing
  - relative heading
  - visibility
  - threat to own base
- `block_interference_gate` and `risk_preference` can express an active collision/blocking tactic.
- Tactical contact receives positive reward only when it blocks a meaningful opponent threat and is not near own base or own targets.
- Contact near own critical assets remains penalized.
- Failed target attempts now enter a cooldown so the robot does not repeatedly attack the same blocked or invalid target.
- IsaacLab replay controllers also track per-target failure counts and skip targets after repeated withheld shots.

## Training Command

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py \
  --config configs/mappo_selfplay.yaml \
  --timesteps 250000 \
  --num-envs 32 \
  --rollout-steps 256 \
  --device cuda \
  --policy-mode residual_expert \
  --residual-scale 0.55 \
  --residual-l2-coef 0.002 \
  --actor-mode dual \
  --output ../output/rl/mappo_tactical_contact_open_gate_gpu
```

## Evaluation Snapshot

64-episode stochastic evaluation:

- yellow win rate: `0.4219`
- blue win rate: `0.5469`
- draw/timeout rate: `0.0312`
- normal hits per episode: `3.8281`
- base hit wins per episode: `0.9688`
- own-target penalties: `0.0`
- block steps per episode: `11.125`
- interference steps per episode: `11.0469`
- robot contacts per episode: `0.0`

Strict 16-episode replay audit:

- hard violations: `0`
- recovery events: `0`
- own-target penalties: `0.0`

## Interpretation

The learned policy currently uses interference mostly as lane blocking rather than direct collision. This is the desired default: collision is available, but the policy does not ram unless it sees a useful advantage. If future training needs more direct contact, increase the tactical-contact reward only when the opponent is close to a base-shot window and add a mild penalty for passive blocking that does not change opponent progress.
