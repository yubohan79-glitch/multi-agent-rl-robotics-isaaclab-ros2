# RL Convergence Report

Date: 2026-05-04

## Goal

Train and select a two-robot RoboCup visual duel policy whose self-play result is close to a fair 50/50 equilibrium:

- Yellow and blue win rates should stay close to 50%.
- Neither side should collapse to a low win rate.
- Draw/timeout rate should stay low.
- Own-target fire, hard robot contact, and wall/armor penetration must remain zero.
- The policy must preserve the 5-50 cm laser range and physical differential-drive motion.

## Method

### Residual Expert MAPPO

The trainer uses MAPPO/self-play with a safe residual expert contract:

- `policy_mode=residual_expert`
- scripted safe expert action as the low-risk baseline
- bounded MAPPO residual on target priority, base-rush timing, fire timing, recovery, and risk
- residual L2 regularization to keep exploration near executable behavior

This keeps the policy legal and physically plausible while still allowing tactical learning.

### Dual Actor CTDE

The selected training run uses two decentralized actors with one centralized critic:

- yellow actor: `pi_y(o_y)`
- blue actor: `pi_b(o_b)`
- centralized critic: `V(o_y, o_b)`
- actor mode: `dual`

This keeps the CTDE structure used by MAPPO while allowing each side to correct side-specific residuals during training.

### Symmetric Deployment Export

Raw dual-actor checkpoints still showed window-dependent color bias. The deployment checkpoint is exported with a canonical symmetry constraint:

- average the trained yellow and blue actor weights into one canonical actor
- copy the canonical actor back to both decentralized actors
- zero the first-layer `team_id` weight so color does not leak into deployment decisions
- keep the canonical-frame observation, so each robot still acts from its own local state
- reduce deployment exploration by setting `log_std = -1.95`

Export command:

```bash
python3 isaaclab_sim/rl/build_balanced_dual_actor_policy.py \
  --checkpoints isaaclab_sim/output/rl/mappo_fair_dual_actor_v1_seed_17_005_050_gpu/policy.pt \
  --log-std -1.95 \
  --output isaaclab_sim/output/rl/best_fair_dual_actor
```

## Training Run

Source checkpoint:

```text
isaaclab_sim/output/rl/mappo_fair_dual_actor_v1_seed_17_005_050_gpu/policy.pt
```

Training command family:

```bash
RUN_TAG=dual_actor_v1 \
TIMESTEPS=420000 \
NUM_ENVS=32 \
ROLLOUT_STEPS=256 \
DEVICE=cuda \
ENT_COEF=0.001 \
POLICY_MODE=residual_expert \
RESIDUAL_SCALE=0.16 \
RESIDUAL_L2_COEF=0.004 \
ACTOR_MODE=dual \
SEEDS="7 17 29" \
bash isaaclab_sim/rl/run_fair_multiseed_training.sh
```

Hardware:

```text
CUDA: true
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.11.0+cu128
```

## Selected Checkpoint

Recommended policy for replay, evaluation, and GitHub release:

```text
isaaclab_sim/output/rl/best_fair_dual_actor/policy.pt
```

Manifest:

```text
isaaclab_sim/output/rl/best_fair_dual_actor/balanced_policy_manifest.json
```

## Evaluation

### 512-Episode Stochastic Validation

```text
isaaclab_sim/output/eval/sym_seed17_temp014_eval512_stochastic_100000.json
```

| Metric | Value |
|---|---:|
| Episodes | 512 |
| Yellow win rate | 47.07% |
| Blue win rate | 50.59% |
| Draw / timeout | 2.34% |
| Win-rate gap | 3.52 pp |
| Mean episode time | 24.54 s |
| Normal hits / episode | 3.67 |
| Base-hit terminal episodes | 97.66% |
| Own-target penalties / episode | 0.00 |
| Robot contacts / episode | 0.00 |
| Collision recovery events / episode | 0.00 |

### 128-Episode Deterministic Check

```text
isaaclab_sim/output/eval/sym_seed17_eval128_deterministic_37500.json
```

| Metric | Value |
|---|---:|
| Episodes | 128 |
| Yellow win rate | 51.56% |
| Blue win rate | 48.44% |
| Draw / timeout | 0.00% |
| Own-target penalties / episode | 0.00 |
| Robot contacts / episode | 0.00 |

### Combined Stochastic Evidence

Across the low-temperature validation windows used for final selection:

| Metric | Value |
|---|---:|
| Episodes | 768 |
| Yellow wins | 365 |
| Blue wins | 381 |
| Draw / timeout | 22 |
| Yellow win rate | 47.53% |
| Blue win rate | 49.61% |
| Draw / timeout | 2.86% |
| Win-rate gap | 2.08 pp |

## Current Conclusion

The selected policy is not a mathematically exact Nash equilibrium, but it is the most balanced deployment checkpoint so far:

- both sides remain above 47% win rate in the combined stochastic validation
- deterministic deployment is close to 50/50
- own-target fire remains zero
- hard robot contact remains zero
- base target victories still occur in nearly all episodes
- the policy keeps the close-range 5-50 cm shooting contract

The next improvement, if more time is available, is to add a fairness validation callback during MAPPO training so checkpoint selection is automatic instead of post-training export based.
