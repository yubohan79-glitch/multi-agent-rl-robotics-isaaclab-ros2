# RL Engineering Hardening Report

This report records the follow-up engineering pass after the full GPU MAPPO/self-play run. The goal was to make the reinforcement-learning module easier to reproduce, safer to modify, and cleaner for GitHub review.

## What Was Hardened

### 1. MAPPO configuration became the source of truth

The full-run configuration in `isaaclab_sim/rl/configs/mappo_selfplay.yaml` now matches the completed GPU experiment:

- `num_envs: 32`
- `timesteps: 500000`
- `rollout_steps: 256`
- `update_epochs: 4`
- `minibatch_size: 2048`
- `hidden_dim: 256`
- `gamma: 0.995`
- `device: cuda`
- output path: `../output/rl/mappo_selfplay_full_gpu`

The training script now accepts:

```bash
python3 isaaclab_sim/rl/train_mappo_selfplay_parallel_torch.py --config configs/mappo_selfplay.yaml
```

This prevents future reproduction runs from silently drifting away from the documented experiment.

### 2. The six-dimensional tactical action contract is protected by tests

The self-play environment now has tests that verify:

- `TACTICAL_ACTION_DIM == 6`
- YAML `deployment.action_contract` matches `TACTICAL_ACTION_LABELS`
- every robot action space uses the same six tactical controls
- each observation remains 34-dimensional
- the tactical target selector never chooses a same-side target
- localization recovery actions enter the relocalization branch
- archived GPU result data is present and consistent with the policy interface

The environment still accepts legacy 3-dimensional actions for backward-compatible smoke tests, but CI now also checks the real 6D MAPPO contract.

### 3. GPU experiment data is Git-reviewable

Runtime outputs under `isaaclab_sim/output/` remain ignored by Git, but the completed run data is archived under:

```text
docs/rl_data/mappo_selfplay_full_gpu/
```

Tracked files include:

- `training_curve.csv`
- `training_curve.jsonl`
- `training_summary.json`
- `mappo_full_gpu_eval.json`
- `mappo_full_gpu_eval_stochastic.json`
- `scripted_rules_baseline_eval.json`

This makes the SVG figures auditable: the figures are not hand-drawn claims; they are generated from committed experiment data.

### 4. CI coverage was extended

`.github/workflows/ros2-ci.yml` now compiles the new RL scripts:

- `evaluate_mappo_policy.py`
- `export_policy.py`
- `generate_rl_figures.py`
- `replay_mappo_policy_strict.py`
- `train_tactical_selfplay_numpy.py`
- `train_mappo_selfplay_parallel_torch.py`

The existing `python -m pytest tests -q -p no:cacheprovider` step will also run the new strategy-contract tests.

## New Tests

Added:

```text
tests/test_rl_strategy_contract.py
```

Updated:

```text
tests/test_rl_env_smoke.py
```

Important checks:

```bash
python3 -m pytest tests/test_rl_env_smoke.py tests/test_rl_strategy_contract.py -q
```

Expected result from this pass:

```text
10 passed
```

## Generated and Reviewed Figures

The data-driven figures remain:

- `docs/figures/rl/rl_training_curve_gpu.svg`
- `docs/figures/rl/rl_strategy_event_metrics.svg`
- `docs/figures/rl/rl_policy_episode_trace.svg`
- `docs/figures/rl/rl_tactical_contract.svg`

Regenerate them with:

```bash
python isaaclab_sim/rl/generate_rl_figures.py
```

The figure manifest points at the committed data copy under `docs/rl_data/mappo_selfplay_full_gpu/`.

## Verification Performed

This pass verified:

- WSL CUDA Python can display the MAPPO training script help and config entry
- a tiny config-entry training run completes
- the MAPPO checkpoint exports to a TorchScript decentralized actor
- the trained stochastic MAPPO actor passes a 32-episode strict replay audit with zero hard violations
- RL scripts compile in WSL
- strategy-contract tests pass
- `git diff --check` passes
- all referenced report, data and figure paths exist

Windows local `python` currently resolves to Inkscape's Python, which does not include `pytest` or `torch`. RL validation should therefore be run through WSL or a normal project Python environment.

## Current State

The project now has:

- a completed GPU MAPPO training record
- archived CSV/JSON data
- generated SVG result figures
- reproducibility commands
- a full strategy report
- CI-backed strategy-contract tests
- a config-driven MAPPO training entry point
- a TorchScript export path for the decentralized tactical actor
- a strict post-training replay audit report and archived replay summary

The next useful engineering step is to add a ROS2-facing policy adapter node that loads the exported actor and publishes high-level tactical commands into `rcvrl_behavior`.
