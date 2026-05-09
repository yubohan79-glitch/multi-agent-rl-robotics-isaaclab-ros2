# Getting Started Guide

This guide gives a minimal path for reproducing the public repository evidence. The project can be inspected at three levels: Python-only rule tests, ROS2 dry run, and IsaacLab replay.

## 1. Repository Scope

Main directories:

| Path | Purpose |
| --- | --- |
| `crc_robocup_vision_ws/` | ROS2 Jazzy workspace for robot bringup, navigation, vision, behavior, shooter and interfaces |
| `isaaclab_sim/` | IsaacLab scene, replay utilities, rule environment and RL tooling |
| `isaaclab_sim/rl/` | Self-play environments, world-model SAC Flow training, evaluation and export scripts |
| `config/` | Public arena, target layout and scoring contracts |
| `docs/rl_data/` | Published training summaries, evaluation JSON/CSV and replay audit data |
| `docs/media/` | Final MP4/GIF replay media |
| `tests/` | Pytest checks for rule contracts, target layout, strategy logic and Sim2Real configuration |

## 2. Environment Levels

### Level 0: Python-Only Smoke Test

Use this first if you only want to validate the rule environment and evaluation utilities.

Requirements:

- Python 3.10 or newer
- `pip`
- Optional CUDA PyTorch for training; CPU is enough for many smoke tests

Commands:

```bash
python -m pip install -r isaaclab_sim/rl/requirements.txt
python -m pytest tests -q
```

Quick rule-environment evaluation:

```bash
cd isaaclab_sim/rl
python evaluate_selfplay.py --episodes 8
```

Expected behavior:

- tests complete without rule-contract failures;
- evaluation prints match score, winner, target hits, base hits, collision and penetration fields;
- no IsaacLab or ROS2 installation is required for this level.

### Level 1: ROS2 Dry Run

Use this to validate that the ROS2 workspace builds and launch files start without requiring physical robot hardware.

Recommended platform:

- Ubuntu 24.04
- ROS2 Jazzy
- `colcon`
- `rosdep`

If you use WSL, copy `crc_robocup_vision_ws/` to a native Linux path such as `~/crc_robocup_vision_ws`. Building ROS2 packages directly under a Windows-mounted path with non-ASCII characters can break ROSIDL generation.

Commands:

```bash
cd ~/crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

Useful launch variants:

```bash
ros2 launch rcvrl_bringup competition.launch.py team_color:=yellow target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.yellow.yaml
ros2 launch rcvrl_bringup competition.launch.py team_color:=blue target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.blue.yaml
```

Expected behavior:

- bringup, behavior, navigation, vision and shooter service nodes can be launched;
- shooter can run in dry-run mode;
- no physical robot is required for the dry run.

### Level 2: IsaacLab Replay

Use this to inspect the published replay behavior. IsaacLab/Isaac Sim setup is heavier than the Python rule tests, so start here only after Level 0 works.

On Windows, use the project wrapper so runtime files stay under `.isaaclab_runtime/` instead of the global Isaac Sim cache:

```powershell
.\scripts\run_isaaclab_project.ps1 -Headless -DemoFlow -Duration 120
```

To inspect or stop only this project's IsaacLab processes:

```powershell
.\scripts\stop_project_isaaclab.ps1 -WhatIfOnly
.\scripts\stop_project_isaaclab.ps1
```

Published replay media:

```text
docs/media/最终回放_顶视角.mp4
docs/media/最终回放_黄车第一视角.mp4
docs/media/最终回放_蓝车第一视角.mp4
docs/media/最终回放_顶视角.gif
docs/media/最终回放_黄车第一视角.gif
docs/media/最终回放_蓝车第一视角.gif
```

## 3. Training and Evaluation

The public training and evaluation artifacts are already included under `docs/rl_data/`. If you want to regenerate them, use the commands in `docs/reproducibility.md`.

Important generated-output rule:

- local training outputs go under `isaaclab_sim/output/`;
- temporary videos, cache files and debug frames should not be committed unless they are selected final evidence;
- public claims should point to JSON/CSV metrics, replay audits and MP4/GIF files.

## 4. First Files to Read

Recommended order:

1. `README.md`
2. `docs/admissions_project_brief.md`
3. `docs/capability_boundaries.md`
4. `docs/reproducibility.md`
5. `docs/parameter_tuning.md`
6. `docs/scene_adaptation.md`

## 5. Common Issues

### ROS2 build fails under WSL

Move the workspace to a native Linux path:

```bash
cp -r /mnt/c/path/to/crc_robocup_vision_ws ~/crc_robocup_vision_ws
```

Then rebuild from `~/crc_robocup_vision_ws`.

### IsaacLab opens global cache or conflicts with another project

Use `scripts/run_isaaclab_project.ps1`, which sets project-local runtime paths. Inspect running processes with `scripts/stop_project_isaaclab.ps1 -WhatIfOnly` before stopping anything.

### A result looks too good

Do not rely on reward alone. Check:

- `docs/capability_boundaries.md`
- `docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json`
- `docs/rl_data/world_model_sacflow_final/strict_replay_summary.json`
- replay MP4/GIF files under `docs/media/`

### A 50v50 claim is being evaluated

Treat 50v50 as simulation-stage rule-level evidence only. The current repository does not claim 100-robot hardware deployment or full rigid-body RL training for all 100 vehicles.
