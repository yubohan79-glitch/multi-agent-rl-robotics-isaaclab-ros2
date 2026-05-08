# Getting Started Guide

本文档解决“上手门槛高、缺少可复现路径”的问题。它把项目拆成三个可独立验证的层级：Python 规则环境、ROS2 运行栈、IsaacLab 回放/训练。新用户不需要一次性装完整 Isaac Sim，也可以先完成规则和算法 smoke test。

## 1. Repository Scope

推荐使用的仓库根目录：

```text
C:\Users\Administrator\Desktop\作品集\RoboCupVisionRL_IsaacLab_ROS2
```

主要目录：

| Path | Purpose |
| --- | --- |
| `crc_robocup_vision_ws/` | ROS2 Jazzy 工作空间，包含 bringup、navigation、vision、behavior、shooter、description、interfaces |
| `isaaclab_sim/` | IsaacLab 场景、规则环境、RL 训练/评估/导出脚本 |
| `config/` | 公开规则、靶子布局、计分契约 |
| `docs/rl_data/world_model_sacflow_final/` | 已发布的训练摘要、128 局评估、严格回放审计数据 |
| `docs/media/` | 最终三视角回放 MP4/GIF |
| `tests/` | pytest 合约测试 |

## 2. Environment Levels

### Level 0: Python-Only Smoke Test

用途：不启动 ROS2、不启动 IsaacLab，只验证规则环境、评估脚本和测试是否能跑通。

环境建议：

- Python 3.10 或更高版本
- PyTorch 可选；只跑部分测试时 CPU 也能工作
- Windows PowerShell 或 WSL Bash 均可

命令：

```powershell
cd "C:\Users\Administrator\Desktop\作品集\RoboCupVisionRL_IsaacLab_ROS2"
python -m pip install -r isaaclab_sim\rl\requirements.txt
python -m pytest tests -q
```

快速评估：

```powershell
cd "C:\Users\Administrator\Desktop\作品集\RoboCupVisionRL_IsaacLab_ROS2\isaaclab_sim\rl"
python evaluate_selfplay.py --episodes 8
```

预期结果：

- pytest 通过。
- 评估脚本输出比赛胜负、分数、普通靶击倒、基地命中、碰撞/穿模等字段。
- 如果失败，优先检查 Python 版本、依赖安装、当前路径是否正确。

### Level 1: ROS2 Dry Run

用途：验证 ROS2 包、launch、行为节点、视觉/射击接口能启动，但不要求真实机器人硬件。

建议环境：

- Ubuntu 24.04
- ROS2 Jazzy
- `colcon`
- `rosdep`

重要说明：如果在 Windows + WSL 中使用，建议把 `crc_robocup_vision_ws/` 复制到 Linux 原生路径，例如 `~/crc_robocup_vision_ws`。直接在 Windows 挂载路径或含非 ASCII 字符路径下构建，ROSIDL 可能失败。

命令：

```bash
cd ~/crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

黄方启动：

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  team_color:=yellow \
  target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.yellow.yaml
```

蓝方启动：

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  team_color:=blue \
  target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.blue.yaml
```

预期结果：

- `rcvrl_bringup` 能启动 competition launch。
- `shooter_dry_run:=true` 时不会调用真实激光硬件。
- 行为节点会读取 `team_color` 和 target route，并拒绝攻击己方 target owner。

### Level 2: IsaacLab Preview

用途：查看物理场景、靶子倒下、红色箱子推动、挡板遮挡、双车回放。

Windows PowerShell：

```powershell
cd "C:\Users\Administrator\Desktop\作品集\RoboCupVisionRL_IsaacLab_ROS2"
.\scripts\run_isaaclab_project.ps1 -Headless -DemoFlow -Duration 120
```

检查项目内 IsaacLab 进程：

```powershell
.\scripts\stop_project_isaaclab.ps1 -WhatIfOnly
```

仅在确认要停止本项目 IsaacLab 进程时执行：

```powershell
.\scripts\stop_project_isaaclab.ps1
```

预期结果：

- IsaacLab runtime 写入 `.isaaclab_runtime/`，不污染全局 Isaac Sim 目录。
- 回放能展示双车同时出发、普通靶击倒、红色箱子位移、基地挡板阻挡和最终基地击中。

## 3. Quick Demo Tutorial

这是给第一次打开仓库的用户准备的最短路径。

### Step 1: Read Current Evidence

无需运行训练，先看已发布结果：

```text
docs/rl_data/world_model_sacflow_final/training_summary.json
docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json
docs/rl_data/world_model_sacflow_final/strict_replay_summary.json
docs/media/最终回放_顶视角.mp4
docs/media/最终回放_黄车第一视角.mp4
docs/media/最终回放_蓝车第一视角.mp4
```

这些文件对应当前 README 中的 128 局随机评估和 8 局严格回放审计。

### Step 2: Run Contract Tests

```powershell
cd "C:\Users\Administrator\Desktop\作品集\RoboCupVisionRL_IsaacLab_ROS2"
python -m pytest tests -q
```

测试覆盖重点：

- 靶子 owner 和己方/敌方规则。
- 激光命中距离和 0.80 s dwell gate。
- 红色箱子、墙体、挡板、出发区隔板的规则合约。
- Sim2Real 配置格式和关键字段。

### Step 3: Run a Short Evaluation

```powershell
cd "C:\Users\Administrator\Desktop\作品集\RoboCupVisionRL_IsaacLab_ROS2"
python isaaclab_sim\rl\evaluate_strategy_contract.py --episodes 8 --stochastic
```

关注字段：

- `yellow_win_rate`
- `blue_win_rate`
- `draw_rate`
- `normal_hit_count_distribution`
- `base_success_by_hits`
- `push_events_per_episode`
- `robot_contacts_per_episode`
- `static_penetrations_total`
- `box_penetrations_total`

如果策略卡在点位附近，先看 `docs/parameter_tuning.md` 中的 micro-aim、side candidate 和 action shield 调参建议。

### Step 4: Inspect Replay

优先看顶视角，再看黄车和蓝车第一视角：

```text
docs/media/最终回放_顶视角.mp4
docs/media/最终回放_黄车第一视角.mp4
docs/media/最终回放_蓝车第一视角.mp4
```

检查项：

- 三视角是否从完整比赛开始展示。
- 两车是否同时离开起点。
- 红色箱子是否真实位移。
- 蓝色基地挡板是否在未拆除前阻挡车和激光。
- 小车是否没有穿箱、穿墙、穿挡板。
- 是否没有反复攻击不可见靶或己方靶。

## 4. Full Training Path

正式训练命令：

```bash
python3 isaaclab_sim/rl/train_world_model_sacflow_selfplay.py \
  --config isaaclab_sim/rl/configs/world_model_flow.yaml \
  --timesteps 200000 \
  --num-envs 32 \
  --batch-size 1024 \
  --learning-starts 4096 \
  --gradient-steps 2 \
  --hidden-dim 256 \
  --device cuda \
  --seed 260707 \
  --output isaaclab_sim/output/rl/world_model_sacflow_seed260707
```

训练后评估：

```bash
python3 isaaclab_sim/rl/evaluate_strategy_contract.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 128 \
  --stochastic \
  --output-json isaaclab_sim/output/eval/world_model_sacflow_contract_eval128.json \
  --output-csv isaaclab_sim/output/eval/world_model_sacflow_contract_eval128.csv
```

导出策略：

```bash
python3 isaaclab_sim/rl/export_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --format torchscript \
  --output-dir isaaclab_sim/output/policy_export/world_model_sacflow_seed260707
```

## 5. Troubleshooting

| Symptom | Likely Cause | Action |
| --- | --- | --- |
| ROS2 build fails under Windows path | ROSIDL/path encoding issue | Copy workspace to Linux native path and rebuild |
| pytest cannot import RL modules | wrong working directory or missing requirements | run from repo root and install `isaaclab_sim/rl/requirements.txt` |
| GPU idle, CPU high | too few gradient updates or CPU-bound env collection | tune `num_envs`, `batch_size`, `gradient_steps`; see `docs/parameter_tuning.md` |
| Robot freezes near target/base | poor aim pose or shield over-constraining action | add slow micro-rotation, side candidate poses, and inspect fire gate |
| Base target hit before blocker removal | scene/raycast rule bug | stop training, fix blocker geometry and raycast contract first |
| Red box only moves visually | missing dynamic rigid-body/collider path | fix scene physical properties before training |
| Results look good but replay violates rules | evaluation/replay contract mismatch | align target layout, collision, scoring and line-of-sight code before reporting |
