# Scene Adaptation Tutorial

本文档解决“缺少场景适配教程”的问题。目标是让用户能把本项目从当前 RoboCup-style 视觉对抗场景迁移到新的靶子布局、障碍配置或路线配置，同时保持训练、评估、回放共用同一套规则契约。

## 1. Adaptation Principle

场景适配必须遵守一个原则：

```text
layout/config -> rule environment -> IsaacLab scene -> ROS2 route -> evaluation/replay
```

不要只改渲染场景，也不要只改 RL 环境。正式结果必须保证以下内容一致：

- target layout
- target owner
- target yaw
- wall/blocker/start partition geometry
- red pushable box geometry
- base blocker geometry
- laser line-of-sight
- collision
- hit dwell
- shooting range
- scoring

## 2. Files to Edit

| File | Purpose |
| --- | --- |
| `config/arena_rules.yaml` | 场地尺寸、墙体、挡板、出发区隔板、物理规则 |
| `config/target_layout.yaml` | 靶子名字、owner、位置、朝向、普通靶/基地靶分类 |
| `config/scoring.yaml` | 普通靶、基地靶、胜负和惩罚计分 |
| `crc_robocup_vision_ws/src/rcvrl_navigation/config/targets.elimination.yellow.yaml` | 黄方 ROS2 路线 |
| `crc_robocup_vision_ws/src/rcvrl_navigation/config/targets.elimination.blue.yaml` | 蓝方 ROS2 路线 |
| `crc_robocup_vision_ws/src/rcvrl_bringup/config/sim2real.yaml` | 实车标定、传感器、射击和域随机化参数 |
| `isaaclab_sim/rl/robocup_visionrl_selfplay_env.py` | 快速规则环境和 self-play contract |
| `isaaclab_sim/rl/evaluate_strategy_contract.py` | 正式评估和审计字段 |

如果只改了某一个文件，通常是不完整适配。

## 3. Target Layout Checklist

每个靶子都需要定义和审计：

- `name`: 唯一名字，例如 `T01_NorthMiddle`。
- `owner`: 归属方。黄车只能攻击蓝方靶，蓝车只能攻击黄方靶。
- `xy`: 靶面中心在 map/arena 平面的坐标。
- `yaw`: 靶面朝向。普通靶应约与相邻墙面成 45 度。
- target type: normal target 或 base target。
- blocker dependency: 基地靶是否需要拆掉对应挡板后才可见。
- shooting range: 普通靶和基地靶的合法射击距离不同。

审计要点：

1. 靶子不能嵌墙。
2. 靶子不能夹在墙和挡板之间。
3. front-face probe 不能直接落入墙体或挡板。
4. owner 不能反。
5. 基地靶不能在挡板未拆除时被 raycast 命中。

## 4. Obstacle and Blocker Checklist

### Red Pushable Box

必须满足：

- 是真实可推动刚体。
- 有 collider。
- 小车接触时只能推动或被阻挡，不能穿箱。
- 推动后 box pose 在 trace 中持续变化。
- RL observation 或 world model 能看到 box pose/displacement。

验证指标：

- `push_events_per_episode`
- `mean_final_box_displacement_m`
- `box_penetrations_total`

### Blue Base Blocker

必须满足：

- 落地。
- 阻挡车辆。
- 阻挡激光 raycast。
- 未拆挡板前基地靶 100% 不可命中。
- 拆除后只能从合法侧和合法距离命中。

验证指标：

- blocked base raycast test。
- base hit success grouped by normal target count。
- strict replay hard violations。

### Walls and Start Partitions

必须满足：

- 车体不能穿墙。
- 起点隔板不能被穿过。
- 导航路线不能要求机器人从非法区域穿越。
- collision 或 penetration 一旦出现，应进入失败/重定位/终止逻辑，而不是继续计成功。

## 5. Laser and Hit Rule Checklist

普通靶：

- 距离范围：5 cm 到 50 cm。
- 需要合法 opponent target。
- 需要 line-of-sight。
- 需要 0.80 s dwell gate。

基地靶：

- 距离范围：20 cm 到 80 cm。
- 未拆对应挡板前不可命中。
- 只能从已拆挡板一侧和合法视线命中。

命中概率：

- 0.80 s 以内必须 100% 不倒。
- 0.80 s 到 2.00 s 概率线性增强。
- 即使 dwell 增加，也应保留失败概率，不能变成任意角度秒杀。

## 6. ROS2 Route Adaptation

路线文件要和 target layout 一致：

```text
crc_robocup_vision_ws/src/rcvrl_navigation/config/targets.elimination.yellow.yaml
crc_robocup_vision_ws/src/rcvrl_navigation/config/targets.elimination.blue.yaml
```

修改路线时检查：

- 每个 route target 的 owner 与 `config/target_layout.yaml` 一致。
- 黄方 route 不包含黄方靶作为攻击目标。
- 蓝方 route 不包含蓝方靶作为攻击目标。
- 基地攻击 pose 位于合法侧，不应隔着挡板或墙体射击。
- 到点后允许小范围微调 yaw/side pose，不要把路线点放到墙角死点。

## 7. RL Adaptation

场景变更后，不建议直接沿用旧 checkpoint 作为正式结果。推荐顺序：

1. 修改 layout/rules。
2. 跑 pytest 和规则审计。
3. 跑专家/脚本策略短评估，确认场景可解。
4. 过滤失败专家数据。
5. warm start 或 distillation。
6. World Model + SAC Flow self-play。
7. 64 到 128 局评估。
8. 严格回放审计。
9. 三视角视频人工检查。

旧 checkpoint 可以作为 baseline 或 warm start，但不能把旧结果直接改名为新场景结果。

## 8. Required Validation Commands

Python tests：

```powershell
cd "C:\Users\Administrator\Desktop\作品集\RoboCupVisionRL_IsaacLab_ROS2"
python -m pytest tests -q
```

短评估：

```powershell
python isaaclab_sim\rl\evaluate_strategy_contract.py --episodes 8 --stochastic
```

正式评估：

```bash
python3 isaaclab_sim/rl/evaluate_strategy_contract.py \
  --checkpoint isaaclab_sim/output/rl/<run_name>/policy.pt \
  --episodes 128 \
  --stochastic \
  --output-json isaaclab_sim/output/eval/<run_name>_contract_eval128.json \
  --output-csv isaaclab_sim/output/eval/<run_name>_contract_eval128.csv
```

## 9. Promotion Criteria

新场景结果可以写进 README 或论文图表前，至少满足：

- 靶子审计无嵌墙、夹墙、owner 错误、yaw 错误。
- 红色箱子真实可推动。
- 蓝色基地挡板真实阻挡车和激光。
- 0.80 s dwell gate 生效。
- 128 局评估胜率不过度失衡。
- `static_penetrations_total == 0`。
- `box_penetrations_total == 0`。
- 三视角回放完整展示比赛开始到结束。
- 视频人工检查通过。
