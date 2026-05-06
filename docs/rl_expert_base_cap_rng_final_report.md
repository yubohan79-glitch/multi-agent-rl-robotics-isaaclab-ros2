# 专家策略基地命中概率修复最终报告

日期：2026-05-06

## 红色箱子实体化补丁

- 之前视频里出现红箱盖到车身上的画面，根因不是渲染截图误差，而是旧回放审计用圆形近似检查车箱接触，漏掉了带朝向的矩形车身与 30 cm 红箱 AABB 的重叠。旧 seed `10622` trace 复查出 62 个车箱重叠点，最大重叠约 4.8 cm。
- 已将 IsaacLab 红色障碍箱改成真实 PhysX 动态刚体：`rigidBodyEnabled=true`、`kinematicEnabled=false`、`collisionEnabled=true`、质量 `1.8 kg`、启用重力、高摩擦物理材质，并显式设置 contact/rest offset。导出审计文件：`isaaclab_sim/output/robocup_visionrl_pushable_physics_audit.json`。
- 规则环境、严格回放审计和 IsaacLab 回放层都改为“有朝向的机器人矩形外廓 vs 红箱 AABB”的 SAT 碰撞检测，不再用圆形近似判断车箱穿模。
- 重新生成 8 局物理箱严格回放：`docs/rl_expert_base_cap_rng_physical_boxes_strict8.md`，结果为 `hard_violations=0`、`warnings=0`、`base_wins_per_episode=1.0`。
- 重新录制通过视频抽帧检查的三视角回放：
  - `docs/media/isaaclab_expert_base_cap_physical_boxes_seed10705_top.mp4`
  - `docs/media/isaaclab_expert_base_cap_physical_boxes_seed10705_yellow_pov.mp4`
  - `docs/media/isaaclab_expert_base_cap_physical_boxes_seed10705_blue_pov.mp4`

## 修改内容

- 对照 `docs/rule_reference_pages/rule_page_07.png` 中的国赛规则参考图，确认红色障碍物为两块 30 cm 立方体，分别位于场地东北和西南区域。当前默认箱子中心保持为 `box_ne=(0.80, 0.80)`、`box_sw=(-0.80, -0.80)`，训练时只在同一象限内进行小范围随机扰动。
- 在 `isaaclab_sim/rl/robocup_visionrl_selfplay_env.py` 中加入 `base_retry_min_normal_hits`。如果两靶早攻基地触发基地命中上限抽签失败，则锁定当前普通靶数量窗口，小车必须先击倒更多普通靶，才能再次攻击基地靶。
- 将基地命中上限抽样拆分为独立的 `base_cap_rng`，与普通激光射击随机数分离。这样普通靶阶段的随机 miss 不会影响后续基地早攻概率。
- 在 `isaaclab_sim/rl/expert_policy.py` 中加入黄方、蓝方专家在基地早攻失败后的补靶顺序，让早攻失败后转为第三靶配置，而不是反复攻击同一个失败窗口。
- 调整黄方、蓝方专家节奏，在保留双方路线差异的同时，使 64 局胜率尽量接近 50%。
- 更新 `docs/media/README.md`，加入最新三个 IsaacLab 回放视频。

## 修改原因

之前的策略在两靶攻基地失败后，可能继续用同一个普通靶数量窗口反复攻击基地。由于该窗口已经被基地命中上限判定为失败，继续重试等价于死循环，会导致策略单一、比赛超时或观感异常。

此外，原来基地命中上限与普通射击共用随机数流，普通靶阶段消耗随机数的差异会间接影响基地命中概率。拆出独立 `base_cap_rng` 后，基地早攻概率更公平、更可复现。

## 验证结果

- 修改前基线测试：`71 passed`。
- 修改后最终测试：`72 passed`。
- 几何审计：`target_count=10`、`physics_checks=4`、`failures=0`。
- 严格回放 seed `10622`：`hard_violations=0`、`warnings=0`、`normal_hits_per_episode=6`、`base_wins_per_episode=1`。
- IsaacLab 三个视频均能正常打开，分辨率为 1280x720，帧率为 30 FPS。抽帧检查显示：靶子位置正确、基地挡板落地、红色箱子位移持续存在、没有机器人与箱子穿模。

## 训练与评估

- 续训 checkpoint：
  `isaaclab_sim/output/rl/mappo_dual_experts_post_retry_bluetempo_rs005_seed261020/policy.pt`
- 最终选用部署模式：
  `policy-mode expert`
- 评估 JSON：
  `isaaclab_sim/output/eval/expert_base_cap_rng_contract64.json`
- 评估 CSV：
  `isaaclab_sim/output/eval/expert_base_cap_rng_contract64.csv`

64 局评估结果：

| 指标 | 数值 |
|---|---:|
| yellow_win_rate | 0.4688 |
| blue_win_rate | 0.5312 |
| draw_rate | 0.0000 |
| mean_episode_time_s | 26.5016 |
| mean_yellow_score | 39.2188 |
| mean_blue_score | 42.7344 |
| mean_normal_hits_yellow | 2.2188 |
| mean_normal_hits_blue | 2.1719 |
| static_penetrations_total | 0 |
| box_penetrations_total | 0 |
| repeat_target_order_events_total | 0 |

普通靶击倒数量分布：

| 队伍 | 1 个靶 | 2 个靶 | 3 个靶 | 4 个靶 |
|---|---:|---:|---:|---:|
| yellow | 0.0469 | 0.6875 | 0.2656 | 0.0000 |
| blue | 0.0469 | 0.7344 | 0.2188 | 0.0000 |

按普通靶数量分组统计的基地命中率：

| 队伍 | 1 个靶 | 2 个靶 | 3 个靶 | 4 个靶 |
|---|---:|---:|---:|---:|
| yellow | 0.0000 | 0.4098 | 0.3333 | 0.0000 |
| blue | 0.0000 | 0.4754 | 0.3571 | 0.0000 |

箱子与接触指标：

| 指标 | 数值 |
|---|---:|
| yellow_push_events_per_episode | 2.4219 |
| blue_push_events_per_episode | 2.2188 |
| box_ne_mean_final_displacement_m | 0.1060 |
| box_sw_mean_final_displacement_m | 0.0605 |
| robot_contacts_per_episode | 0.0000 |
| relocalization_events_per_episode | 0.0000 |
| abnormal_spin_steps_per_episode | 0.0000 |

## 回放输出

- 严格回放 trace：
  `isaaclab_sim/output/replay/expert_base_cap_rng_seed10622/strict_replay_trace.csv`
- 严格回放事件：
  `isaaclab_sim/output/replay/expert_base_cap_rng_seed10622/strict_replay_events.jsonl`
- 严格回放审计报告：
  `docs/rl_expert_base_cap_rng_seed10622_strict_audit.md`
- 顶视角视频：
  `docs/media/isaaclab_expert_base_cap_seed10622_top.mp4`
- 黄车第一视角视频：
  `docs/media/isaaclab_expert_base_cap_seed10622_yellow_pov.mp4`
- 蓝车第一视角视频：
  `docs/media/isaaclab_expert_base_cap_seed10622_blue_pov.mp4`

## 视频检查结论

- 所有靶子均存在，没有嵌入墙体或基地挡板。
- 红色箱子被推动后位置持续变化；发生 jammed 事件时，机器人会被保持在箱子外侧，不允许穿过箱体。
- 小车没有穿过箱子、墙体 blocker 或基地挡板。
- 基地靶只在对应基地挡板被拆除后才被击中。
- 两辆小车同步开始比赛，选定回放中没有异常重定位原地旋转。

## 剩余风险

- 最终 64 局评估采用 `expert` 部署模式，因为 residual 续训仍会扰动某一方的射击精度。训练得到的 residual checkpoint 已保留，但当前回放和评估选择更稳定的专家控制器。
- 当前专家策略中 4 靶后攻基地的样本较少。现在已经能看到 2 靶和 3 靶之间的策略差异，但如果要显著增加 4 靶样本，需要更长课程学习或更细的奖励设计。
- IsaacLab 回放按严格 trace 逐帧应用箱子位置，用于验证渲染、遮挡和阻挡行为。完整 IsaacLab 闭环动态强化学习训练仍属于后续更大的工程。

## 建议绘制的图

- 系统架构图。
- ROS2 + IsaacLab + RL 闭环图。
- 多传感器融合结构图。
- 胜率柱状图。
- 训练曲线。
- 普通靶击倒数量分布图。
- 基地命中率随普通靶数量变化图。
- 可推动箱子位移统计图。
- 三视角回放截图拼图。
