# Parameter Tuning Guide

本文档解决“算法参数调优说明不足”的问题。调参目标不是单纯提高 reward，而是在规则可信的前提下同时维持胜率均衡、穿模为零、基地命中合法、箱子真实推动和回放行为可解释。

## 1. Current Reference Configuration

当前公开结果使用的配置来自：

```text
docs/rl_data/world_model_sacflow_final/training_summary.json
```

核心参数：

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `timesteps` | 200000 | 总训练步数 |
| `num_envs` | 32 | 并行规则环境数量 |
| `hidden_dim` | 256 | actor/critic/world model 隐层宽度 |
| `batch_size` | 1024 | SAC 更新 batch size |
| `replay_size` | 200000 | replay buffer 容量 |
| `learning_starts` | 4096 | 开始梯度更新前的采样步数 |
| `gradient_steps` | 2 | 每轮环境采样后的更新次数 |
| `gamma` | 0.995 | 长时序折扣因子 |
| `tau` | 0.01 | target critic soft update |
| `actor_lr` | 0.0003 | actor 学习率 |
| `critic_lr` | 0.0003 | critic 学习率 |
| `world_model_lr` | 0.0003 | object-centric dynamics model 学习率 |
| `alpha_lr` | 0.0003 | entropy temperature 学习率 |
| `target_entropy` | -6.0 | SAC exploration target |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `flow_steps` | 3 | flow policy refinement steps |
| `flow_velocity_scale` | 0.2 | flow velocity scaling |
| `policy_mode` | `residual_expert` | 专家先验 + residual policy |
| `residual_scale` | 0.04 | learned residual action scale |
| `domain_randomization` | true | 域随机化 |
| `action_shield` | true | 几何/规则安全屏蔽 |
| `world_model_coef` | 0.25 | 世界模型辅助损失权重 |

公开评估结果：

| Metric | Value |
| --- | ---: |
| Episodes | 128 |
| Yellow win rate | 49.22% |
| Blue win rate | 50.78% |
| Draw rate | 0.00% |
| Mean episode time | 30.8148 s |
| Yellow mean score | 40.8984 |
| Blue mean score | 41.7188 |
| Static penetrations | 0 |
| Box penetrations | 0 |
| Robot contacts/game | 0.00 |

严格回放审计：

| Metric | Value |
| --- | ---: |
| Episodes | 8 |
| Hard violations | 0 |
| Warnings | 0 |
| Own-target penalties/episode | 0.0 |
| Base wins/episode | 1.0000 |

## 2. Do Not Tune Around Broken Rules

下面问题必须先修环境或规则，不应该先调 reward：

- 小车穿过红色箱子、墙体、蓝色挡板或出发区隔板。
- 红色箱子不能被真实推动，只是视觉上移动。
- 蓝色基地挡板没有落地，或不能阻挡激光 raycast。
- 未拆挡板前可以命中基地靶。
- 靶子嵌墙、夹墙、朝向错误。
- 激光 0.80 s 以内也能稳定击倒靶子。
- 普通靶和基地靶射击距离不符合规则。

这些问题会污染 replay buffer，使任何训练结果失去可信度。

## 3. Tuning Recipes

### Robot Stuck Near Target or Base

症状：

- 机器人到达射击点后停住。
- 目标在视野附近但命中率低。
- `abnormal_spin_steps` 或超时上升。

优先检查：

1. 目标点是否离墙、挡板、箱子过近。
2. action shield 是否把合法微调动作全部屏蔽。
3. fire gate 是否要求过窄的角度或距离。
4. 目标 yaw 和 front-face probe 是否正确。

可调方向：

- 在射击 pose 后加入小角度慢速扫描，例如每步小幅改变 yaw，而不是原地大幅旋转。
- 为基地靶添加侧向候选点，使机器人可以从已拆挡板一侧微调坐标和角度。
- 适度提高 `residual_scale`，让策略能修正专家先验；如果安全性下降，则回退。
- 降低过强的 stuck penalty，避免机器人在合法微调阶段被过早判坏。
- 增加 DAgger/专家样本中“到点后微调并合法命中”的片段。

### Yellow/Blue Win Rate Imbalance

目标区间：

- yellow win rate: 45% 到 55%
- blue win rate: 45% 到 55%
- draw rate: 尽量接近 0

优先检查：

1. 黄/蓝 target route 是否对称但不完全同质。
2. 出发时机是否一致。
3. 目标 owner 是否正确。
4. 某一侧是否有更短的基地攻击窗口。
5. 箱子路线是否只偏向一方。

可调方向：

- 调整 yellow/blue expert prior，使两侧都有正常靶、推箱和早攻基地窗口。
- 检查 reward symmetry，避免某一侧同样事件获得不同 reward。
- 增加 self-play 采样中弱势一方起手和中局状态。
- 对过强的 early base rush 加成本，而不是直接禁用基地攻击。

### Base Hit Too Easy

目标：

- 打 1 个普通靶后攻基地：成功率不应过高，约不超过 40%。
- 打 2 个普通靶后攻基地：约不超过 55%。
- 打 3 个或 4 个普通靶后，基地成功率应明显提升。

优先检查：

1. 未拆挡板前 raycast 是否 100% 失败。
2. 基地射击距离是否限制在 20 cm 到 80 cm。
3. 0.80 s dwell 是否生效。
4. 0.80 s 到 2.00 s 命中概率是否线性增强且保留失败概率。

可调方向：

- 增加挡板侧向合法射击点，而不是允许任意角度命中。
- 提高错误侧攻击成本。
- 增加“已拆挡板数量/普通靶数量”对基地命中概率的影响。

### Pushable Box Not Used

症状：

- `push_events_per_episode` 接近 0。
- `mean_final_box_displacement_m` 接近 0。
- 回放中箱子没有持续位移。

优先检查：

1. 红色箱子是否是 dynamic rigid body。
2. collider 是否足够厚，不能被小车穿过。
3. 摩擦、质量、速度上限是否导致箱子几乎不可推动。
4. 推箱路线是否真的缩短攻击路径。

可调方向：

- 增加 safe push progress reward。
- 降低无意义撞箱惩罚，但保留穿模/卡住惩罚。
- 在专家数据中加入成功推箱路线。
- 让世界模型显式观测 box pose 和 box displacement。

### Penetration or Illegal Collision Appears

处理原则：

1. 暂停训练。
2. 用最短 replay 复现。
3. 修 collider、物理步进、raycast 或 done reason。
4. 重新跑 contract tests。
5. 丢弃被污染的 replay buffer 或 checkpoint。

不要用 reward 惩罚穿模来替代物理修复。穿模如果存在，训练很容易学到投机路径。

### CPU High, GPU Low

症状：

- CPU 占用高。
- GPU 利用率低。
- 训练速度慢。

可调方向：

- 降低 `num_envs`，检查是否单个环境 step 过慢或死循环。
- 增大 `batch_size` 或 `gradient_steps`，把更多时间转到 GPU 更新。
- 确认 PyTorch device 是 `cuda`。
- 不要盲目加并行环境；如果环境逻辑 CPU-bound，会让系统更卡。

### GPU Memory Pressure

症状：

- CUDA OOM。
- 显存长期接近上限。

可调方向：

- 降低 `batch_size`。
- 降低 `hidden_dim`。
- 降低 `gradient_steps`。
- 保持 `replay_size` 不变也可以，replay buffer 多数在 CPU 内存；真正影响显存的是 batch、网络尺寸和中间激活。

## 4. Evaluation Gate Before Promotion

一个 checkpoint 不能只凭 reward 上升进入 README 或论文图表。至少需要：

1. 64 局以上多 seed 评估，推荐 128 局。
2. 输出 JSON/CSV。
3. 检查 `yellow_win_rate`、`blue_win_rate`、`draw_rate`。
4. 检查普通靶击倒数量分布。
5. 检查按普通靶数量分组的基地命中率。
6. 检查 `static_penetrations_total == 0`。
7. 检查 `box_penetrations_total == 0`。
8. 检查 `robot_contacts_per_episode` 和 abnormal spin。
9. 生成三视角完整回放。
10. 人工检查视频，确认没有穿模、非法基地命中和不可解释卡死。

## 5. Output Naming

建议按以下结构保存：

```text
isaaclab_sim/output/rl/<run_name>/policy.pt
isaaclab_sim/output/rl/<run_name>/training_curve.csv
isaaclab_sim/output/eval/<run_name>_contract_eval128.json
isaaclab_sim/output/eval/<run_name>_contract_eval128.csv
isaaclab_sim/output/replay/<run_name>/strict_replay_summary.json
isaaclab_sim/output/policy_export/<run_name>/
```

提交到 Git 前只保留公开必要的小体积结果和最终媒体。完整训练输出默认应留在 ignored output 目录中。
