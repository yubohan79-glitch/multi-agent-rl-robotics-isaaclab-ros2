# 世界模型与流/扩散策略强化学习改造方案

本文档面向 `RoboCupVisionRL_IsaacLab_ROS2`，目标是评估并规划将当前 PPO/MAPPO-style 训练路线升级为更前沿的世界模型 RL、Flow Policy 或 Diffusion Policy RL。结论先行：如果不考虑移植成本，我推荐采用 **多智能体对象中心世界模型 + PolicyFlow/SAC Flow + action-chunk 扩散/流策略微调** 的混合路线，而不是只替换成单一算法。

核验日期：2026-05-07。

## 1. 总体推荐

### 推荐路线

```text
阶段 A：PolicyFlow / SAC Flow 替换高斯 PPO/MAPPO 策略头
阶段 B：加入 TD-MPC2 / Newt 类对象中心世界模型
阶段 C：用 DPPO / ReinFlow 做 action chunk 策略微调
阶段 D：保留规则安全盾、几何审计、IsaacLab 三视角回放闭环
```

### 推荐排名

| 排名 | 路线 | 研究价值 | 与项目匹配度 | 结论 |
|---:|---|---:|---:|---|
| 1 | 对象中心世界模型 + PolicyFlow/SAC Flow | 极高 | 极高 | 最推荐。能同时表达长程推箱、靶子选择、多路线策略和双车差异。 |
| 2 | SAC Flow / PolicyFlow 直接替换 PPO/MAPPO | 很高 | 很高 | 最适合先落地。比 Gaussian PPO 更能表达多峰动作分布。 |
| 3 | TD-MPC2-style latent planner | 很高 | 高 | 适合推箱、绕障、基地攻坚窗口的长程规划。 |
| 4 | Newt-style 多任务世界模型 | 极高 | 中高 | 研究亮点最大，但需要构造任务 token / demonstration 数据。 |
| 5 | DreamerV3-style imagination actor-critic | 高 | 中高 | 成熟，但对当前低维对象状态不是最经济。 |
| 6 | DPPO / ReinFlow | 高 | 中 | 适合动作片段微调，单独替换主算法不如 Flow/SAC Flow。 |

我的最终建议是：**先做 PolicyFlow/SAC Flow 训练器，再把世界模型接进来**。如果直接上 Newt/DreamerV3 完整世界模型，工程跨度会非常大，而且容易把已有规则、几何约束、回放审计打散。

## 2. 算法介绍与发行时间

### 2.1 TD-MPC2

**发行时间**

- arXiv 首次提交：2023-10-25。
- ICLR 2024 Spotlight。
- 论文：`TD-MPC2: Scalable, Robust World Models for Continuous Control`。

**核心原理**

TD-MPC2 是模型式强化学习。它学习一个隐空间世界模型，不直接重建完整观测，而是在 latent state 中建模：

```text
encoder: observation -> latent state
dynamics: latent state + action -> next latent state
reward head: latent state + action -> reward
Q head: latent state + action -> value
policy prior: latent state -> candidate action
planner: 在 latent 中做短时域 MPC/CEM 搜索
```

它的关键不是“预测视频”，而是学习一个足够支持控制的隐空间动力学。执行时可以用模型预测未来若干步，在候选动作序列中选最优动作。

**适合本项目的原因**

- 推箱是典型长程动力学问题：小车动作会改变箱子位置，箱子又改变之后的可行路径。
- 基地攻击需要规划“拆挡板的一侧”“20-80 cm 距离”“0.8 s 驻留”等未来条件。
- TD-MPC2 的 latent planning 比纯 PPO/MAPPO 更适合评估“现在清第几个靶、之后基地命中率是否更高”。

**局限**

- 原生 TD-MPC2 多为单智能体连续控制；本项目需要扩展为 multi-agent centralized training。
- 需要稳定的 transition dataset 和世界模型验证指标。

### 2.2 DreamerV3

**发行时间**

- arXiv 首次提交：2023-01-10。
- 论文：`Mastering Diverse Domains through World Models`。

**核心原理**

DreamerV3 学习一个 recurrent state-space world model，然后在模型想象出的 latent rollout 中训练 actor-critic：

```text
真实交互数据 -> replay buffer
RSSM world model -> 预测 latent、reward、continue
imagined rollout -> actor-critic 更新
真实环境只负责采样，策略优化主要在想象空间完成
```

它的特点是通用性强，能处理图像、低维状态、连续动作、离散动作、稀疏奖励等多种任务。

**适合本项目的原因**

- 比 PPO 更样本高效。
- 可以在想象空间中训练“不同靶子数量后攻基地”的长程收益。
- 适合未来接入真实视觉输入或更复杂 IsaacLab 传感器。

**局限**

- DreamerV3 更像通用世界模型路线，默认不显式知道“靶子、箱子、挡板”等对象。
- 当前项目已经有明确对象几何，直接做对象中心模型会更高效。

### 2.3 Newt 类世界模型

**发行时间**

- arXiv 首次提交：2025-11-24。
- ICLR 2026。
- 论文：`Learning Massively Multitask World Models for Continuous Control`。

**核心原理**

Newt 是语言条件、多任务、连续控制世界模型。它先从 demonstrations 中预训练，学习任务表征和动作先验，再用在线交互继续优化。核心结构可以抽象为：

```text
task/language token + observation history -> world model latent
world model latent -> action prior
online RL -> joint optimization
unseen task -> 快速适配
```

**适合本项目的原因**

本项目天然有多个任务 token：

- `normal_target_clear`
- `early_base_rush`
- `push_box_route`
- `block_opponent`
- `recover_after_contact`
- `yellow_expert`
- `blue_expert`

也就是说，我们可以把 Newt 的“语言条件”改造成“规则/战术 token 条件”，让一个世界模型学习多种战术子任务。

**局限**

- Newt 不是一个小型单任务替换算法，而是一套大规模多任务世界模型范式。
- 如果没有足够多的 demonstrations、专家轨迹和多任务数据，效果可能不如 SAC Flow 直接训练。

### 2.4 World Model + Flow/Diffusion Policy RL

**相关发行时间**

- Diffusion Policy：arXiv 2023-03-07。
- DPPO：arXiv 2024-09-01。
- ReinFlow：arXiv 2025-05-28，NeurIPS 2025。
- SAC Flow：arXiv 2025-09-30，2026-01-14 修订。
- PolicyFlow：arXiv 2026-02-01，投稿 ICLR 2026。

**核心原理**

传统 PPO/MAPPO 通常使用高斯策略：

```text
obs -> mean, std -> action
```

它很难表达多峰动作分布。例如同一个局面里，合理动作可能同时包括：

- 左侧绕箱；
- 右侧推箱；
- 先清普通靶；
- 直接早攻基地；
- 暂时阻挡对手。

Flow/Diffusion Policy 把策略变成一个生成模型：

```text
noise + condition(obs/history/task token) -> action or action sequence
```

这样策略不再只是“一个均值附近采样”，而是可以生成多种合理动作模式。

**在本项目中的价值**

本项目的策略空间是明显多模态的。对同一观测，黄车和蓝车可能需要学到不同节奏：

- 黄车快攻；
- 蓝车先推箱；
- 黄车打 2 个靶再攻基地；
- 蓝车打 3 个靶提高基地成功率；
- 某一方在领先时转为阻挡。

Flow/Diffusion Policy 更适合这种多策略分布。

### 2.5 SAC Flow

**发行时间**

- arXiv 首次提交：2025-09-30。
- 修订版本：2026-01-14。
- 论文：`SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling`。

**核心原理**

SAC Flow 把 flow-based policy 接到 off-policy SAC 框架里。它指出 flow rollout 和残差 RNN 计算等价，因此容易出现梯度爆炸/消失。解决办法是重新参数化 velocity network，例如：

```text
Flow-G: gated velocity
Flow-T: decoded velocity
noise-augmented rollout
SAC objective + entropy regularization + twin Q
```

**适合本项目的原因**

- 当前环境可以大量并行采样，off-policy replay buffer 很有价值。
- 推箱、碰撞、射击失败这些稀有事件可以反复回放学习。
- 双车自博弈非平稳，但 replay + target Q 能缓冲训练震荡。

**局限**

- 需要自己实现 flow actor、Q critic、temperature、replay buffer。
- 多智能体 CTDE 版需要额外设计 joint critic。

### 2.6 PolicyFlow

**发行时间**

- arXiv 首次提交：2026-02-01。
- 论文：`PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning`。

**核心原理**

PolicyFlow 是 on-policy flow matching / continuous normalizing flow 策略优化。它针对 PPO 难以直接使用 CNF 策略的问题，提出不用完整 flow path likelihood 的近似重要性比率，并加入 Brownian Regularizer 防止模式崩溃。

可以理解为：

```text
PPO-style on-policy objective
+ CNF/flow policy
+ velocity field ratio approximation
+ Brownian regularizer for exploration/diversity
```

**适合本项目的原因**

- 它更像 PPO 的直接升级，便于从当前 MAPPO-style 代码迁移。
- 论文实验包含 IsaacLab 和 MuJoCo Playground，方向上与本项目接近。
- 对多峰行为特别友好，适合黄/蓝双方策略分化。

**局限**

- 公开时间很新，工程生态不如 PPO/SAC。
- 如果完全按论文实现，需要处理 flow velocity、ratio approximation、regularizer、action bounds。

### 2.7 DPPO 与 ReinFlow

**DPPO 发行时间**

- arXiv 首次提交：2024-09-01。
- 论文：`Diffusion Policy Policy Optimization`。

**ReinFlow 发行时间**

- arXiv 首次提交：2025-05-28。
- NeurIPS 2025。
- 论文：`ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning`。

**核心原理**

DPPO 是对 diffusion policy 做 policy gradient 微调；ReinFlow 是对 flow matching policy 做在线 RL 微调。它们都适合先用专家轨迹/历史数据训练一个 action-chunk 生成模型，再用在线 RL 提升成功率。

本项目可以让 action chunk 表示：

```text
未来 8-16 个控制步的战术动作序列
```

例如：

- 连续推箱 1.2 s；
- 接近靶子并保持瞄准；
- 遇碰撞后退、绕开、重定位；
- 从已拆挡板侧进入基地射击位。

**局限**

- 单步控制任务里 diffusion/flow chunk 可能过重。
- 最适合做第二阶段微调，而不是一开始就完全替换训练器。

## 3. 对本项目的最终技术选择

我推荐采用以下架构：

```text
Multi-Agent Object-Centric World Model
        +
PolicyFlow / SAC Flow Actor
        +
Centralized Twin Critic / Value Head
        +
Rule-aware Safety Shield
        +
IsaacLab Strict Replay Audit
```

中文名称建议：

```text
多智能体对象中心世界模型驱动的流策略强化学习
```

英文名称建议：

```text
MA-OCWM-FlowRL
Multi-Agent Object-Centric World-Model Flow Reinforcement Learning
```

### 为什么不是只用 DreamerV3

DreamerV3 很成熟，但它是通用 latent imagination 路线。本项目的关键难点不是像 Minecraft 那样从图像中发现世界，而是：

- 箱子必须可推且位置持续变化；
- 靶子不能嵌墙；
- 蓝色挡板必须阻挡车和激光；
- 基地靶必须在拆挡板侧射击；
- 激光 0.8 s 驻留和 20-80 cm 距离必须严格满足；
- 双方胜率要平衡。

这些都是对象与规则结构。对象中心世界模型比纯 DreamerV3 更直接。

### 为什么不是只用 TD-MPC2

TD-MPC2 的 latent planning 很适合推箱和长程规划，但策略表达仍需要强动作分布。如果只做 TD-MPC2 planner，可能会偏向短视局部最优。加入 PolicyFlow/SAC Flow actor 可以保留多模态战术选择。

### 为什么不是只用 SAC Flow

SAC Flow 是最适合直接替换 PPO 的算法，但它仍然是 model-free/off-policy。对于“打 1/2/3/4 个靶后基地成功率不同”这种长程收益，世界模型能更高效地做想象评估。

## 4. 具体改造计划

### 阶段 0：冻结基线与数据合同

目标：保证改算法时不破坏当前已修好的规则和几何。

需要固定：

- 当前 `robocup_visionrl_selfplay_env.py` 的 observation/action/reward/info 字段。
- 当前 strict replay audit 指标。
- 当前三视角视频生成流程。
- 当前 yellow/blue expert 策略。

新增文件：

```text
isaaclab_sim/rl/configs/world_model_flow.yaml
isaaclab_sim/rl/world_model/README.md
docs/rl_world_model_flow_policy_plan.md
```

验收：

- 旧 MAPPO checkpoint 仍可评估。
- 旧 replay/video pipeline 不变。
- `pytest tests -q` 通过。

### 阶段 1：采集世界模型数据集

目标：把环境交互变成可训练世界模型的数据。

新增脚本：

```text
isaaclab_sim/rl/collect_selfplay_dataset.py
```

每一步存储：

```text
obs_yellow
obs_blue
central_obs
action_yellow
action_blue
reward_yellow
reward_blue
done
info
object_state
```

`object_state` 必须显式记录：

- 两车 pose、速度、队伍；
- 每个靶子的 owner、xy、yaw、knocked；
- 红色箱子 xy、位移、是否被推动；
- 蓝色挡板 armor/blocker 状态；
- 激光目标、距离、横向误差、驻留时间；
- 碰撞、穿模、重定位、异常旋转；
- 当前比分和剩余时间。

数据来源：

- expert policy rollout；
- 当前 MAPPO policy rollout；
- random exploration；
- domain randomization rollout；
- 人工构造的推箱、会车、基地攻坚片段。

### 阶段 2：对象中心世界模型

新增模块：

```text
isaaclab_sim/rl/world_model/object_state.py
isaaclab_sim/rl/world_model/encoder.py
isaaclab_sim/rl/world_model/dynamics.py
isaaclab_sim/rl/world_model/losses.py
isaaclab_sim/rl/train_object_world_model.py
```

模型结构：

```text
ObjectEncoder:
  robot tokens
  target tokens
  pushable box tokens
  blocker tokens
  rule-state tokens

InteractionTransformer:
  object-object attention
  team-conditioned attention mask

DynamicsHead:
  next robot pose delta
  next box pose delta
  target knocked probability
  blocker removed probability
  collision probability
  reward prediction
  terminal prediction
```

损失函数：

```text
L_state_delta
L_box_motion
L_target_knock
L_laser_hit_probability
L_collision
L_reward
L_done
L_rule_violation
```

关键点：

- 靶子/箱子/挡板不要压成普通向量，必须作为 object tokens。
- 推箱必须预测持续位移，而不是一步接触后回弹。
- 激光命中概率要显式条件化：距离、横向误差、驻留时间、挡板遮挡、普通靶击倒数量。

验收：

- one-step prediction error；
- 10-step rollout box displacement error；
- target knock / base hit probability calibration；
- collision/penetration recall；
- imagined rollout 不产生明显穿模。

### 阶段 3A：PolicyFlow 作为 PPO/MAPPO 的直接升级

新增模块：

```text
isaaclab_sim/rl/policies/flow_policy.py
isaaclab_sim/rl/train_policyflow_selfplay.py
```

替换点：

当前 MAPPO actor：

```text
obs -> MLP -> Gaussian mean/std -> action
```

替换为：

```text
noise z + obs + team_id -> CNF velocity field -> action
```

保留：

- centralized critic；
- GAE / return；
- clipping 思路；
- entropy / Brownian regularizer；
- yellow/blue dual actor mode；
- residual expert 模式。

训练目标：

```text
PolicyFlow objective
+ centralized value loss
+ Brownian regularizer
+ residual expert L2
+ rule shield penalty
```

优点：

- 最像当前 PPO/MAPPO 的升级版。
- 易于做 ablation：Gaussian MAPPO vs PolicyFlow MAPPO。

### 阶段 3B：SAC Flow 作为 off-policy 强化版

新增模块：

```text
isaaclab_sim/rl/replay_buffer.py
isaaclab_sim/rl/policies/sac_flow_policy.py
isaaclab_sim/rl/train_sacflow_selfplay.py
```

结构：

```text
Flow actor:
  obs + noise -> action

Centralized twin Q:
  central_obs + joint_action -> Q1, Q2

Target Q:
  soft update

Temperature:
  automatic entropy tuning
```

多智能体处理：

```text
训练时 critic 看：
  obs_yellow, obs_blue, action_yellow, action_blue

执行时 actor 只看：
  local_obs_team
```

优点：

- replay buffer 能重复利用推箱/碰撞/射击失败样本。
- 对稀有事件学习比 on-policy PPO 更高效。

风险：

- self-play 下 replay buffer 非平稳，需要 opponent snapshot 或 league buffer。
- Q overestimation 需要 twin Q、target smoothing、gradient clipping、conservative regularization。

### 阶段 4：接入世界模型想象训练

有两条路线：

#### 4.1 TD-MPC2-style planner

用于高层战术：

```text
候选 target order / route / base rush timing
    -> world model rollout 5-20 steps
    -> 选 expected return 最高的 action chunk
```

适合：

- 推箱路径；
- 早攻基地窗口；
- 会车后恢复；
- 普通靶数量与基地成功率权衡。

#### 4.2 DreamerV3-style imagination actor-critic

用于策略学习：

```text
world model latent rollout
    -> imagined rewards
    -> train actor/critic inside latent space
```

适合：

- 大量想象训练；
- domain randomization；
- 低成本评估不同战术节奏。

我的建议是优先做 **TD-MPC2-style short horizon planner**，因为当前环境的规则结构明确，不需要先完全复刻 DreamerV3。

### 阶段 5：动作片段 Flow/Diffusion 微调

当单步 Flow actor 稳定后，引入 action chunk：

```text
obs history + task token -> action[0:H]
```

候选算法：

- DPPO：扩散策略在线 PG 微调；
- ReinFlow：flow matching policy 在线 RL 微调；
- SAC Flow：flow action chunk 的 off-policy 微调。

推荐 chunk 长度：

```text
H = 8 到 16
dt = 0.10 s
对应 0.8 s 到 1.6 s 动作片段
```

这正好覆盖：

- 激光 0.8 s 驻留；
- 推箱连续接触；
- 会车恢复；
- 进入基地射击位。

### 阶段 6：评估与视频闭环

必须保持用户当前要求的闭环：

```text
修改 -> 训练/续训 -> 评估 -> 三视角回放 -> 报告
```

新增评估：

```text
isaaclab_sim/rl/evaluate_world_model_policy.py
isaaclab_sim/rl/replay_world_model_policy_strict.py
```

指标：

- yellow_win_rate；
- blue_win_rate；
- draw_rate；
- average_score；
- average_match_time；
- normal target hit distribution；
- base hit success by normal-hit count；
- push count；
- final box displacement；
- penetration count；
- robot contact count；
- relocalization count；
- abnormal spin count；
- model rollout prediction error；
- imagined-vs-real reward gap。

三视角输出：

```text
docs/media/世界模型流策略_顶视角.mp4
docs/media/世界模型流策略_黄车第一视角.mp4
docs/media/世界模型流策略_蓝车第一视角.mp4
```

报告：

```text
docs/rl_world_model_flow_final_report.md
```

## 5. 建议的文件变更清单

建议新增：

```text
isaaclab_sim/rl/configs/world_model_flow.yaml
isaaclab_sim/rl/collect_selfplay_dataset.py
isaaclab_sim/rl/replay_buffer.py
isaaclab_sim/rl/world_model/__init__.py
isaaclab_sim/rl/world_model/object_state.py
isaaclab_sim/rl/world_model/encoder.py
isaaclab_sim/rl/world_model/dynamics.py
isaaclab_sim/rl/world_model/losses.py
isaaclab_sim/rl/policies/__init__.py
isaaclab_sim/rl/policies/flow_policy.py
isaaclab_sim/rl/policies/sac_flow_policy.py
isaaclab_sim/rl/train_object_world_model.py
isaaclab_sim/rl/train_policyflow_selfplay.py
isaaclab_sim/rl/train_sacflow_selfplay.py
isaaclab_sim/rl/evaluate_world_model_policy.py
isaaclab_sim/rl/replay_world_model_policy_strict.py
docs/rl_world_model_flow_policy_plan.md
docs/rl_world_model_flow_final_report.md
```

建议修改：

```text
isaaclab_sim/rl/export_policy.py
isaaclab_sim/rl/evaluate_strategy_contract.py
isaaclab_sim/rl/generate_rl_figures.py
isaaclab_sim/rl/README.md
README.md
```

## 6. 最小可行版本

如果要避免“一步到世界模型太大”，最小可行版本如下：

```text
MVP-1:
  PolicyFlow self-play 替换 Gaussian MAPPO actor

MVP-2:
  SAC Flow off-policy replay 版本

MVP-3:
  Object world model 只预测 5 个头：
    reward
    done
    box_delta
    target_knock
    collision

MVP-4:
  TD-MPC2-style 10-step latent planner

MVP-5:
  action chunk + ReinFlow/DPPO fine-tune
```

如果必须先选一个开始，我建议：

```text
先做 PolicyFlow self-play。
```

理由：

- 它最像 PPO/MAPPO 的直接升级；
- 论文已经包含 IsaacLab 实验；
- 能立即体现“前沿算法替换 PPO”；
- 之后可以无缝接入世界模型和 action chunk。

如果更重视训练效率而不是论文感：

```text
先做 SAC Flow self-play。
```

## 7. 风险与应对

| 风险 | 原因 | 应对 |
|---|---|---|
| 世界模型学会错误物理 | 推箱/挡板/靶子接触是稀有事件 | 构造专项数据集，增加碰撞/推箱 oversampling。 |
| Flow policy 训练不稳定 | flow path 长、梯度容易异常 | 先做短 flow steps，加入 gradient clipping 和 Brownian regularizer。 |
| Self-play 非平稳 | 对手策略不断变化 | 使用 opponent snapshot、league buffer、固定评估池。 |
| 策略投机利用世界模型漏洞 | 模型想象与真实 IsaacLab 不一致 | 每轮 imagined policy 必须回到真实 env strict replay 审计。 |
| 动作片段导致反应慢 | action chunk 太长 | 使用 receding horizon，只执行前 1-2 步，随后重规划。 |
| 规则被算法绕开 | 高级策略可能学到漏洞 | 保留 hard rule shield，禁止不可见靶、未拆挡板基地命中、穿箱穿挡板。 |

## 8. 最终结论

最有研究价值的改造不是“把 PPO 换成一个更强优化器”，而是把项目升级为：

```text
对象中心世界模型
+ 流/扩散式多峰策略
+ 多智能体集中训练分散执行
+ 规则安全盾
+ IsaacLab 严格回放审计
```

如果立刻开始实现，推荐顺序：

1. `PolicyFlow self-play`：替换当前 MAPPO 高斯 actor。
2. `SAC Flow self-play`：引入 replay buffer 和 off-policy 样本效率。
3. `Object-Centric World Model`：学习箱子、靶子、挡板、射击概率动力学。
4. `TD-MPC2-style latent planner`：在世界模型中做短时域战术搜索。
5. `DPPO/ReinFlow action chunk`：微调连续推箱、驻留射击、恢复动作片段。

## 9. 当前落地状态

已新增一版可运行 MVP，用来替换原先的 PPO/MAPPO 主训练入口：

```text
isaaclab_sim/rl/world_model/object_state.py
isaaclab_sim/rl/replay_buffer.py
isaaclab_sim/rl/policies/flow_policy.py
isaaclab_sim/rl/train_world_model_sacflow_selfplay.py
isaaclab_sim/rl/configs/world_model_flow.yaml
```

该版本已经完成：

- 对象中心状态抽取：机器人、靶子、红色可推动箱、基地装甲挡板、比分和接触状态。
- SAC Flow-style actor：用 velocity-reparameterized flow actor 替换旧 Gaussian actor。
- 集中式 twin-Q critic：训练时接收对象中心全局状态和 joint action。
- replay buffer：反复利用推箱、碰撞、射击失败、基地攻坚等稀有事件。
- 辅助世界模型：预测下一步 object state、reward 和 done，为后续 TD-MPC2/Dreamer-style imagined rollout 做数据接口准备。

当前 MVP 仍然是“一步真实交互 + 一步世界模型辅助损失”，尚未启用多步 imagined rollout。下一步应加入 TD-MPC2-style 短时域 latent planner。

## 10. 参考资料

- TD-MPC2: Scalable, Robust World Models for Continuous Control, arXiv:2310.16828, 2023-10-25, ICLR 2024 Spotlight. <https://arxiv.org/abs/2310.16828>
- TD-MPC2 project page. <https://www.tdmpc2.com/>
- DreamerV3: Mastering Diverse Domains through World Models, arXiv:2301.04104, 2023-01-10. <https://arxiv.org/abs/2301.04104>
- Newt: Learning Massively Multitask World Models for Continuous Control, arXiv:2511.19584, 2025-11-24, ICLR 2026. <https://arxiv.org/abs/2511.19584>
- Newt project page. <https://www.nicklashansen.com/NewtWM/>
- Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, arXiv:2303.04137, 2023-03-07. <https://arxiv.org/abs/2303.04137>
- DPPO: Diffusion Policy Policy Optimization, arXiv:2409.00588, 2024-09-01. <https://arxiv.org/abs/2409.00588>
- ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning, arXiv:2505.22094, 2025-05-28, NeurIPS 2025. <https://arxiv.org/abs/2505.22094>
- SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling, arXiv:2509.25756, 2025-09-30, revised 2026-01-14. <https://arxiv.org/abs/2509.25756>
- SAC Flow project page. <https://sac-flow.github.io/>
- PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning, arXiv:2602.01156, 2026-02-01. <https://arxiv.org/abs/2602.01156>
