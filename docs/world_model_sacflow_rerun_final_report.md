# 世界模型 SAC Flow 续训与三视角回放报告

日期：2026-05-07

## 2026-05-07 微瞄准与回放刷新

- 新增基地/普通靶 fire pose 微扫：小车到达合法射击点后不再完全静止，而是在安全命中余量内做慢速小角度扫描，提高 0.8 秒驻留期间的命中稳定性。
- 新增 2 靶及以上基地攻坚的厘米级径向/侧向候选点；1 靶早攻仍保持严格开口侧约束，避免从非法侧投机命中基地。
- 若当前点位已经满足开火几何，策略保持当前点位，不再追逐附近几厘米的“更优候选点”，减少基地附近来回抖动和卡住。
- 重新生成 strict replay trace，并重新录制三视角 IsaacLab MP4 与 README GIF：
  - `docs/media/最终回放_顶视角.mp4`
  - `docs/media/最终回放_黄车第一视角.mp4`
  - `docs/media/最终回放_蓝车第一视角.mp4`
- 最新 8 局 strict replay：yellow_win_rate=37.50%，blue_win_rate=62.50%，draw_or_timeout_rate=0.00%，hard_violations=0，warnings=0，robot_contacts_per_episode=0.00，base_wins_per_episode=1.00。
- 128 局 stochastic contract eval 仍作为当前胜率主指标：yellow_win_rate=49.22%，blue_win_rate=50.78%，draw_rate=0.00%，static_penetrations_total=0，box_penetrations_total=0，robot_contacts_per_episode=0.00。

## 本轮改动

- 继续使用正式主线：对象中心世界模型 + SAC Flow / PolicyFlow 自博弈。
- 接管并完成 `world_model_sacflow_seed260707_rerun` 训练，训练步数 `200000`。
- 为 IsaacLab wrapper 增加 `top` 录制视角，便于生成真正顶视角回放。
- 重新生成三视角中文命名 MP4：顶视角、黄车第一视角、蓝车第一视角。

## 为什么改

- 旧 PPO/MAPPO 只能作为历史 baseline；本轮结果必须来自新 SAC Flow checkpoint。
- 顶视角、第一视角回放需要直接使用严格 replay trace，不能复用旧视频。
- 训练后的胜率、推箱、基地命中、穿模和碰撞必须用统一规则环境审计。

## 训练结果

- checkpoint：
  `isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun/policy.pt`
- 训练曲线：
  `isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun/training_curve.csv`
- 训练摘要：
  `isaaclab_sim/output/rl/world_model_sacflow_seed260707_rerun/training_summary.json`
- 配置要点：
  - `num_envs=32`
  - `batch_size=1024`
  - `gradient_steps=2`
  - `actor_mode=dual`
  - `policy_mode=residual_expert`
  - `residual_scale=0.04`
  - `domain_randomization=true`
  - `action_shield=true`

## 测试与评估

- 测试：
  `pytest tests -q`
- 结果：
  `80 passed`

64 局多 seed contract eval：

- JSON：
  `isaaclab_sim/output/eval/world_model_sacflow_seed260707_rerun_contract_eval64.json`
- CSV：
  `isaaclab_sim/output/eval/world_model_sacflow_seed260707_rerun_contract_eval64.csv`
- 普通 eval JSON：
  `isaaclab_sim/output/eval/world_model_sacflow_seed260707_rerun_eval64.json`

关键指标：

| 指标 | 数值 |
|---|---:|
| yellow_win_rate | 42.19% |
| blue_win_rate | 57.81% |
| draw_rate | 0.00% |
| 平均比赛时间 | 30.29 s |
| 黄方平均得分 | 36.72 |
| 蓝方平均得分 | 45.70 |
| 黄方平均普通靶击倒数 | 2.28 |
| 蓝方平均普通靶击倒数 | 2.20 |
| 静态穿模总数 | 0 |
| 箱子穿模总数 | 0 |
| 机器人相撞 / 局 | 0.00 |
| 重定位 / 局 | 0.00 |
| 异常旋转 / 局 | 0.00 |

普通靶击倒数分布：

| 队伍 | 1 个 | 2 个 | 3 个 | 4 个 |
|---|---:|---:|---:|---:|
| yellow | 1.56% | 68.75% | 29.69% | 0.00% |
| blue | 3.12% | 73.44% | 23.44% | 0.00% |

基地命中率分组：

| 队伍 | 1 靶后 | 2 靶后 | 3 靶后 | 4 靶后 |
|---|---:|---:|---:|---:|
| yellow | 0.00% | 31.75% | 36.84% | 0.00% |
| blue | 0.00% | 46.77% | 53.33% | 0.00% |

推箱指标：

| 指标 | 数值 |
|---|---:|
| yellow 推箱事件 / 局 | 2.0938 |
| blue 推箱事件 / 局 | 0.2812 |
| box_ne 平均最终位移 | 0.1193 m |
| box_sw 平均最终位移 | 0.0169 m |

## 严格回放审计

- 输出目录：
  `isaaclab_sim/output/replay/world_model_sacflow_seed260707_rerun_strict8/`
- summary：
  `isaaclab_sim/output/replay/world_model_sacflow_seed260707_rerun_strict8/strict_replay_summary.json`
- trace：
  `isaaclab_sim/output/replay/world_model_sacflow_seed260707_rerun_strict8/strict_replay_trace.csv`
- events：
  `isaaclab_sim/output/replay/world_model_sacflow_seed260707_rerun_strict8/strict_replay_events.jsonl`

8 局严格审计结果：

- yellow_win_rate：37.50%
- blue_win_rate：62.50%
- draw_or_timeout_rate：0.00%
- hard_violations：0
- warnings：0
- own_target_penalties_per_episode：0.00
- robot_contacts_per_episode：0.00
- recovery_events_per_episode：0.00
- base_wins_per_episode：1.00

## 三视角回放

本轮使用严格审计 episode 0 生成视频。该 episode 时长约 `33.2 s`，有普通靶命中、基地护甲移除、推箱事件和最终基地命中。

- 顶视角：
  `docs/media/最终回放_顶视角.mp4`
- 黄车第一视角：
  `docs/media/最终回放_黄车第一视角.mp4`
- 蓝车第一视角：
  `docs/media/最终回放_蓝车第一视角.mp4`
- 视频检查摘要：
  `isaaclab_sim/output/replay/world_model_sacflow_seed260707_rerun_strict8/video_check_summary.json`

视频检查结论：

- 三个 MP4 均可打开，分辨率 `1600x900`，帧率 `30 fps`。
- 抽帧检查不是黑屏。
- 顶视角能看到两辆车、红色实体箱、靶子和蓝色基地挡板。
- 日志显示红色箱子被推动后位置更新。
- 日志显示普通靶击倒后才进入基地命中阶段。
- 严格 replay 审计为 0 hard violation、0 warning。

## 剩余风险

- 胜率还没有完全进入 45%-55% 目标区间：本轮黄方 42.19%，蓝方 57.81%，蓝方略强。
- 两队主要集中在击倒 2 或 3 个普通靶后攻基地，4 靶路线占比仍为 0，需要继续提高多路线策略分布。
- blue 推箱事件明显少于 yellow，说明蓝方虽然胜率高，但推箱路线学习还不充分。
- 当前三视角视频通过抽帧和日志检查，但仍建议人工完整播放确认小车没有视觉上轻微陷入箱子或挡板。

## 下一步建议

- 轻微增强 yellow expert 的早攻基地窗口或路线效率，目标把 yellow_win_rate 拉回 45% 以上。
- 增加 4 靶后攻基地的奖励塑形或课程采样，避免策略长期集中在 2/3 靶。
- 给 blue 增加推箱路线探索奖励，使双方都有可见的推箱节目。
- 再跑一轮 64 局评估和三视角回放，确认胜率回到目标区间。
