# Object-Centric World-Model Flow RL 项目深度说明

本文档用于完整介绍 `Object-Centric World-Model Flow RL for Multi-Agent Robotics` 项目的工程设计、规则建模、ROS2 运行栈、IsaacLab 仿真、对象中心世界模型、自博弈训练、评估审计、三视角回放和 Sim2Real 复现路径。

项目仓库不是一个单点算法 demo，而是一套围绕 RoboCup 风格视觉对抗任务构建的端到端机器人学习系统。它同时覆盖：

- 两车多智能体对抗策略学习
- 视觉靶识别与规则约束射击
- 动态可推动红色障碍箱
- 蓝色基地挡板与基地靶遮挡逻辑
- ROS2/Nav2 真实部署接口
- IsaacLab 可视化回放
- 严格规则审计与多 seed 统计评估
- 面向论文/作品集展示的图表、GIF 和结果文档

## 1. 项目一句话概括

该项目使用“对象中心世界模型 + SAC Flow/PolicyFlow 风格策略 + 规则专家残差 + 多智能体自博弈”的方法，训练两台差速机器人在 3m x 3m RoboCup 视觉挑战赛场中完成靶子选择、推箱路线、基地攻坚、早攻窗口判断、碰撞规避和合法射击。

系统核心思想是：不要只让策略看一串扁平状态，而是显式建模机器人、靶子、基地挡板、红色箱子、分数、装甲状态、激光视线和比赛时间等对象；策略层负责高层战术，ROS2/Nav2/行为节点负责低层导航、安全门和真实机器人接口。

## 2. 项目目标

项目目标分成四层。

第一层是比赛规则可信：

- 靶子不能嵌墙或夹墙。
- 普通靶必须约 45 度朝向相邻墙面。
- 红色箱子必须是真实可推动物体。
- 蓝色基地挡板必须真实阻挡车辆和激光视线。
- 基地靶必须在对应挡板被移除后才可能被击中。
- 激光命中必须满足距离、视线、驻留时间和概率门。
- 小车不能穿墙、穿箱、穿挡板或穿过出发区隔板。

第二层是训练结果可信：

- 不能只看 reward。
- 必须看胜率、普通靶击倒数量、基地命中分组、推箱次数、箱子位移、穿模次数、碰撞次数、异常旋转次数和严格回放审计。
- 正式结果必须来自完整比赛 rollout，而不是截取短片段。

第三层是系统工程可信：

- ROS2 运行栈和 IsaacLab/RL 环境共享同一套规则口径。
- 训练、评估、回放和 README 图表都能追溯到同一批结果文件。
- 本地训练输出放在 `isaaclab_sim/output/`，仓库只保留精选的最终结果和可复现脚本。

第四层是展示可信：

- README 第一张项目媒体是最终俯视 GIF。
- 三视角回放同时提供 GIF 和 MP4。
- 方法图和实验图使用当前对象中心世界模型架构，不混入旧训练口径。
- PowerPoint master 与 PNG 图同步生成，便于论文或答辩使用。

## 3. 仓库结构总览

| 路径 | 作用 |
|---|---|
| `README.md` | 项目首页，展示最终三视角回放、方法图、结果表和复现入口。 |
| `config/arena_rules.yaml` | 公共赛场、机器人、比赛时长、传感器和安全规则契约。 |
| `config/target_layout.yaml` | 普通靶、基地靶、装甲板数量和几何位置。 |
| `config/scoring.yaml` | 得分规则、命中事件、奖励模型和终局逻辑。 |
| `crc_robocup_vision_ws/` | ROS2 Jazzy 工作空间，包含真实机器人运行包。 |
| `isaaclab_sim/` | IsaacLab 场景、仿真回放和 RL 接口。 |
| `isaaclab_sim/rl/` | 规则环境、自博弈环境、世界模型训练、评估、导出与图表生成。 |
| `docs/` | 架构、规则、Sim2Real、策略、复现、训练计划和结果文档。 |
| `docs/figures/paper/` | 顶会风格 PNG 方法图和可编辑 PPTX master。 |
| `docs/figures/rl/` | 由结果数据生成的训练/评估 SVG 图。 |
| `docs/media/` | 最终中文命名三视角 GIF/MP4 回放。 |
| `docs/rl_data/` | 精选后的最终评估、训练曲线、规则审计和回放审计数据。 |
| `assets/readme/` | README 中使用的机器人、赛场和 ROS2 结构图。 |
| `scripts/` | 训练、IsaacLab 启动、GIF 制作、图表生成和自动提交脚本。 |
| `tests/` | pytest 测试，覆盖规则环境、目标配置、策略契约和 Sim2Real 配置。 |

## 4. 任务与比赛规则建模

### 4.1 赛场

赛场参数来自 `config/arena_rules.yaml`：

- 坐标系：`map`
- 场地大小：`3.0 m x 3.0 m`
- 墙高：`0.5 m`
- 墙厚：`0.04 m`
- 黄/蓝出发区：各 `0.5 m x 0.5 m`
- 黄/蓝基地：各 `0.5 m x 0.5 m`
- 红色障碍箱尺寸：`0.3 m x 0.3 m x 0.3 m`
- 比赛时长：`180 s`
- 无进展保护：`20 s`
- 自己靶射击：禁止
- 自己基地被击中：终局失败

机器人使用差速驱动。规则配置中的机器人包络尺寸为：

- 长宽高：`0.34 m x 0.24 m x 0.245 m`
- 黄色起点：`[0.25, -1.25, 1.5708]`
- 蓝色起点：`[-0.25, 1.25, -1.5708]`

### 4.2 靶子

靶子配置来自 `config/target_layout.yaml`。

普通靶：

| 名称 | 所属方 | 坐标 | 朝向 |
|---|---|---|---|
| `T01_NorthMiddle` | blue | `[0.18, 1.26]` | `-45 deg` |
| `T02_NorthEast` | blue | `[1.26, 1.26]` | `-135 deg` |
| `T03_WestAboveGate` | blue | `[-1.26, 0.24]` | `45 deg` |
| `T04_WestBelowGate` | yellow | `[-1.26, -0.24]` | `-45 deg` |
| `T05_EastAboveGate` | blue | `[1.26, 0.24]` | `135 deg` |
| `T06_EastBelowGate` | yellow | `[1.26, -0.24]` | `-135 deg` |
| `T07_SouthWest` | yellow | `[-1.26, -1.26]` | `45 deg` |
| `T08_SouthMiddle` | yellow | `[-0.18, -1.26]` | `135 deg` |

基地靶：

- 蓝色基地靶：`BlueBaseTarget`，坐标 `[-1.36, 1.36]`，朝向 `-45 deg`。
- 黄色基地靶：`YellowBaseTarget`，坐标 `[1.36, -1.36]`，朝向 `135 deg`。
- 每个基地有 4 块装甲挡板。
- 普通靶击中后，移除对方基地的一块装甲。
- 基地靶被合法击中后，射击方获胜。

### 4.3 得分与事件

得分配置来自 `config/scoring.yaml`：

- 击中对方普通靶：`+5` 分。
- 击中对方基地靶：终局获胜。
- 击中己方基地靶：终局失败。
- 时间耗尽：高分方获胜。
- 碰撞导致靶倒：按规则奖励给非接触方。

训练奖励模型中，普通靶、基地靶、非法射击、运动阻塞、有效重定位和时间消耗都有独立权重。正式评估不只依赖 reward，而是同时检查规则事件和回放审计。

## 5. ROS2 工程架构

ROS2 工作空间位于 `crc_robocup_vision_ws/`，目标平台为 Ubuntu 24.04 + ROS2 Jazzy。

### 5.1 ROS2 包职责

| 包 | 职责 |
|---|---|
| `rcvrl_bringup` | 启动完整比赛系统，统一加载 robot description、Nav2、EKF、视觉、发射器和行为节点。 |
| `rcvrl_description` | 发布机器人 URDF/Xacro、TF 和基础几何。 |
| `rcvrl_navigation` | 保存 Nav2、地图、SLAM 和目标路线配置。 |
| `rcvrl_vision` | 基于 AprilTag Tag36h11 检测视觉靶。 |
| `rcvrl_behavior` | 比赛状态机、规则门、目标选择、导航调度和发射器调用。 |
| `rcvrl_shooter` | 固定低功率激光模块控制，支持 dry-run。 |
| `rcvrl_motion` | 运动漂移记录与仿真数据采集。 |
| `rcvrl_interfaces` | 自定义消息定义，例如 `TargetDetection`。 |
| `rcvrl_docs` | ROS2 工作空间内的文档包。 |

### 5.2 启动入口

主启动文件：

```text
crc_robocup_vision_ws/src/rcvrl_bringup/launch/competition.launch.py
```

它会按需启动：

- `rcvrl_description`
- `rcvrl_navigation`
- `robot_localization/ekf_node`
- `rcvrl_motion/motion_drift_recorder`
- `rcvrl_shooter/shooter_controller`
- `rcvrl_vision/apriltag_detector`
- `rcvrl_behavior/competition_behavior`

关键 launch 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_sim_time` | `false` | 是否使用仿真时间。 |
| `start_navigation` | `true` | 是否启动 Nav2。 |
| `start_sensor_fusion` | `true` | 是否启动 EKF。 |
| `start_motion_drift_recorder` | `false` | 是否记录运动漂移。 |
| `shooter_dry_run` | `false` | 发射器 dry-run 模式。 |
| `auto_start` | `true` | 行为节点是否自动开始。 |
| `team_color` | `yellow` | 当前机器人队伍颜色。 |
| `target_file` | `auto` | 自动选择黄/蓝路线文件。 |

### 5.3 ROS2 运行图

核心运行链路：

```text
/camera/image_raw + /camera/camera_info
        |
        v
rcvrl_vision/apriltag_detector
        |
        v
/target_detection  --------------------+
                                       |
Nav2 /navigate_to_pose action <--- rcvrl_behavior/competition_behavior
                                       |
/cmd_vel ------------------------------+
                                       |
/shooter/enable / /fire / /disable <---+
        |
        v
rcvrl_shooter/shooter_controller -> serial laser module
```

### 5.4 关键话题、服务和动作

| 名称 | 类型 | 用途 |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 视觉靶检测输入。 |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | 相机内参。 |
| `/depth/image_raw` | `sensor_msgs/Image` | 近距离深度辅助。 |
| `/scan` | `sensor_msgs/LaserScan` | 建图、定位和 Nav2 costmap。 |
| `/imu/data_raw` | `sensor_msgs/Imu` | EKF 和碰撞冲击检测。 |
| `/wheel/odom` | `nav_msgs/Odometry` | 轮速里程计。 |
| `/odometry/filtered` | `nav_msgs/Odometry` | EKF 融合输出。 |
| `/range/front_left` / `/range/front_right` | `sensor_msgs/Range` | 前向 ToF 避障。 |
| `/bumper/front_left` / `/bumper/front_right` | `std_msgs/Bool` | 碰撞触发。 |
| `/target_detection` | `rcvrl_interfaces/TargetDetection` | 结构化靶子观测。 |
| `/cmd_vel` | `geometry_msgs/Twist` | 差速底盘速度控制。 |
| `/shooter/enable` | `std_srvs/Trigger` | 启用激光。 |
| `/shooter/fire` | `std_srvs/Trigger` | 触发发射。 |
| `/shooter/disable` | `std_srvs/Trigger` | 关闭激光。 |
| `navigate_to_pose` | `nav2_msgs/NavigateToPose` | Nav2 目标点动作。 |

自定义检测消息 `TargetDetection.msg` 包含：

```text
std_msgs/Header header
int32 tag_id
string target_type
geometry_msgs/PoseStamped pose
float64 center_x
float64 distance_z
float64 confidence
bool aligned
```

### 5.5 行为状态机

`rcvrl_behavior/src/competition_behavior.cpp` 实现比赛状态机：

```text
INIT
  -> NAVIGATE
  -> SEARCH
  -> ALIGN
  -> FIRE
  -> NEXT_TARGET
  -> RETURN_HOME
  -> END
```

异常路径：

```text
NAVIGATE / SEARCH / ALIGN
  -> RECOVER_LOCALIZATION
  -> NAVIGATE
```

核心规则门：

- 目标所属方必须与自身队伍相反。
- 自己靶和自己基地检测会在发射前被拒绝。
- 靶子数组中的 `target_owner` 与 `team_color` 一致时会被跳过。
- 定位置信度低、IMU 冲击、bumper 触发、长时间无进展都会进入重定位恢复。
- 重定位恢复通过原地旋转刷新激光雷达、IMU、相机和里程计一致性。

## 6. Sim2Real 设计

项目并不把仿真全状态直接部署到真实机器人，而是将可迁移层定义为 ROS2 契约：

- `/cmd_vel`
- `/target_detection`
- Nav2 目标点
- 发射器服务
- TF 坐标
- EKF 融合里程计
- 对手靶安全门

### 6.1 真实机器人参数

`crc_robocup_vision_ws/src/rcvrl_bringup/config/sim2real.yaml` 中定义：

- 机器人包络：`[0.34, 0.24, 0.245]`
- 轮半径：`0.035 m`
- 轮距：`0.205 m`
- 最大线速度：`0.30 m/s`
- 最大角速度：`1.20 rad/s`
- 线加速度限制：`0.55 m/s^2`
- 角加速度限制：`2.0 rad/s^2`
- 电机死区：`18 PWM`

### 6.2 传感器栈

真实机器人与仿真都围绕以下传感器建模：

- 轮编码器
- IMU
- 2D 激光雷达
- RGB 相机
- 深度相机
- 前向 ToF
- bumper 接触开关
- 固定低功率激光模块

关键 TF frame：

- `base_link`
- `imu_link`
- `laser_link`
- `camera_link`
- `depth_camera_link`
- `front_tof_left_link`
- `front_tof_right_link`

### 6.3 相机与发射器标定

视觉部分：

- AprilTag family：`Tag36h11`
- tag 尺寸：`0.05 m`
- tag 底部高度：约 `0.07 m`
- 需要锁定曝光，避免比赛光照导致检测漂移。

发射器部分：

- 服务：`/shooter/enable`、`/shooter/fire`、`/shooter/disable`
- 发射延迟：`80 ms`
- 光束相对相机偏移：`y=-0.015 m`、`z=-0.035 m`
- 命中半径：`0.035 m`

### 6.4 域随机化

为了让策略更接近真实环境，训练配置中启用域随机化，覆盖：

- 光照：`600-1200 lux`
- 相机延迟：`30-120 ms`
- 执行延迟：`40-160 ms`
- 轮胎打滑：`0.00-0.18`
- 摩擦系数：`0.65-1.05`
- tag 角度：`30/45/60 deg`
- 障碍位置扰动：`0.03 m`
- 靶子位置扰动：`0.02 m`

## 7. IsaacLab 与规则环境

项目同时使用三类仿真/规则层。

### 7.1 快速规则环境

`isaaclab_sim/rl/robocup_visionrl_gym_env.py` 是单车规则 smoke 环境，用于快速检查：

- 靶子布局
- 激光命中范围
- 普通靶/基地靶规则
- 推箱碰撞
- 墙体/挡板阻挡
- 基础奖励逻辑

它不是最终正式训练入口，而是快速合同测试和规则调试工具。

### 7.2 两车自博弈环境

`isaaclab_sim/rl/robocup_visionrl_selfplay_env.py` 是正式两车规则环境，包含：

- 黄/蓝双智能体
- 每车 46 维局部观测
- 每车 6 维高层战术动作
- 普通靶与基地靶
- 动态可推动红色箱子
- 基地装甲挡板
- 激光射线阻挡
- 0.80 秒驻留门
- 距离和角度命中概率
- 机器人接触与分离
- 定位置信度与恢复
- 目标重复攻击检测
- 穿模审计字段

`isaaclab_sim/rl/robocup_visionrl_selfplay_vec.py` 提供并行环境，用于训练时批量收集自博弈 transition。

### 7.3 IsaacLab 可视化场景

`isaaclab_sim/robocup_visionrl_arena_sim.py` 负责 IsaacLab/Isaac Sim 侧的完整可视化场景：

- 场地、墙体、隔板、基地区
- 两辆差速机器人
- 普通靶和基地靶
- 蓝色基地挡板
- 红色可推动箱子
- 激光射线、命中、倒靶和装甲移除
- 顶视角、黄车第一视角、蓝车第一视角录制

正式视频不是手工剪辑，而是从严格 replay trace 渲染而来。

## 8. 物理与射击规则细节

### 8.1 红色箱子

红色箱子不是贴图或装饰，它在规则环境和 IsaacLab 回放中都作为可推动障碍建模：

- 初始参考位置：`box_ne = [0.8, 0.8]`，`box_sw = [-0.8, -0.8]`
- 尺寸：`0.3 m`
- 碰撞半径与机器人视觉包络一致
- 推动后位置必须持久更新
- 机器人不能穿箱
- 推箱事件和最终位移进入评估指标

### 8.2 基地挡板

基地挡板是合法基地攻击的关键：

- 每个基地有 4 块装甲。
- 普通靶命中后按顺序移除对方装甲。
- 未移除的装甲同时阻挡车辆和激光。
- 基地靶必须从已打开的合法侧窗口攻击。
- 早攻基地不允许从任意方向投机命中。

### 8.3 激光命中

激光规则被建模为多条件门：

- 靶子必须属于对方。
- 目标必须未被击倒。
- 发射口到靶面必须在合法距离内。
- 视线不能被墙体、箱子或装甲挡板阻挡。
- 角度误差和横向误差必须在命中半径内。
- 激光驻留不足 `0.80 s` 时命中概率为 0。
- `0.80 s` 到 `2.00 s` 之间命中概率随驻留时间增加。
- 基地靶还有按已击倒普通靶数量分组的成功率上限。

距离门：

| 靶子类型 | 发射口到靶面距离 |
|---|---|
| 普通靶 | `0.05 m` 到 `0.50 m` |
| 内凹基地靶 | `0.20 m` 到 `0.80 m` |

### 8.4 微调瞄准

策略层加入了安全微调：

- 到达合法射击点后不会完全静止。
- 普通靶和基地靶射击姿态会进行小角度慢速扫描。
- 基地附近增加厘米级侧向/径向候选点。
- 如果当前位置已经满足开火几何，就保持当前位置，不再反复追逐相邻候选点。
- 目标是减少“小车到点但差一点打不中”或“基地附近来回抖动”的问题。

## 9. 对象中心世界模型

对象中心状态由 `isaaclab_sim/rl/world_model/object_state.py` 构造。

总维度：

```text
OBJECT_STATE_DIM = 165
```

它由五类 token 拼接而成。

### 9.1 全局特征

全局特征维度为 9：

- 比赛已用时间比例
- 黄蓝分差
- 黄方得分
- 蓝方得分
- 黄方剩余装甲比例
- 蓝方剩余装甲比例
- 两车距离
- 最近是否接触
- 是否已经终局

### 9.2 机器人特征

每车 8 维：

- 归一化 x
- 归一化 y
- `cos(yaw)`
- `sin(yaw)`
- 自身装甲比例
- 已击倒对方普通靶比例
- 定位置信度
- 是否接触可推动箱子

### 9.3 靶子特征

最多 10 个靶子，每个 9 维：

- 归一化 x
- 归一化 y
- `cos(yaw)`
- `sin(yaw)`
- 所属方编码
- 靶子类型编码
- 是否已击倒
- 从黄方看是否被阻挡
- 从蓝方看是否被阻挡

### 9.4 箱子特征

最多 2 个箱子，每个 5 维：

- 归一化 x
- 归一化 y
- 半宽
- 半高
- 是否存在

### 9.5 装甲挡板特征

最多 8 块挡板，每块 5 维：

- 归一化 x
- 归一化 y
- 半宽
- 半高
- 是否仍然有效

这种对象中心状态让 critic 和世界模型直接知道“哪个箱子被推了”“哪些挡板还在”“哪一侧基地窗口打开了”“普通靶还剩几个”，而不是被迫从扁平观测中隐式推断。

## 10. 策略动作空间

每个机器人输出 6 维高层战术动作：

| 动作维度 | 含义 |
|---|---|
| `target_selector` | 在可见/可达的对方靶中选择攻击目标。 |
| `base_rush_gate` | 控制何时从清普通靶切换到攻基地。 |
| `block_interference_gate` | 判断是否需要干扰、阻挡或压迫对方路线。 |
| `recovery_gate` | 在定位置信度低时请求恢复。 |
| `fire_gate` | 控制是否保持激光开火以满足驻留门。 |
| `risk_preference` | 在近距离、推箱、绕路、早攻之间调节风险偏好。 |

动作会被裁剪到 `[-1, 1]`，再通过规则专家残差和 action shield 映射为实际环境动作。

## 11. SAC Flow/PolicyFlow 风格自博弈算法

正式训练入口：

```text
isaaclab_sim/rl/train_world_model_sacflow_selfplay.py
```

核心组件来自：

```text
isaaclab_sim/rl/policies/flow_policy.py
isaaclab_sim/rl/replay_buffer.py
isaaclab_sim/rl/world_model/object_state.py
```

### 11.1 FlowActor

`FlowActor` 使用速度重参数化 flow actor：

1. 先由 MLP 根据局部观测输出高斯 base distribution 的均值和方差。
2. 从 base distribution 采样 raw action。
3. 将 raw action 与时间特征一起输入 velocity field。
4. 经过若干 flow steps 后得到 flowed raw action。
5. 用 `tanh` squash 到 `[-1, 1]`。
6. 使用 base log-probability 加上 tanh 修正近似 SAC 的熵项。

当前配置：

- `flow_steps = 3`
- `flow_velocity_scale = 0.20`
- `hidden_dim = 256`

### 11.2 双 actor

训练使用 `actor_mode=dual`：

- 黄车有独立 actor。
- 蓝车有独立 actor。
- 两车共享同一套环境与规则，但可以学习不同节奏。

这对本任务很重要，因为黄方和蓝方在地图位置、靶子顺序、侧门角度、推箱机会和基地攻坚窗口上并不完全对称。

### 11.3 中心化 Twin-Q critic

`CentralizedTwinQ` 在训练时接收：

- 对象中心状态
- 两车局部观测
- 两车动作

输出：

- 每个智能体的 Q1
- 每个智能体的 Q2

Twin-Q 用于降低过估计。执行时不需要中心化全状态，导出的是本地高层 actor 与规则门结合的策略。

### 11.4 辅助世界模型

`ObjectWorldModel` 学习：

- 下一步对象中心状态
- 每个智能体 reward
- 每个智能体 done 概率

训练损失包括：

- 下一状态 Smooth L1 loss
- reward MSE loss
- done BCE loss

当前权重：

```text
total = state_loss + 0.40 * reward_loss + 0.20 * done_loss
weighted_total = world_model_coef * total
world_model_coef = 0.25
```

当前世界模型主要作为辅助表征与动态学习模块；后续可以扩展为短时域想象 rollout。

### 11.5 规则专家残差

`isaaclab_sim/rl/expert_policy.py` 定义了黄车和蓝车的专家 profile。

黄车专家倾向：

- 开局优先 `T01_NorthMiddle`
- 然后走 `T03_WestAboveGate`
- 根据机会选择 `T05_EastAboveGate` 或基地窗口
- 基地攻坚和推箱风险略保守

蓝车专家倾向：

- 开局优先 `T08_SouthMiddle`
- 然后走 `T06_EastBelowGate`
- 根据机会选择 `T04_WestBelowGate` 或基地窗口
- 推箱与节奏风险略高

学习策略不从零开始控制底层导航，而是对专家高层动作做残差修正：

```text
final_action = expert_action + residual_scale * learned_residual
residual_scale = 0.04
```

这样做有三个目的：

- 保留对手靶安全门和基本比赛常识。
- 让学习集中在靶子顺序、基地窗口、推箱、干扰和微调。
- 避免在坏探索早期生成大量非法样本。

### 11.6 Replay buffer

`MultiAgentReplayBuffer` 存储：

- 当前局部观测
- 当前对象中心状态
- 两车动作
- 两车 reward
- 下一局部观测
- 下一对象中心状态
- 两车 done

训练使用 off-policy 更新。

### 11.7 训练循环

训练流程：

1. 创建 `RoboCupVisionRLSelfPlayVector` 并行环境。
2. reset 得到初始局部观测和对象中心状态。
3. learning starts 前使用随机动作探索。
4. learning starts 后由 flow actor 采样动作。
5. 动作经过规则专家残差映射到环境动作。
6. 并行环境 step。
7. 将 transition 写入 replay buffer。
8. 对完成 episode 的环境单独 reset。
9. 从 replay buffer 采样 batch。
10. 更新 critic。
11. 更新 actor。
12. 更新温度 alpha。
13. 更新世界模型。
14. soft update target critic。
15. 写入 `training_curve.csv`。
16. 训练结束保存 `policy.pt` 和 `training_summary.json`。

## 12. 当前训练配置

配置文件：

```text
isaaclab_sim/rl/configs/world_model_flow.yaml
```

主要参数：

| 参数 | 当前值 |
|---|---:|
| `algorithm` | `object_centric_world_model_sac_flow_selfplay` |
| `seed` | `260707` |
| `num_envs` | `32` |
| `timesteps` | `200000` |
| `hidden_dim` | `256` |
| `batch_size` | `1024` |
| `replay_size` | `300000` |
| `learning_starts` | `4096` |
| `gradient_steps` | `2` |
| `gamma` | `0.995` |
| `tau` | `0.01` |
| `actor_lr` | `0.0003` |
| `critic_lr` | `0.0003` |
| `world_model_lr` | `0.0003` |
| `alpha_lr` | `0.0003` |
| `target_entropy` | `-6.0` |
| `max_grad_norm` | `1.0` |
| `flow_steps` | `3` |
| `flow_velocity_scale` | `0.20` |
| `actor_mode` | `dual` |
| `policy_mode` | `residual_expert` |
| `residual_scale` | `0.04` |
| `domain_randomization` | `true` |
| `action_shield` | `true` |
| `world_model_coef` | `0.25` |

最终镜像训练摘要位于：

```text
docs/rl_data/world_model_sacflow_final/training_summary.json
```

其中记录：

- 观测维度：`46`
- 对象中心状态维度：`165`
- 动作维度：`6`
- agents：`yellow`、`blue`
- 设备：`cuda`
- 训练 wall time：约 `3847 s`

## 13. 评估体系

正式评估不是单一指标，而是多层审计。

### 13.1 多 seed 合同评估

数据文件：

```text
docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.json
docs/rl_data/world_model_sacflow_final/contract_eval_multiseed.csv
```

当前 128 局汇总：

| 指标 | 数值 |
|---|---:|
| episodes | 128 |
| yellow win rate | 49.22% |
| blue win rate | 50.78% |
| draw rate | 0.00% |
| mean episode time | 30.8148 s |
| mean yellow score | 40.8984 |
| mean blue score | 41.7188 |
| mean normal hits yellow | 2.2734 |
| mean normal hits blue | 2.2500 |
| static penetrations | 0 |
| box penetrations | 0 |
| robot contacts per episode | 0.00 |
| repeat target order events | 0 |

普通靶击倒数量分布：

| 队伍 | 1 个 | 2 个 | 3 个 | 4 个 |
|---|---:|---:|---:|---:|
| yellow | 0.78% | 71.88% | 26.56% | 0.78% |
| blue | 2.34% | 70.31% | 27.34% | 0.00% |

基地命中率按已击倒普通靶数量分组：

| 队伍 | 1 个后 | 2 个后 | 3 个后 | 4 个后 |
|---|---:|---:|---:|---:|
| yellow | 0.00% | 36.22% | 45.71% | 100.00% |
| blue | 0.00% | 42.40% | 34.29% | 0.00% |

推箱指标：

| 指标 | 数值 |
|---|---:|
| yellow push events/game | 2.0938 |
| blue push events/game | 0.4219 |
| `box_ne` final displacement | 0.1193 m |
| `box_sw` final displacement | 0.0253 m |

### 13.2 严格回放审计

数据文件：

```text
docs/rl_data/world_model_sacflow_final/strict_replay_summary.json
docs/rl_data/world_model_sacflow_final/strict_replay_audit.md
```

严格回放 8 局摘要：

| 指标 | 数值 |
|---|---:|
| episodes | 8 |
| yellow win rate | 37.50% |
| blue win rate | 62.50% |
| draw or timeout | 0.00% |
| hard violations | 0 |
| warnings | 0 |
| normal hits per episode | 3.75 |
| base wins per episode | 1.00 |
| own target penalties per episode | 0.00 |
| blocked steps per episode | 0.00 |
| target contact events per episode | 0.00 |
| robot contacts per episode | 0.00 |
| recovery events per episode | 0.00 |

严格回放检查的不是“看起来成功”，而是逐步检查：

- 是否穿静态障碍
- 是否穿箱
- 是否非法击中自己靶
- 是否未拆挡板直接命中基地
- 是否出现不可能的差速运动步长
- 分数和装甲变化是否与规则事件一致
- 三视角视频是否来自同一条 trace

### 13.3 几何审计

数据文件：

```text
docs/rl_data/rule_geometry_audit.json
docs/rl_data/rule_geometry_audit.csv
```

审计内容：

- 每个靶子的名称、所属方、坐标、yaw。
- 靶面前探针位置。
- 是否满足 45 度几何。
- 与墙、挡板、隔板是否重叠。
- 基地靶是否被剩余装甲挡板遮挡。
- 红色箱子是否可推动并持久改变位置。

当前审计 `failures = []`。

## 14. 三视角回放与媒体

最终媒体位于：

```text
docs/media/最终回放_三视角同步拼接版.gif
```

README 中第一张本地项目媒体是三视角同步 GIF，用于同时展示完整赛场行为和两车第一视角。全分辨率 MP4 与单视角 GIF 视为本地生成产物，不作为精简 GitHub 仓库的默认提交内容。三个视角分别用于：

- 顶视角：检查整体路线、推箱、基地攻坚、两车相对位置和靶子状态。
- 黄车第一视角：检查黄车是否真实看见靶、是否合法瞄准、是否卡在基地附近。
- 蓝车第一视角：检查蓝车是否真实执行路线、是否非法穿过挡板或箱子。

视频验收重点：

- 两辆车从起点同时出发。
- 靶子没有嵌墙或夹墙。
- 红色箱子被推动后位置持续变化。
- 小车没有穿箱、穿墙或穿挡板。
- 基地靶必须在挡板移除后才可能被击中。
- 小车不会反复打不可见靶。
- 碰撞或会车后不会触发异常重定位旋转。

## 15. 图表系统

### 15.1 Paper-style 方法图

图表路径：

```text
docs/figures/paper/fig01_project_overview.png
docs/figures/paper/fig02_method_architecture.png
docs/figures/paper/fig03_training_and_results.png
docs/figures/paper/fig04_ablation_and_safety.png
docs/figures/paper/fig05_sim2real_replay_pipeline.png
docs/figures/paper/world_model_sacflow_paper_figures_master.pptx
```

生成脚本：

```text
scripts/generate_paper_figures.py
```

这些图不是只改标题，而是围绕当前架构重画：

- Figure 1：项目总览、对象抽象、模型栈、评估快照和三视角证据。
- Figure 2：对象中心观测、世界模型、Flow actor、Twin-Q、Replay、loss 和部署安全门。
- Figure 3：闭环训练、优化曲线、多 seed 胜率、普通靶分布和基地命中分组。
- Figure 4：安全审计、消融协议、失败模式覆盖。
- Figure 5：ROS2/IsaacLab/策略/审计/README 媒体证据链。

### 15.2 RL 结果图

图表路径：

```text
docs/figures/rl/rl_training_curve_gpu.svg
docs/figures/rl/rl_strategy_event_metrics.svg
docs/figures/rl/rl_target_base_metrics.svg
docs/figures/rl/rl_box_push_metrics.svg
```

生成脚本：

```text
isaaclab_sim/rl/generate_rl_figures.py
```

数据来源优先读取：

```text
docs/rl_data/world_model_sacflow_final/
```

如果本地训练输出存在，也可以从 `isaaclab_sim/output/` 重建。

## 16. 测试体系

测试目录：

```text
tests/
```

主要测试文件：

| 文件 | 作用 |
|---|---|
| `tests/test_rl_env_smoke.py` | 规则环境、射击、推箱、碰撞、观测、策略合同等 smoke 测试。 |
| `tests/test_target_config.py` | 靶子布局、基地挡板、顶视角回放能力等配置检查。 |
| `tests/test_rule_gate.py` | 行为节点的对手靶安全门检查。 |
| `tests/test_rl_strategy_contract.py` | 高层策略动作、训练接口和规则契约检查。 |
| `tests/test_sim2real_config.py` | Sim2Real 参数、传感器和安全配置检查。 |

常用验证命令：

```bash
python -m pip install -r isaaclab_sim/rl/requirements.txt
python -m pytest tests -q
```

核心快速验证可以运行：

```bash
python -m pytest tests/test_rl_env_smoke.py tests/test_target_config.py -q
```

## 17. 复现实验命令

### 17.1 ROS2 构建与启动

```bash
cd crc_robocup_vision_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch rcvrl_bringup competition.launch.py
```

黄方路线：

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  team_color:=yellow \
  target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.yellow.yaml
```

蓝方路线：

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  team_color:=blue \
  target_file:=$(ros2 pkg prefix rcvrl_navigation)/share/rcvrl_navigation/config/targets.elimination.blue.yaml
```

无硬件 smoke test：

```bash
ros2 launch rcvrl_bringup competition.launch.py \
  start_navigation:=false \
  shooter_dry_run:=true \
  auto_start:=false
```

### 17.2 训练

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
  --output ../output/rl/world_model_sacflow_seed260707
```

输出：

```text
isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt
isaaclab_sim/output/rl/world_model_sacflow_seed260707/training_curve.csv
isaaclab_sim/output/rl/world_model_sacflow_seed260707/training_summary.json
```

### 17.3 多局评估

```bash
python3 isaaclab_sim/rl/evaluate_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 64 \
  --stochastic \
  --output isaaclab_sim/output/eval/world_model_sacflow_eval64.json
```

合同评估：

```bash
python3 isaaclab_sim/rl/evaluate_strategy_contract.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 64 \
  --stochastic \
  --output-json isaaclab_sim/output/eval/world_model_sacflow_contract_eval64.json \
  --output-csv isaaclab_sim/output/eval/world_model_sacflow_contract_eval64.csv
```

### 17.4 严格回放

```bash
python3 isaaclab_sim/rl/replay_policy_strict.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --episodes 8 \
  --seed 261000 \
  --stochastic \
  --policy-mode residual_expert \
  --residual-scale 0.04 \
  --output-dir isaaclab_sim/output/replay/world_model_sacflow_strict_replay \
  --report isaaclab_sim/output/replay/world_model_sacflow_strict_replay/strict_replay_audit.md
```

### 17.5 策略导出

```bash
python3 isaaclab_sim/rl/export_world_model_sacflow_policy.py \
  --checkpoint isaaclab_sim/output/rl/world_model_sacflow_seed260707/policy.pt \
  --format torchscript \
  --output-dir isaaclab_sim/output/export/world_model_sacflow_seed260707
```

### 17.6 图表与 GIF

```bash
python3 isaaclab_sim/rl/generate_rl_figures.py
python3 scripts/generate_paper_figures.py
# README GIF is generated from the synchronized three-view replay MP4 with ffmpeg/imageio_ffmpeg,
# then committed as docs/media/最终回放_三视角同步拼接版.gif.
```

## 18. 数据与产物管理

仓库保留的正式产物：

- `docs/rl_data/world_model_sacflow_final/`
- `docs/rl_data/rule_geometry_audit.json`
- `docs/rl_data/rule_geometry_audit.csv`
- `docs/media/最终回放_三视角同步拼接版.gif`
- `docs/figures/paper/`
- `docs/figures/rl/`

本地生成但默认不提交的产物：

- `isaaclab_sim/output/rl/`
- `isaaclab_sim/output/eval/`
- full-resolution replay MP4 files under `docs/media/`
- `isaaclab_sim/output/replay/`
- `isaaclab_sim/output/export/`

这样做是为了避免把大量中间 checkpoint、调试视频和历史 run 塞进 GitHub，同时保留最终结果的可复现数据。

## 19. 当前项目亮点

### 19.1 方法亮点

- 对象中心状态显式描述机器人、靶子、箱子和挡板。
- Flow actor 能表达多模态战术选择。
- 双 actor 支持黄/蓝差异化路线。
- 中心化 Twin-Q 在训练时利用全局对象信息。
- 世界模型学习对象动态，为后续想象 rollout 留接口。
- 规则专家残差降低非法探索。
- action shield 保证策略输出不会绕过安全规则。

### 19.2 工程亮点

- ROS2 与 IsaacLab 共用规则口径。
- Nav2、EKF、AprilTag、发射器服务形成完整机器人运行闭环。
- 严格回放逐步审计，不只看视频。
- README 图、PPTX、SVG、GIF 均由脚本生成或同步维护。
- 测试覆盖规则、靶子、射击、Sim2Real 和训练合同。

### 19.3 展示亮点

- README 首屏有最终俯视 GIF。
- 三视角回放覆盖整体行为和两车第一视角。
- 顶会风格图表展示方法、训练、结果、消融与证据链。
- 128 局统计评估展示黄蓝胜率接近均衡。
- 8 局严格回放显示 0 hard violations。

## 20. 当前局限与后续优化

当前系统已经具备完整闭环，但仍有几个值得继续优化的方向。

### 20.1 路线多样性

当前普通靶击倒数量主要集中在 2 个和 3 个，4 个普通靶后再攻基地的样本较少。后续可以通过课程采样或奖励 shaping 增强四靶清场路线。

### 20.2 蓝方推箱行为

当前 yellow push events/game 高于 blue。蓝方胜率均衡，但推箱路线表现不如黄方明显。可以增加蓝方推箱探索奖励，或对称化可推动箱路径机会。

### 20.3 世界模型想象 rollout

当前世界模型已学习 one-step object dynamics，但正式更新仍主要依赖真实 transition。后续可加入短时域 imagined rollout，用于更高效学习基地窗口与推箱后果。

### 20.4 真实机器人闭环

真实部署还需要继续强化：

- rosbag2 日志回放
- 相机曝光和 tag 检测稳定性
- 激光与相机光轴误差标定
- 轮胎打滑与地面摩擦拟合
- 会车碰撞后的定位恢复

## 21. 推荐阅读顺序

如果是第一次看这个项目，建议按以下顺序阅读：

1. `README.md`：看最终效果、图表和核心结果。
2. `docs/project_deep_dive.md`：读完整系统说明。
3. `docs/rules_summary.md`：理解规则。
4. `docs/architecture.md`：理解 ROS2 运行栈。
5. `docs/sim2real.md`：理解真实机器人迁移。
6. `docs/strategy.md`：理解战术设计。
7. `docs/rl_world_model_flow_policy_plan.md`：理解当前算法口径。
8. `isaaclab_sim/rl/README.md`：查看训练、评估、回放和导出命令。
9. `tests/`：查看规则和合同如何被测试。

## 22. 总结

这个项目的价值不只在于训练出一个能赢的策略，而在于把“机器人比赛规则、仿真物理、ROS2 部署、世界模型强化学习、严格审计和可展示结果”合成了一个一致的工程系统。

对象中心世界模型让策略知道场上对象如何变化；SAC Flow 风格 actor 让高层战术保持多模态；规则专家残差和 action shield 保证探索不破坏比赛契约；严格 replay 和三视角视频则让结果可以被检查，而不是只停留在数值曲线上。

最终，这套系统可以作为多智能体机器人学习、视觉目标交互、Sim2Real 工程、IsaacLab 评估和 ROS2 真实部署的完整作品集案例。
