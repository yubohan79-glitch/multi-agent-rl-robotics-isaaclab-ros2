# National Top-Three Solution Context

RoboCup VisionRL documents the engineering solution evolved from a national top-three RoboCup China visual challenge entry. The public repository presents the system as a clean, reproducible project rather than as a one-off competition archive.

## Competition Constraints

- Two robots compete in a 3m x 3m visual-duel arena.
- Each robot must enter the opponent side and attack opponent targets only.
- Ordinary target hits score points and remove opponent base armor.
- Base target knockdown is the primary win condition.
- Own-base knockdown is terminal loss and own-target fire is forbidden by the software safety gate.
- Practical execution must handle short match time, occlusion, collision, localization drift and imperfect target visibility.

## Autonomous Design Points

- ROS2 Jazzy runtime with separated bringup, navigation, perception, shooter, behavior and interface packages.
- AprilTag-based target perception isolated from behavior decisions.
- Opponent-target safety gate before `/shooter/fire`.
- Nav2 and `slam_toolbox` for field navigation and recovery-friendly localization.
- IsaacLab two-robot simulation for rule rehearsal, collision handling, target falling and armor removal.
- MAPPO-style self-play for high-level tactical decisions while keeping low-level control engineered and auditable.

## Technical Route

The project follows a layered design:

1. Classical control handles differential-drive tracking, AprilTag alignment, shooter timing and EKF localization.
2. A ROS2 behavior state machine handles match execution, safety checks, retries and timeout behavior.
3. A high-level RL policy chooses target priority, base rush timing, blocking, route risk and recovery.
4. Sim2Real transfer is constrained through ROS2 topics, services, actions and TF frames rather than simulator-only state.

## Portfolio Framing

This repository emphasizes reproducibility, ownership and engineering clarity:

- official rules are summarized instead of redistributed
- third-party dependencies are attributed
- generated figures are stored as editable assets where available
- tests and CI check the public rule contract
- real hardware metrics are separated from simulation metrics

## Next Competition-Ready Milestones

- Fill hardware metrics from rosbag2 logs.
- Export a trained MAPPO actor and replay it in IsaacLab.
- Bridge high-level policy decisions into `rcvrl_behavior`.
- Record a full match video showing target fall, armor removal and base attack.
