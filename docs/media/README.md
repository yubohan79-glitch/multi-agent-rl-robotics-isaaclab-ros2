# Media Assets

This directory contains compact Git-tracked media assets used by the README and portfolio documentation.

Large raw recordings remain outside Git. The tracked files are compact MP4s generated from strict replay traces.

- `最终回放_顶视角.mp4`: current final top-view IsaacLab replay with Chinese filename for portfolio delivery.
- `最终回放_黄车第一视角.mp4`: current final yellow robot first-person IsaacLab replay with Chinese filename for portfolio delivery.
- `最终回放_蓝车第一视角.mp4`: current final blue robot first-person IsaacLab replay with Chinese filename for portfolio delivery.
- `isaaclab_expert_base_cap_physical_boxes_seed10705_top.mp4`: current IsaacLab top-view replay generated after the real PhysX pushable-box fix, strict episode seed 10705.
- `isaaclab_expert_base_cap_physical_boxes_seed10705_yellow_pov.mp4`: current yellow robot first-person replay from the same audited physical-box trace.
- `isaaclab_expert_base_cap_physical_boxes_seed10705_blue_pov.mp4`: current blue robot first-person replay from the same audited physical-box trace.
- `isaaclab_expert_base_cap_seed10622_top.mp4`: archived top-view replay from the earlier expert base-cap strict trace, seed 10622.
- `isaaclab_expert_base_cap_seed10622_yellow_pov.mp4`: archived yellow robot first-person replay from the earlier trace.
- `isaaclab_expert_base_cap_seed10622_blue_pov.mp4`: archived blue robot first-person replay from the earlier trace.
- `isaaclab_contact_hull_top.mp4`: archived IsaacLab top-view replay generated from the dual-expert contact-hull strict trace, episode 5.
- `isaaclab_contact_hull_yellow_pov.mp4`: archived yellow robot first-person replay from the same audited trace and corrected contact hull.
- `isaaclab_contact_hull_blue_pov.mp4`: archived blue robot first-person replay from the same audited trace and corrected contact hull.
- `isaaclab_blend052_contactfix_overview.mp4`: archived overview replay from the recovery-cooldown/contact-fix dual-expert strict trace, episode 3.
- `isaaclab_blend052_contactfix_yellow_pov.mp4`: archived yellow robot first-person replay from that older trace.
- `isaaclab_blend052_contactfix_blue_pov.mp4`: archived blue robot first-person replay from that older trace.
- `isaaclab_dual_experts_overview.mp4`: archived overview replay from the earlier dual yellow/blue expert MAPPO strict trace, episode 0.
- `isaaclab_dual_experts_yellow_pov.mp4`: archived yellow robot first-person replay from that older trace.
- `isaaclab_dual_experts_blue_pov.mp4`: archived blue robot first-person replay from that older trace.
- `final_training_replay_overview.mp4`: archived IsaacLab overview replay from the earlier side-gated, rigid-blocker, recessed-base MAPPO strict trace, episode 8.
- `final_training_replay_yellow_pov.mp4`: archived yellow robot first-person replay from that older trace.
- `final_training_replay_blue_pov.mp4`: archived blue robot first-person replay from that older trace.
- `offaxis_base_ros2_overview.mp4`: archived overview replay from the earlier off-axis base-rush MAPPO strict trace, episode 12.
- `offaxis_base_ros2_yellow_pov.mp4`: archived yellow robot first-person replay from that older trace.
- `offaxis_base_ros2_blue_pov.mp4`: archived blue robot first-person replay from that older trace.

The current replay trace records pushable obstacle state (`box_ne_x/y`, `box_sw_x/y`). In IsaacLab, the two red obstacles are spawned as real PhysX dynamic rigid boxes with explicit collision, 1.8 kg mass, high-friction physics material, gravity enabled, and `kinematicEnabled=false`; the replay synchronizes their persistent pushed state instead of drawing fixed props. The exported USD physics audit is stored at `isaaclab_sim/output/robocup_visionrl_pushable_physics_audit.json`.

Latest MP4 metadata:

| File | View | Duration | Frames | Size | Strict source |
|---|---|---:|---:|---:|---|
| `最终回放_顶视角.mp4` | top | 35.00 s | 1050 | 10.38 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |
| `最终回放_黄车第一视角.mp4` | yellow POV | 33.80 s | 1014 | 9.19 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |
| `最终回放_蓝车第一视角.mp4` | blue POV | 25.97 s | 779 | 4.69 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |
| `isaaclab_expert_base_cap_physical_boxes_seed10705_top.mp4` | top | 35.00 s | 1050 | 10.38 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |
| `isaaclab_expert_base_cap_physical_boxes_seed10705_yellow_pov.mp4` | yellow POV | 33.80 s | 1014 | 9.19 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |
| `isaaclab_expert_base_cap_physical_boxes_seed10705_blue_pov.mp4` | blue POV | 25.97 s | 779 | 4.69 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |

The selected physical-box strict episode has zero hard violations, zero warnings, zero own-target penalties, persistent north-east red-box displacement of about 25 cm, and ends with a legal yellow base-target win from the removed-armor side after the recessed blue base armor is opened. The new strict audit uses an oriented robot visual hull against each red-box AABB, which catches the top-view overlap case that the old circle approximation missed.
