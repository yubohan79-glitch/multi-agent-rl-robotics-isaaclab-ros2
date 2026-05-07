# Media Assets

This directory intentionally keeps only the final three IsaacLab replay MP4s used by the README and portfolio delivery.

Large raw recordings, archived probes and older experiment videos stay outside Git. The final replay trace records pushable obstacle state (`box_ne_x/y`, `box_sw_x/y`); in IsaacLab the two red obstacles are spawned as real PhysX dynamic rigid boxes with collision, mass, friction, gravity and persistent pushed poses.

## Final Replay Videos

| File | View | Duration | Frames | Size | Strict source |
|---|---|---:|---:|---:|---|
| `最终回放_顶视角.mp4` | top | 35.00 s | 1050 | 10.38 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |
| `最终回放_黄车第一视角.mp4` | yellow POV | 33.80 s | 1014 | 9.19 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |
| `最终回放_蓝车第一视角.mp4` | blue POV | 25.97 s | 779 | 4.69 MB | `docs/rl_expert_base_cap_rng_physical_boxes_strict8.md` |

The selected physical-box strict episode has zero hard violations, zero warnings, zero own-target penalties, persistent north-east red-box displacement of about 25 cm, and ends with a legal yellow base-target win from the removed-armor side after the recessed blue base armor is opened.
