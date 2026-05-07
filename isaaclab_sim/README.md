# IsaacLab Simulation

This folder contains the IsaacLab/Isaac Sim scene for the two-robot `RoboCup VisionRL` visual challenge simulation. It is the visual playback layer for audited RL traces and the place where arena geometry, sensors, target placement, armor blockers and video capture are assembled.

Run from the local IsaacLab checkout on Windows:

```powershell
cd <isaaclab-root>
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py
```

Headless smoke test:

```powershell
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py --headless --duration 5
```

Record the latest audited three-view replay:

```powershell
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py `
  --headless --duration 40 `
  --replay_trace <repo-root>\isaaclab_sim\output\replay\world_model_sacflow_strict_replay_abs\strict_replay_trace.csv `
  --replay_events <repo-root>\isaaclab_sim\output\replay\world_model_sacflow_strict_replay_abs\strict_replay_events.jsonl `
  --replay_episode 0 `
  --record_video <repo-root>\docs\media\最终回放_顶视角.mp4 `
  --record_view top --record_fps 30 --record_width 1600 --record_height 900
```

Repeat with `--record_view yellow_pov` / `--record_video <repo-root>\docs\media\最终回放_黄车第一视角.mp4` and `--record_view blue_pov` / `--record_video <repo-root>\docs\media\最终回放_蓝车第一视角.mp4` for the two robot first-person videos.

Live IsaacLab camera/lidar streams are opt-in because this PC's Isaac Sim 5.1 build can keep Replicator alive during headless shutdown:

```powershell
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py --enable_sensor_streams --enable_cameras
```

The scene uses metric dimensions from the competition material:

- 3m x 3m arena
- 0.5m wall height
- 0.5m x 0.5m start/base zones
- two 0.3m x 0.3m x 0.3m obstacles
- rule-page-aligned blue/yellow start zones and bases
- mid-field and base/start-zone fence segments matching the competition diagram
- eight normal targets placed about 45 degrees to the corner/wall geometry
- smaller yellow/blue base targets recessed inside the base so they are blocked until armor is removed
- Tag36h11 visual target mockups with 5cm tag size and 7cm bottom height
- ground-touching blue armor blockers registered as both navigation and laser blockers
- two real PhysX dynamic pushable boxes with explicit collision, mass, gravity and high-friction physics material; their pushed poses can also be synchronized from strict replay traces
- two robot envelopes aligned to the portfolio robot: 0.34m length, 0.24m width, 0.245m height

Each robot model includes an RGB camera, depth camera, 2D lidar ray-caster, IMU, ToF modules, bumper contacts, wheel encoders, differential-drive wheel layout, and a fixed low-power laser/shooter preview. Scripted previews and trained replay traces are checked against inflated wall, armor, target and obstacle blockers, so robots are rendered outside static blockers and pushable boxes instead of visually passing through them. The robot footprints are also resolved against each other, so yellow/blue contact is treated as a collision instead of a pass-through.

Competition rule logic is active in the GUI scene:

- yellow robot enters the blue side and attacks only blue targets
- blue robot enters the yellow side and attacks only yellow targets
- own normal target hit attempts are rejected by the strategy layer; own base target hits are a hard replay violation
- normal target hit: target falls and one opponent base armor plate is removed
- opponent base target hit: base target falls and the firing team wins
- target contact is a brush/relocalization event only; contact does not knock targets down
- laser hits require legal opponent target ownership, clear line of sight, distance-dependent accuracy and at least 0.80 s dwell; normal targets use 5-50 cm shooter-outlet range, recessed base targets use 20-80 cm
- base armor plates are active navigation and laser blockers until removed in rule order
- pushable boxes are real dynamic rigid bodies in IsaacLab (`rigidBodyEnabled=true`, `kinematicEnabled=false`, 1.8 kg, high-friction material), while strict replay traces provide the persistent pushed state used for audited video reproduction

The reinforcement-learning bridge lives in `rl/`. The selected training path is object-centric world-model SAC Flow self-play: a flow-policy actor, centralized twin-Q critic, auxiliary object dynamics model, team-specific expert priors, Sim2Real domain randomization, contact-safe robot separation, geometry-aware action shielding and strict replay auditing before rendering in IsaacLab. The selected strict replay passes 8 audited episodes with 0 hard violations, 0 warnings, 0 own-target penalties and 1.0 base wins per episode. The USD export is written to `output/robocup_visionrl_arena.usd` whenever the script starts.
