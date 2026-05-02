# IsaacLab Simulation

This folder contains the IsaacLab/Isaac Sim scene for the two-robot `RoboCup VisionRL` visual challenge simulation.

Run from the local IsaacLab checkout on Windows:

```powershell
cd <isaaclab-root>
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py
```

Headless smoke test:

```powershell
.\isaaclab.bat -p <repo-root>\isaaclab_sim\robocup_visionrl_arena_sim.py --headless --duration 5
```

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
- eight normal targets plus yellow/blue base targets with corrected base IDs
- Tag36h11 visual target mockups with 5cm tag size and 7cm bottom height
- two robot envelopes aligned to the portfolio robot: 0.34m length, 0.24m width, 0.245m height

Each robot model includes an RGB camera, depth camera, 2D lidar ray-caster, IMU, ToF modules, bumper contacts, wheel encoders, differential-drive wheel layout, and a fixed low-power laser/shooter preview. The demo routes are checked against inflated wall, armor, and obstacle blockers before the scene starts, so the visible patrol does not pass through fences or obstacles. The robot footprints are also resolved against each other, so yellow/blue contact is treated as a collision instead of a pass-through.

Competition rule logic is active in the GUI scene:

- yellow robot enters the blue side and attacks only blue targets
- blue robot enters the yellow side and attacks only yellow targets
- own normal target hit attempts are rejected by the strategy layer; own base target hits lose the match
- normal target hit: target falls and one opponent base armor plate is removed
- opponent base target hit: base target falls and the firing team wins
- collision-knocked targets fall; the non-contact team receives the rule score, and own-base collision ends the match
- base armor plates are active navigation and laser blockers until removed in rule order

The reinforcement-learning bridge lives in `rl/`. The selected training path is MAPPO-style self-play for high-level strategy, with PPO kept as a fast single-agent baseline. The USD export is written to `output/robocup_visionrl_arena.usd` whenever the script starts.
