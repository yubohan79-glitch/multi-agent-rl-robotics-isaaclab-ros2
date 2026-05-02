# GitHub Submission Notes

Recommended repository name:

```text
robocup-visionrl-isaaclab-ros2
```

Submission contents:

- `crc_robocup_vision_ws/`: clean ROS2 workspace.
- `isaaclab_sim/`: IsaacLab arena, rule simulation, and RL interfaces.
- `docs/`: architecture, strategy, Sim2Real, migration, and result notes.
- `THIRD_PARTY_NOTICES.md`: attribution for dependencies and mesh references.
- `.github/workflows/ros2-ci.yml`: ROS2 Jazzy build and Python RL smoke checks.
- `docs/figures/`: grouped portfolio, paper-style, and reinforcement-learning figures.

Official competition PDFs, extracted rule text, and screenshots are intentionally not committed. Keep them as local references and use `docs/rules_summary.md` for the public repository.

Create the GitHub repository as an empty repository:

- Visibility: `Public`
- Add README: off
- Add .gitignore: `No .gitignore`
- Add license: `No license`

Before pushing:

```bash
git add .
git status --short
git commit -m "Initial RoboCup VisionRL IsaacLab ROS2 portfolio"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

Do not commit local runtime artifacts:

- ROS2 `build/`, `install/`, `log/`
- IsaacLab `output/`, `runs/`, `wandb/`
- rosbags and camera captures
- `.env` or machine-specific hardware configuration
