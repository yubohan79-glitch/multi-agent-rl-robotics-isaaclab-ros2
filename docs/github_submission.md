# GitHub Submission Notes

Recommended repository URL:

```text
https://github.com/yubohan79-glitch/RoboCupVisionRL_IsaacLab_ROS2
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

Before pushing, use these exact commands from the repository root. The
`.gitignore` is configured so `git add -A` keeps runtime outputs, checkpoints,
official PDF extracts and debug frame dumps out of the commit while preserving
source code, ROS2 packages, RL scripts, reproducibility data, figures and compact
README media.

```bash
git remote set-url origin https://github.com/yubohan79-glitch/RoboCupVisionRL_IsaacLab_ROS2.git
git add -A
git status --short
git commit -m "Publish RoboCup VisionRL IsaacLab ROS2 project"
git branch -M main
git push -u origin main
```

Do not commit local runtime artifacts:

- ROS2 `build/`, `install/`, `log/`
- IsaacLab `output/`, `runs/`, `wandb/`
- rosbags and camera captures
- `.env` or machine-specific hardware configuration
- official competition PDFs, extracted official rule text or local DOC/DOCX submissions
