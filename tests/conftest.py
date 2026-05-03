from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RL_DIR = ROOT / "isaaclab_sim" / "rl"

if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))
