from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="IsaacLab replay for the large-scale 50v50 battle benchmark.")
parser.add_argument("--trace_npz", type=str, default="docs/rl_data/large_scale_50v50/isaaclab_replay_trace.npz")
parser.add_argument("--record_video", type=str, default="docs/media/large_scale_50v50_isaaclab_replay.mp4")
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--record_fps", type=int, default=30)
parser.add_argument("--record_width", type=int, default=1920)
parser.add_argument("--record_height", type=int, default=1080)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaacsim.core.utils.stage import get_current_stage
from isaaclab.sensors.camera import Camera, CameraCfg
from pxr import Gf, UsdGeom


ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = (ROOT / args_cli.trace_npz).resolve() if not Path(args_cli.trace_npz).is_absolute() else Path(args_cli.trace_npz)
VIDEO_PATH = (ROOT / args_cli.record_video).resolve() if not Path(args_cli.record_video).is_absolute() else Path(args_cli.record_video)

YELLOW = (0.95, 0.78, 0.08)
BLUE = (0.12, 0.38, 0.95)
GRAY = (0.55, 0.60, 0.68)
GREEN = (0.05, 0.65, 0.32)
RED = (0.85, 0.12, 0.10)


def material(color, opacity=1.0, emissive=(0.0, 0.0, 0.0)):
    return sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        emissive_color=emissive,
        roughness=0.58,
        metallic=0.0,
        opacity=opacity,
    )


def spawn_box(path: str, size, pos, color, opacity=1.0, emissive=(0.0, 0.0, 0.0)):
    cfg = sim_utils.CuboidCfg(size=size, visual_material=material(color, opacity=opacity, emissive=emissive))
    cfg.func(path, cfg, translation=pos)


def spawn_cylinder(path: str, radius: float, height: float, pos, color, opacity=1.0):
    cfg = sim_utils.CylinderCfg(radius=radius, height=height, axis="Z", visual_material=material(color, opacity=opacity))
    cfg.func(path, cfg, translation=pos)


def set_pose(path: str, x: float, y: float, z: float, yaw: float):
    prim = get_current_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        return
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    if len(ops) < 2:
        xform.ClearXformOpOrder()
        translate = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble)
        orient = xform.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble)
    else:
        translate, orient = ops[0], ops[1]
    translate.Set(Gf.Vec3d(float(x), float(y), float(z)))
    orient.Set(Gf.Quatd(math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)))


def set_visible(path: str, visible: bool):
    prim = get_current_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    imageable.MakeVisible() if visible else imageable.MakeInvisible()


class Recorder:
    def __init__(self, width: int, height: int, fps: int, output_path: Path, trace):
        self.width = width
        self.height = height
        self.fps = fps
        self.panel_width = min(430, max(320, int(width * 0.22)))
        self.scene_width = width - self.panel_width
        self.trace = trace
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_path}")
        self.camera = Camera(
            CameraCfg(
                prim_path="/World/RecordingCamera",
                update_period=0.0,
                height=height,
                width=self.scene_width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=28.0,
                    focus_distance=66.0,
                    horizontal_aperture=38.0,
                    clipping_range=(0.1, 200.0),
                ),
            )
        )
        self.frames = 0

    def initialize_view(self):
        eye = torch.tensor([[40.0, 25.0, 66.0]], dtype=torch.float32, device=self.camera.device)
        target = torch.tensor([[40.0, 25.0, 0.0]], dtype=torch.float32, device=self.camera.device)
        self.camera.set_world_poses_from_view(eye, target)

    def write(self, sim_dt: float, frame_idx: int):
        self.camera.update(dt=sim_dt)
        rgb = self.camera.data.output.get("rgb")
        if rgb is None:
            return
        scene = rgb[0].detach().cpu().numpy()
        if scene.dtype != np.uint8:
            scene = np.clip(scene * (255.0 if scene.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if scene.shape[-1] == 4:
            scene = scene[..., :3]
        scene_bgr = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
        scene_bgr = cv2.resize(scene_bgr, (self.scene_width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.full((self.height, self.width, 3), 245, dtype=np.uint8)
        frame[:, : self.scene_width] = scene_bgr
        self._panel(frame, frame_idx)
        self.writer.write(frame)
        self.frames += 1

    def _put(self, frame, text, x, y, scale=0.55, color=(35, 35, 35), thickness=2):
        cv2.putText(frame, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _bar(self, frame, label, x, y, value, max_value, color):
        self._put(frame, label, x, y, 0.48)
        cv2.rectangle(frame, (x, y + 10), (x + 280, y + 28), (220, 224, 230), -1)
        cv2.rectangle(frame, (x, y + 10), (x + int(280 * max(0.0, min(1.0, value / max_value))), y + 28), color, -1)

    def _panel(self, frame, frame_idx):
        x0 = self.scene_width
        cv2.rectangle(frame, (x0, 0), (self.width, self.height), (248, 250, 252), -1)
        cv2.line(frame, (x0, 0), (x0, self.height), (185, 190, 200), 2)
        x = x0 + 28
        t = frame_idx * float(self.trace["dt"][0])
        y_alive = int(np.count_nonzero(self.trace["yellow_alive"][frame_idx]))
        b_alive = int(np.count_nonzero(self.trace["blue_alive"][frame_idx]))
        yhp = float(self.trace["yellow_base_hp"][frame_idx])
        bhp = float(self.trace["blue_base_hp"][frame_idx])
        zone = self.trace["zone_state"][frame_idx]
        self._put(frame, "50v50 IsaacLab Replay", x, 54, 0.74, (25, 25, 25), 2)
        self._put(frame, f"time {t:05.1f}s", x, 92, 0.58, (70, 70, 70), 2)
        self._bar(frame, "Yellow base HP", x, 145, yhp, 45.0, (40, 180, 90))
        self._bar(frame, "Blue base HP", x, 210, bhp, 45.0, (40, 180, 90))
        self._put(frame, f"Yellow alive {y_alive}/50", x, 292, 0.58, (30, 120, 155), 2)
        self._put(frame, f"Blue alive   {b_alive}/50", x, 330, 0.58, (180, 80, 25), 2)
        self._put(frame, "Control zones", x, 400, 0.58, (30, 30, 30), 2)
        for i, value in enumerate(zone):
            label = "Y" if value > 0.35 else "B" if value < -0.35 else "N"
            color = (34, 120, 220) if label == "B" else (40, 160, 95) if label == "Y" else (130, 130, 130)
            self._put(frame, f"Z{i+1}: {label}  {value:+.2f}", x, 438 + i * 34, 0.52, color, 2)
        self._put(frame, "Rule closure", x, 575, 0.58, (30, 30, 30), 2)
        self._put(frame, f"Y shield open: {bool(self.trace['yellow_base_open'][frame_idx])}", x, 615, 0.50)
        self._put(frame, f"B shield open: {bool(self.trace['blue_base_open'][frame_idx])}", x, 648, 0.50)
        self._put(frame, "zones -> shield -> base assault", x, 710, 0.48, (60, 60, 60), 1)
        if yhp <= 0.0 or bhp <= 0.0:
            winner = "YELLOW" if bhp <= 0.0 else "BLUE"
            cv2.rectangle(frame, (x, self.height - 115), (self.width - 28, self.height - 48), (34, 170, 90), -1)
            self._put(frame, f"WINNER: {winner}", x + 14, self.height - 72, 0.68, (255, 255, 255), 2)

    def close(self):
        self.writer.release()
        print(f"[VIDEO] wrote {self.frames} frames")


def spawn_scene(trace):
    width = float(trace["width_m"][0])
    height = float(trace["height_m"][0])
    spawn_box("/World/Arena/Floor", (width, height, 0.05), (width / 2.0, height / 2.0, -0.03), (0.92, 0.95, 0.98))
    for idx, y in enumerate((13.0, 25.0, 37.0)):
        spawn_box(
            f"/World/Arena/TacticalLane_{idx}",
            (68.0, 1.05, 0.025),
            (40.0, y, 0.015),
            (0.80, 0.86, 0.94),
            0.42,
        )
    spawn_box("/World/Arena/WallWest", (0.35, height, 0.8), (-0.18, height / 2.0, 0.4), (0.18, 0.20, 0.24))
    spawn_box("/World/Arena/WallEast", (0.35, height, 0.8), (width + 0.18, height / 2.0, 0.4), (0.18, 0.20, 0.24))
    spawn_box("/World/Arena/WallSouth", (width, 0.35, 0.8), (width / 2.0, -0.18, 0.4), (0.18, 0.20, 0.24))
    spawn_box("/World/Arena/WallNorth", (width, 0.35, 0.8), (width / 2.0, height + 0.18, 0.4), (0.18, 0.20, 0.24))
    for idx, rect in enumerate([(25, 6, 28, 18.5), (52, 31.5, 55, 44), (37.6, 21, 42.4, 29)]):
        xmin, ymin, xmax, ymax = rect
        spawn_box(f"/World/Arena/Obstacle_{idx}", (xmax - xmin, ymax - ymin, 1.6), ((xmin + xmax) / 2, (ymin + ymax) / 2, 0.8), (0.62, 0.68, 0.75))
    for idx, (x, y) in enumerate([(34, 13), (40, 25), (46, 37)]):
        spawn_cylinder(f"/World/Zones/Zone_{idx+1}", 6.0, 0.06, (x, y, 0.04), (0.30, 0.30, 0.34), 0.35)
    spawn_box("/World/Bases/YellowBase", (2.7, 4.8, 1.3), (4.5, 25.0, 0.65), YELLOW, emissive=(0.30, 0.24, 0.02))
    spawn_box("/World/Bases/BlueBase", (2.7, 4.8, 1.3), (75.5, 25.0, 0.65), BLUE, emissive=(0.02, 0.08, 0.30))
    for team, color in [("Yellow", YELLOW), ("Blue", BLUE)]:
        for i in range(50):
            root = f"/World/Agents/{team}_{i:02d}"
            spawn_box(root, (1.45, 0.90, 0.34), (0, 0, 0.20), color, emissive=(0.03, 0.03, 0.02))
            spawn_box(f"{root}_Nose", (0.58, 0.26, 0.22), (0, 0, 0.45), (1.0, 1.0, 1.0), emissive=(0.08, 0.08, 0.08))


def update_scene(trace, idx: int):
    for team_key, team_name in [("yellow", "Yellow"), ("blue", "Blue")]:
        pos = trace[f"{team_key}_pos"][idx]
        alive = trace[f"{team_key}_alive"][idx]
        prev = trace[f"{team_key}_pos"][max(0, idx - 1)]
        delta = pos - prev
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        for i in range(50):
            root = f"/World/Agents/{team_name}_{i:02d}"
            nose = f"{root}_Nose"
            if alive[i]:
                set_visible(root, True)
                set_visible(nose, True)
                set_pose(root, float(pos[i, 0]), float(pos[i, 1]), 0.20, float(yaw[i]))
                set_pose(nose, float(pos[i, 0] + 0.46 * math.cos(float(yaw[i]))), float(pos[i, 1] + 0.46 * math.sin(float(yaw[i]))), 0.45, float(yaw[i]))
            else:
                set_visible(root, False)
                set_visible(nose, False)


def main():
    trace = np.load(TRACE_PATH)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
    sim.set_camera_view(eye=[40.0, -30.0, 45.0], target=[40.0, 25.0, 0.0])
    dome = sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0))
    dome.func("/World/Light/Dome", dome)
    spawn_scene(trace)
    recorder = Recorder(args_cli.record_width, args_cli.record_height, args_cli.record_fps, VIDEO_PATH, trace)
    sim.reset()
    recorder.initialize_view()
    total_frames = max(1, int(args_cli.duration * args_cli.record_fps))
    sim_dt = sim.get_physics_dt()
    try:
        for out_idx in range(total_frames):
            trace_idx = int(round(out_idx * (len(trace["yellow_pos"]) - 1) / max(1, total_frames - 1)))
            update_scene(trace, trace_idx)
            sim.step()
            recorder.write(sim_dt, trace_idx)
            if out_idx % 60 == 0:
                print(f"[REPLAY] frame {out_idx}/{total_frames}")
    finally:
        recorder.close()
        if args_cli.headless:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        simulation_app.close()


if __name__ == "__main__":
    main()
