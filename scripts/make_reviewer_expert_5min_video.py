from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "docs" / "media"
DATA_DIR = ROOT / "docs" / "rl_data" / "world_model_sacflow_final"
REFERENCE_DIR = MEDIA_DIR / "reference_style"
EXTERNAL_AI_DIR = MEDIA_DIR / "external_ai"

OUTPUT = MEDIA_DIR / "最终回放_审稿人专家讲解版_5min_4K30.mp4"
TMP_VIDEO = MEDIA_DIR / "reviewer_expert_5min_video_only.mp4"
NARRATION_WAV = MEDIA_DIR / "reviewer_expert_5min_narration.wav"
LOG_PATH = MEDIA_DIR / "reviewer_expert_5min_render.log"

INPUTS = {
    "top": MEDIA_DIR / "最终回放_顶视角.mp4",
    "yellow": MEDIA_DIR / "最终回放_黄车第一视角.mp4",
    "blue": MEDIA_DIR / "最终回放_蓝车第一视角.mp4",
}

EXTERNAL_AI_CLIPS = [
    EXTERNAL_AI_DIR / "ai_ltx_01_cyber_arena.mp4",
    EXTERNAL_AI_DIR / "ltx_v2_arena.mp4",
    EXTERNAL_AI_DIR / "ltx_v2_world_model.mp4",
    EXTERNAL_AI_DIR / "ltx_v2_policy_flow.mp4",
]

W, H = 3840, 2160
FPS = 30
MIN_DURATION = 340.0

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]

BG = (9, 14, 24)
PANEL = (16, 24, 39)
PANEL2 = (21, 31, 51)
INK = (226, 232, 240)
MUTED = (148, 163, 184)
LINE = (51, 65, 85)
GREEN = (34, 197, 94)
YELLOW = (245, 158, 11)
BLUE = (56, 189, 248)
RED = (239, 68, 68)
PURPLE = (167, 139, 250)
WHITE = (248, 250, 252)


NARRATION = """
先直接看最终三视角回放。左侧是顶视角，用来检查全局路线、红色箱子、蓝色挡板和两车交互。右侧是黄车和蓝车第一视角，用来检查瞄准、遮挡、射击距离和真实视线。这个项目的目标不是剪一个好看的成功片段，而是让完整比赛过程能够被规则、物理和统计同时复核。

RoboCup 风格任务的难点在于，策略不能只学会走到目标点。它必须理解普通靶、基地靶、红色箱子、蓝色挡板、激光驻留和非法命中之间的关系。普通靶决定对方基地挡板的解锁进度；红箱会改变路线空间；基地靶只有在挡板被移除、视线合法、距离合法之后才可能被命中。

物理可信是第一层门槛。红色箱子必须是真实可推动刚体，小车接触箱子时只能推动或者被阻挡，不能穿过去。蓝色基地挡板必须落地，并同时阻挡车辆和激光 raycast。只要这些物理约束不可信，后面的强化学习结果就没有意义。

射击规则也不是碰到就算。激光出口到靶面的距离必须在二十到八十厘米之间，视线不能被挡板或墙体遮挡。命中还需要驻留时间：零点八秒以内一定不倒，零点八秒到两秒之间命中概率线性增强。这个设计避免策略用瞬间擦边、穿模或者错误遮挡获得虚假成功。

工程上，ROS2 负责真实机器人接口，包括视觉检测、导航、定位融合、速度控制和发射器服务。IsaacLab 负责可视化场景、物理回放和训练环境。训练、评估、回放共享同一套规则契约，这样同一个事件不会在训练里算成功、在回放里又变成非法。

方法上，我们使用对象中心世界模型。策略看到的不只是一串扁平向量，而是机器人、靶子、箱子、挡板、激光视线、比分和时间等对象。对象之间的关系描述谁挡住了谁、哪个目标属于对方、哪条路线可能碰撞、下一步会改变什么规则状态。

在策略层，正式主线使用 SAC Flow 或 PolicyFlow 风格的自博弈。黄车和蓝车有独立 actor，用来保留双方路线偏好和攻防节奏。中心化 Twin-Q critic 评估双方交互后的长期价值。Flow 的意义是保留多条候选路线，而不是过早塌缩到单一路线。

执行层还加入了微调瞄准。小车到达射击点后，不是原地卡住，而是进行慢速小角度扫描，让激光在合法视线内稳定驻留。如果基地靶角度不够，策略会尝试向合法侧侧移一点，再重新对准。这个动作很小，但能显著减少到点位后打不中靶子的情况。

最后看评估。当前归档评估使用一百二十八局多 seed，黄方胜率约百分之四十九点二，蓝方胜率约百分之五十点八，平局为零。普通靶击倒数量主要集中在两个和三个，基地命中率随已击倒普通靶数量增加而提升。严格回放审计记录 hard violation 为零，并检查箱子位移、机器人碰撞、穿模、重定位和异常旋转。

因此，这个项目真正想展示的是一条可复核链路：规则可信、物理可信、训练可信、回放可信。三视角视频是最终呈现，背后是对象中心建模、Flow 策略、自博弈训练、ROS2 与 IsaacLab 的工程闭环，以及可以被审稿人复查的评估数据。
""".strip()


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    key: str
    title: str
    caption: str

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    def u(self, t: float) -> float:
        return min(1.0, max(0.0, (t - self.start) / max(self.end - self.start, 1e-6)))


SEGMENTS = [
    Segment(0, 38, "three_open", "最终三视角同步回放", "先看真实完整比赛：顶视角检查全局路线，第一视角检查瞄准、遮挡和真实视线。"),
    Segment(38, 74, "arena", "任务规则：不是单车导航", "普通靶、基地靶、红箱、挡板和激光驻留共同决定比赛结果。"),
    Segment(74, 112, "physics", "物理可信：箱子能推，挡板能挡", "红箱是真实刚体；蓝色挡板必须同时阻挡车辆和激光。"),
    Segment(112, 148, "shooting", "射击规则：距离、视线、驻留", "20-80cm 合法距离；0.8s 以内必不倒，0.8-2s 概率增强。"),
    Segment(148, 186, "system", "ROS2 + IsaacLab 工程闭环", "真实机器人接口、仿真物理、训练评估和回放共享同一套规则契约。"),
    Segment(186, 236, "world_model", "对象中心世界模型", "把机器人、靶子、箱子、挡板、视线、比分和时间显式建模为对象。"),
    Segment(236, 286, "flow", "SAC Flow / PolicyFlow 自博弈", "双 actor 保留黄蓝差异，中心化 Twin-Q critic 评估交互长期价值。"),
    Segment(286, 314, "microaim", "微调瞄准：避免到点后卡住", "到达射击点后慢速小角度扫描，必要时侧移一点再重新对准。"),
    Segment(314, 336, "eval", "多 seed 评估与严格审计", "128 局评估、胜率平衡、箱子位移、穿模、碰撞和 hard violation。"),
    Segment(336, 340, "ending", "可复核的机器人学习闭环", "规则可信、物理可信、训练可信、回放可信。"),
]


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(82)
F_H1 = font(64)
F_H2 = font(46)
F_BODY = font(40)
F_SMALL = font(30)
F_TINY = font(24)


def ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 3 * u * u - 2 * u * u * u


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, text_font: ImageFont.ImageFont) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            cand = cur + ch
            if draw.textlength(cand, font=text_font) <= max_width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, text_font: ImageFont.ImageFont, fill=INK, spacing=10) -> int:
    x, y = xy
    for line in wrap_text(draw, text, width, text_font):
        draw.text((x, y), line, font=text_font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=text_font)
        y += bbox[3] - bbox[1] + spacing
    return y


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill, outline=LINE, radius=26, width=2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def add_glow(img: Image.Image, box: tuple[int, int, int, int], color, radius=16, width=4) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=28, outline=(*color, 160), width=width)
    blur = layer.filter(ImageFilter.GaussianBlur(radius))
    img.alpha_composite(blur)
    img.alpha_composite(layer)


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    sw, sh = image.size
    tw, th = size
    scale = min(tw / sw, th / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (15, 23, 42))
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


class LoopVideo:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.index = -1
        self.last: np.ndarray | None = None

    def read_at(self, t: float) -> np.ndarray:
        target = int((t * self.fps) % self.frames)
        if target < self.index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.index = -1
            self.last = None
        while self.index < target:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.index = -1
                continue
            self.index += 1
            self.last = frame
        if self.last is None:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError(f"cannot read first frame: {self.path}")
            self.index = 0
            self.last = frame
        return self.last

    def close(self) -> None:
        self.cap.release()


def copy_inputs_to_temp(tmp: Path) -> dict[str, Path]:
    copied = {}
    for key, src in INPUTS.items():
        if not src.exists():
            raise FileNotFoundError(src)
        dst = tmp / f"{key}.mp4"
        shutil.copy2(src, dst)
        copied[key] = dst
    return copied


def load_stats() -> dict:
    stats = {}
    try:
        data = json.loads((DATA_DIR / "contract_eval_multiseed.json").read_text(encoding="utf-8"))
        stats["contract"] = data.get("summary", {})
    except Exception:
        stats["contract"] = {}
    try:
        data = json.loads((DATA_DIR / "strict_replay_summary.json").read_text(encoding="utf-8"))
        stats["replay"] = data.get("summary", {})
    except Exception:
        stats["replay"] = {}
    try:
        data = json.loads((DATA_DIR / "training_summary.json").read_text(encoding="utf-8"))
        stats["training"] = data
    except Exception:
        stats["training"] = {}
    return stats


def generate_tts(path: Path) -> float:
    try:
        import win32com.client  # type: ignore

        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voices = voice.GetVoices()
        for i in range(voices.Count):
            desc = voices.Item(i).GetDescription()
            if "Huihui" in desc or "Chinese" in desc:
                voice.Voice = voices.Item(i)
                break
        voice.Rate = -1
        voice.Volume = 96
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(path), 3, False)
        voice.AudioOutputStream = stream
        voice.Speak(NARRATION)
        stream.Close()
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception as exc:
        log(f"TTS failed, continue without narration: {exc}")
        return 0.0


def base_canvas(t: float, seg: Segment) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(img)
    # Subtle grid and animated scan lines.
    for x in range(0, W, 120):
        shade = 26 + int(8 * math.sin(t * 0.6 + x * 0.01))
        draw.line((x, 0, x, H), fill=(shade, shade + 4, shade + 12, 120), width=1)
    for y in range(0, H, 120):
        shade = 24 + int(7 * math.cos(t * 0.5 + y * 0.01))
        draw.line((0, y, W, y), fill=(shade, shade + 4, shade + 12, 120), width=1)
    draw.rectangle((0, 0, W, 116), fill=(6, 10, 18, 245))
    draw.text((86, 30), "RoboCupVisionRL | Object-Centric World-Model Flow RL", font=F_H2, fill=WHITE)
    draw.text((2580, 42), "专家讲解版 · 4K30 · 三视角优先", font=F_SMALL, fill=MUTED)
    progress = min(1.0, max(0.0, t / max(MIN_DURATION, 1)))
    draw.rounded_rectangle((86, 96, W - 86, 104), radius=4, fill=(30, 41, 59, 255))
    draw.rounded_rectangle((86, 96, int(86 + (W - 172) * progress), 104), radius=4, fill=(*GREEN, 255))
    return img, draw


def draw_footer_caption(draw: ImageDraw.ImageDraw, seg: Segment) -> None:
    rounded(draw, (230, 1904, 3610, 2078), (8, 13, 23, 226), (71, 85, 105), 28, 2)
    draw.text((280, 1938), seg.title, font=F_H2, fill=WHITE)
    draw_wrapped(draw, (280, 1998), seg.caption, 3000, F_BODY, fill=(203, 213, 225), spacing=8)


def panel_label(draw: ImageDraw.ImageDraw, box, text, color):
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1 + 22, y1 + 20, x1 + 420, y1 + 70), radius=18, fill=(3, 7, 18, 210), outline=color, width=2)
    draw.text((x1 + 44, y1 + 30), text, font=F_SMALL, fill=WHITE)


def draw_three_view(img: Image.Image, draw: ImageDraw.ImageDraw, videos: dict[str, LoopVideo], t: float, headline=True) -> None:
    top_box = (112, 176, 2470, 1514)
    y_box = (2545, 176, 3728, 826)
    b_box = (2545, 864, 3728, 1514)
    boxes = {"top": top_box, "yellow": y_box, "blue": b_box}
    labels = {"top": "顶视角 / 全局路线", "yellow": "黄车第一视角", "blue": "蓝车第一视角"}
    colors = {"top": GREEN, "yellow": YELLOW, "blue": BLUE}
    sizes = {k: (boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]) for k in boxes}
    for key in ["top", "yellow", "blue"]:
        frame = videos[key].read_at(t * 0.85)
        panel = fit_frame(frame, sizes[key]).convert("RGBA")
        box = boxes[key]
        add_glow(img, box, colors[key], radius=14, width=4)
        img.alpha_composite(panel, (box[0], box[1]))
        draw.rounded_rectangle(box, radius=28, outline=colors[key], width=4)
        panel_label(draw, box, labels[key], colors[key])
    if headline:
        draw.text((120, 1540), "0 秒直接看真实回放", font=F_TITLE, fill=WHITE)
        draw_wrapped(draw, (122, 1640), "顶视角验证全局路径、箱子和挡板；第一视角验证瞄准、遮挡、距离和视线。", 2320, F_BODY, fill=(203, 213, 225))
    checks = [
        ("红箱真实可推动", RED),
        ("蓝挡板阻挡车和激光", BLUE),
        ("0.8s 驻留门槛", GREEN),
        ("20-80cm 合法射距", YELLOW),
        ("穿模 / 碰撞审计", PURPLE),
    ]
    x = 2545
    y = 1570
    for label, color in checks:
        draw.rounded_rectangle((x, y, x + 550, y + 62), radius=18, fill=(15, 23, 42, 230), outline=color, width=2)
        draw.ellipse((x + 24, y + 18, x + 50, y + 44), fill=color)
        draw.text((x + 70, y + 16), label, font=F_SMALL, fill=INK)
        y += 78


def arena_to_px(x: float, y: float, box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int(x1 + (x + 1.5) / 3.0 * (x2 - x1)), int(y2 - (y + 1.5) / 3.0 * (y2 - y1))


def draw_arena_map(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    box = (250, 260, 1780, 1790)
    rounded(draw, box, (11, 18, 32, 238), (71, 85, 105), 34, 3)
    inner = (box[0] + 70, box[1] + 70, box[2] - 70, box[3] - 70)
    draw.rectangle(inner, fill=(15, 23, 42, 255), outline=(100, 116, 139), width=5)
    for i in range(1, 3):
        x = inner[0] + i * (inner[2] - inner[0]) // 3
        y = inner[1] + i * (inner[3] - inner[1]) // 3
        draw.line((x, inner[1], x, inner[3]), fill=(51, 65, 85), width=2)
        draw.line((inner[0], y, inner[2], y), fill=(51, 65, 85), width=2)
    targets = [
        (0.18, 1.26, BLUE, "T1"), (1.26, 1.26, BLUE, "T2"), (-1.26, 0.24, BLUE, "T3"),
        (-1.26, -0.24, YELLOW, "T4"), (1.26, 0.24, BLUE, "T5"), (1.26, -0.24, YELLOW, "T6"),
        (-1.26, -1.26, YELLOW, "T7"), (-0.18, -1.26, YELLOW, "T8"),
    ]
    appear = int(1 + ease(u) * len(targets))
    for x, y, color, label in targets[:appear]:
        px, py = arena_to_px(x, y, inner)
        draw.rounded_rectangle((px - 50, py - 18, px + 50, py + 18), radius=10, fill=(*color, 210), outline=WHITE, width=2)
        draw.text((px - 23, py - 15), label, font=F_TINY, fill=(3, 7, 18))
    for x, y, color, label in [(0.25, -1.25, YELLOW, "Y"), (-0.25, 1.25, BLUE, "B")]:
        px, py = arena_to_px(x, y, inner)
        draw.ellipse((px - 38, py - 38, px + 38, py + 38), fill=(*color, 230), outline=WHITE, width=3)
        draw.text((px - 12, py - 22), label, font=F_SMALL, fill=(3, 7, 18))
    for x, y in [(0.55, 0.1), (-0.55, -0.1)]:
        px, py = arena_to_px(x, y, inner)
        draw.rounded_rectangle((px - 55, py - 55, px + 55, py + 55), radius=12, fill=(*RED, 210), outline=WHITE, width=3)
    for x, y, color, label in [(-1.36, 1.36, BLUE, "蓝基地"), (1.36, -1.36, YELLOW, "黄基地")]:
        px, py = arena_to_px(x, y, inner)
        draw.rectangle((px - 70, py - 70, px + 70, py + 70), fill=(*color, 120), outline=color, width=4)
        draw.text((px - 54, py - 18), label, font=F_TINY, fill=WHITE)
    draw.text((1990, 300), "任务不是“走到点位”", font=F_TITLE, fill=WHITE)
    bullets = [
        "普通靶：决定对方基地挡板解锁进度",
        "红色箱子：改变可达路线，必须真实可推动",
        "基地挡板：必须阻挡车辆和激光 raycast",
        "基地靶：只有合法侧、合法距离、合法视线才可命中",
    ]
    y = 460
    for i, text in enumerate(bullets):
        color = [BLUE, RED, PURPLE, GREEN][i]
        draw.rounded_rectangle((1995, y, 3530, y + 118), radius=24, fill=(15, 23, 42, 230), outline=color, width=3)
        draw.ellipse((2038, y + 40, 2078, y + 80), fill=color)
        draw.text((2110, y + 34), text, font=F_BODY, fill=INK)
        y += 154


def draw_physics(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((170, 240), "物理可信先于训练可信", font=F_TITLE, fill=WHITE)
    draw_wrapped(draw, (175, 345), "如果箱子不能推动、挡板不能阻挡、碰撞不终止，那么任何高胜率都只是场景漏洞。", 1660, F_BODY, fill=(203, 213, 225))
    # Red box collision shell.
    cx, cy = 900, 1040
    angle = t * 0.8
    draw.rounded_rectangle((cx - 210, cy - 210, cx + 210, cy + 210), radius=34, fill=(*RED, 215), outline=WHITE, width=5)
    draw.rounded_rectangle((cx - 260, cy - 260, cx + 260, cy + 260), radius=42, outline=(*RED, 150), width=6)
    draw.text((cx - 200, cy + 270), "红箱：动态刚体 + 碰撞体", font=F_BODY, fill=INK)
    # Robot pushing vector.
    rx = int(cx - 560 + 80 * math.sin(t))
    draw.ellipse((rx - 70, cy - 70, rx + 70, cy + 70), fill=(*YELLOW, 235), outline=WHITE, width=4)
    draw.line((rx + 80, cy, cx - 280, cy), fill=YELLOW, width=10)
    draw.polygon([(cx - 280, cy), (cx - 330, cy - 28), (cx - 330, cy + 28)], fill=YELLOW)
    # Blocker and laser.
    bx, by = 2600, 990
    draw.rounded_rectangle((bx - 80, by - 440, bx + 80, by + 440), radius=32, fill=(*BLUE, 205), outline=WHITE, width=5)
    draw.text((bx - 240, by + 490), "蓝挡板：落地阻挡", font=F_BODY, fill=INK)
    draw.ellipse((bx - 740, by - 55, bx - 630, by + 55), fill=(*YELLOW, 235), outline=WHITE, width=4)
    draw.line((bx - 620, by, bx - 92, by), fill=(250, 204, 21), width=8)
    draw.line((bx + 95, by, bx + 690, by), fill=(250, 204, 21, 80), width=4)
    draw.text((bx - 710, by - 145), "raycast 被挡板截断", font=F_SMALL, fill=(250, 204, 21))
    badges = [("穿箱", False), ("穿墙", False), ("未拆挡板命中基地", False), ("碰撞/位移可审计", True)]
    x = 1920
    y = 1480
    for label, ok in badges:
        color = GREEN if ok else RED
        draw.rounded_rectangle((x, y, x + 710, y + 80), radius=22, fill=(15, 23, 42, 235), outline=color, width=3)
        draw.text((x + 34, y + 20), ("允许检查：" if ok else "禁止：") + label, font=F_SMALL, fill=INK)
        y += 104


def draw_shooting(draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((170, 245), "射击规则：距离、视线、驻留时间", font=F_TITLE, fill=WHITE)
    # Timeline
    x1, y = 420, 820
    x2 = 3420
    draw.line((x1, y, x2, y), fill=(71, 85, 105), width=12)
    p08 = x1 + int((0.8 / 2.0) * (x2 - x1))
    p20 = x2
    draw.line((x1, y, p08, y), fill=RED, width=16)
    draw.line((p08, y, p20, y), fill=GREEN, width=16)
    for x, label in [(x1, "0s"), (p08, "0.8s"), (p20, "2.0s")]:
        draw.line((x, y - 55, x, y + 55), fill=WHITE, width=4)
        draw.text((x - 45, y + 78), label, font=F_BODY, fill=INK)
    pulse = x1 + int((0.08 + 0.84 * ((t * 0.18) % 1.0)) * (x2 - x1))
    draw.ellipse((pulse - 34, y - 34, pulse + 34, y + 34), fill=(250, 204, 21), outline=WHITE, width=4)
    draw.text((420, 610), "0.8 秒以内：100% 不倒", font=F_H2, fill=RED)
    draw.text((1600, 610), "0.8-2 秒：命中概率线性增强", font=F_H2, fill=GREEN)
    # Range diagram
    sx, sy = 1120, 1390
    tx, ty = 2680, 1390
    draw.ellipse((sx - 80, sy - 80, sx + 80, sy + 80), fill=(*YELLOW, 230), outline=WHITE, width=4)
    draw.rounded_rectangle((tx - 70, ty - 220, tx + 70, ty + 220), radius=24, fill=(*BLUE, 200), outline=WHITE, width=4)
    draw.line((sx + 90, sy, tx - 90, ty), fill=(250, 204, 21), width=8)
    draw.line((sx + 160, sy + 110, tx - 160, ty + 110), fill=WHITE, width=4)
    draw.text((1540, sy + 145), "合法射距：20-80cm", font=F_BODY, fill=INK)
    draw_wrapped(draw, (420, 1680), "这套规则避免瞬间擦边、穿模、错误遮挡造成虚假命中。", 2900, F_H2, fill=(203, 213, 225))


def draw_system(draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((170, 235), "ROS2 + IsaacLab：同一套规则契约", font=F_TITLE, fill=WHITE)
    nodes = [
        ((250, 620, 950, 850), "ROS2 真实机器人接口", "vision / Nav2 / EKF / cmd_vel / shooter", BLUE),
        ((1240, 620, 1940, 850), "规则环境", "target owner / blocker / dwell / scoring", GREEN),
        ((2230, 620, 2930, 850), "IsaacLab 回放", "physics / renderer / three-view video", PURPLE),
        ((1240, 1180, 1940, 1410), "World Model + SAC Flow", "object tokens / self-play / twin-Q", YELLOW),
        ((2230, 1180, 2930, 1410), "评估审计", "128 episodes / hard violations / CSV", RED),
    ]
    for box, title, sub, color in nodes:
        rounded(draw, box, (15, 23, 42, 235), color, 30, 4)
        draw.text((box[0] + 40, box[1] + 46), title, font=F_H2, fill=WHITE)
        draw_wrapped(draw, (box[0] + 40, box[1] + 126), sub, box[2] - box[0] - 80, F_SMALL, fill=(203, 213, 225))
    arrows = [((950, 735), (1240, 735)), ((1940, 735), (2230, 735)), ((1590, 850), (1590, 1180)), ((1940, 1295), (2230, 1295)), ((2580, 1180), (2580, 850))]
    for a, b in arrows:
        draw.line((a[0], a[1], b[0], b[1]), fill=WHITE, width=5)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        s = 28
        draw.polygon([b, (b[0] - s * math.cos(ang - 0.45), b[1] - s * math.sin(ang - 0.45)), (b[0] - s * math.cos(ang + 0.45), b[1] - s * math.sin(ang + 0.45))], fill=WHITE)
    draw_wrapped(draw, (380, 1620), "训练、评估、回放不各说各话：命中、遮挡、碰撞、得分和终局全部走同一套契约。", 3060, F_H2, fill=INK)


def draw_world_model(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float, ext: LoopVideo | None) -> None:
    if ext is not None:
        bg = fit_frame(ext.read_at(t), (W, H)).convert("RGBA")
        bg = bg.filter(ImageFilter.GaussianBlur(10))
        bg.putalpha(70)
        img.alpha_composite(bg)
    draw.text((170, 220), "对象中心世界模型", font=F_TITLE, fill=WHITE)
    draw_wrapped(draw, (176, 330), "策略看到的是对象和关系，而不是一串难解释的扁平状态。", 2200, F_BODY, fill=(203, 213, 225))
    left = (240, 620, 1290, 1560)
    rounded(draw, left, (15, 23, 42, 236), GREEN, 30, 3)
    draw.text((300, 670), "Arena State", font=F_H2, fill=WHITE)
    objects = [
        ("yellow_robot", YELLOW), ("blue_robot", BLUE), ("red_box", RED), ("targets", GREEN),
        ("base_blockers", PURPLE), ("laser_line", (250, 204, 21)), ("score_state", WHITE), ("timer", MUTED),
    ]
    for i, (name, color) in enumerate(objects):
        y = 790 + i * 82
        draw.rounded_rectangle((320, y, 1160, y + 56), radius=18, fill=(30, 41, 59, 230), outline=color, width=2)
        draw.ellipse((350, y + 15, 376, y + 41), fill=color)
        draw.text((410, y + 8), name, font=F_SMALL, fill=INK)
    # Graph
    center = (2300, 1110)
    token_pos = []
    for i, (name, color) in enumerate(objects):
        ang = 2 * math.pi * i / len(objects) + t * 0.06
        r = 420 + 35 * math.sin(t * 0.7 + i)
        x = int(center[0] + r * math.cos(ang))
        y = int(center[1] + r * math.sin(ang))
        token_pos.append((x, y, color, name))
    for i, a in enumerate(token_pos):
        for j, b in enumerate(token_pos):
            if j <= i:
                continue
            if (i + j) % 3 == 0:
                draw.line((a[0], a[1], b[0], b[1]), fill=(71, 85, 105, 130), width=2)
    for x, y, color, name in token_pos:
        draw.ellipse((x - 54, y - 54, x + 54, y + 54), fill=(*color[:3], 230), outline=WHITE, width=3)
    draw.text((1960, 480), "Latent Object Graph", font=F_H2, fill=WHITE)
    preds = ["预测下一状态", "预测规则事件", "辅助策略学习", "降低虚假捷径"]
    for i, label in enumerate(preds):
        x = 2920
        y = 700 + i * 150
        rounded(draw, (x, y, x + 650, y + 92), (15, 23, 42, 230), GREEN if i < 3 else RED, 22, 3)
        draw.text((x + 44, y + 24), label, font=F_BODY, fill=INK)


def draw_flow(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float, ext: LoopVideo | None) -> None:
    if ext is not None:
        bg = fit_frame(ext.read_at(t + 4.0), (W, H)).convert("RGBA").filter(ImageFilter.GaussianBlur(12))
        bg.putalpha(58)
        img.alpha_composite(bg)
    draw.text((170, 220), "SAC Flow / PolicyFlow 自博弈", font=F_TITLE, fill=WHITE)
    stages = [
        ((190, 660, 810, 900), "Object State", "对象 token + 关系边", GREEN),
        ((1120, 480, 1800, 760), "Yellow FlowActor", "多条黄方候选路线", YELLOW),
        ((1120, 1000, 1800, 1280), "Blue FlowActor", "多条蓝方候选路线", BLUE),
        ((2160, 720, 2860, 1040), "Centralized Twin-Q", "评估双方交互长期价值", PURPLE),
        ((3100, 720, 3660, 1040), "Replay + Update", "自博弈采样更新", RED),
    ]
    for box, title, sub, color in stages:
        rounded(draw, box, (15, 23, 42, 235), color, 30, 4)
        draw.text((box[0] + 42, box[1] + 48), title, font=F_H2, fill=WHITE)
        draw_wrapped(draw, (box[0] + 42, box[1] + 130), sub, box[2] - box[0] - 84, F_SMALL, fill=(203, 213, 225))
    # Flow curves
    for lane, color, y0 in [(0, YELLOW, 620), (1, BLUE, 1160)]:
        for k in range(6):
            pts = []
            for s in range(100):
                x = 810 + s * 13
                y = y0 + (k - 2.5) * 34 + 95 * math.sin(s * 0.055 + k + t * 0.6 + lane)
                pts.append((x, y))
            draw.line(pts, fill=(*color, 190), width=5)
    draw.line((1800, 620, 2160, 820), fill=WHITE, width=5)
    draw.line((1800, 1140, 2160, 940), fill=WHITE, width=5)
    draw.line((2860, 880, 3100, 880), fill=WHITE, width=5)
    draw_wrapped(draw, (360, 1550), "Flow 的作用：保留多路线策略分布，让黄蓝双方在自博弈中形成不同节奏，而不是过早塌缩到一条固定路线。", 3120, F_H2, fill=INK)


def draw_microaim(draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((170, 230), "微调瞄准：小动作解决“到点打不中”", font=F_TITLE, fill=WHITE)
    cx, cy = 1420, 1120
    target = (2720, 1020)
    angle = -0.20 + 0.40 * math.sin(t * 1.3)
    robot = [(cx + 120 * math.cos(angle + a), cy + 90 * math.sin(angle + a)) for a in [0, 2.35, -2.35]]
    draw.polygon(robot, fill=(*YELLOW, 235), outline=WHITE)
    draw.arc((cx - 260, cy - 260, cx + 260, cy + 260), start=-28, end=28, fill=YELLOW, width=8)
    draw.text((cx - 330, cy + 310), "慢速小角度扫描", font=F_BODY, fill=INK)
    draw.rounded_rectangle((target[0] - 85, target[1] - 240, target[0] + 85, target[1] + 240), radius=28, fill=(*BLUE, 210), outline=WHITE, width=4)
    ray_end = (target[0] - 90, int(target[1] + 120 * math.sin(t * 1.3)))
    draw.line((cx + 130, cy, ray_end[0], ray_end[1]), fill=(250, 204, 21), width=8)
    draw.line((cx - 80, cy + 230, cx + 230, cy + 230), fill=GREEN, width=6)
    draw.polygon([(cx + 230, cy + 230), (cx + 188, cy + 208), (cx + 188, cy + 252)], fill=GREEN)
    draw.text((cx - 80, cy + 260), "侧向微调一点，再重新对准", font=F_SMALL, fill=GREEN)
    checks = ["20-80cm 合法距离", "视线无遮挡", "驻留时间 > 0.8s", "失败则不报告命中"]
    x, y = 250, 710
    for i, label in enumerate(checks):
        color = GREEN if i < 3 else RED
        rounded(draw, (x, y, x + 700, y + 90), (15, 23, 42, 235), color, 22, 3)
        draw.text((x + 40, y + 24), label, font=F_BODY, fill=INK)
        y += 124


def draw_eval(draw: ImageDraw.ImageDraw, stats: dict) -> None:
    c = stats.get("contract", {})
    r = stats.get("replay", {})
    draw.text((170, 220), "评估不是只看 reward", font=F_TITLE, fill=WHITE)
    metrics = [
        ("episodes", f"{c.get('episodes', 128)}"),
        ("yellow_win_rate", f"{100 * c.get('yellow_win_rate', 0.4922):.1f}%"),
        ("blue_win_rate", f"{100 * c.get('blue_win_rate', 0.5078):.1f}%"),
        ("draw_rate", f"{100 * c.get('draw_rate', 0.0):.1f}%"),
        ("mean_time", f"{c.get('mean_episode_time_s', 30.8148):.1f}s"),
        ("hard_violations", f"{r.get('hard_violations', 0)}"),
        ("robot_contacts", f"{r.get('robot_contacts_per_episode', 0):.2f}/ep"),
        ("normal_hits/ep", f"{r.get('normal_hits_per_episode', 3.75):.2f}"),
    ]
    x0, y0 = 260, 520
    for i, (name, value) in enumerate(metrics):
        x = x0 + (i % 4) * 850
        y = y0 + (i // 4) * 330
        color = [GREEN, YELLOW, BLUE, PURPLE][i % 4]
        rounded(draw, (x, y, x + 720, y + 220), (15, 23, 42, 238), color, 28, 4)
        draw.text((x + 46, y + 42), name, font=F_SMALL, fill=MUTED)
        draw.text((x + 46, y + 102), value, font=F_TITLE, fill=WHITE)
    dist = c.get("normal_hit_count_distribution", {}).get("yellow", {"1": 0.0078, "2": 0.7188, "3": 0.2656, "4": 0.0078})
    draw.text((320, 1310), "普通靶击倒数量分布（黄方）", font=F_H2, fill=WHITE)
    bx, by = 320, 1430
    for i, k in enumerate(["1", "2", "3", "4"]):
        val = float(dist.get(k, 0))
        h = int(380 * val)
        x = bx + i * 240
        draw.rectangle((x, by + 380 - h, x + 150, by + 380), fill=(*GREEN, 230))
        draw.text((x + 42, by + 410), k, font=F_SMALL, fill=INK)
        draw.text((x + 12, by + 340 - h), f"{val * 100:.1f}%", font=F_TINY, fill=INK)
    draw_wrapped(draw, (1500, 1370), "可复核交付：JSON / CSV 指标、严格回放审计、三视角完整 MP4。视频只是展示层，真正关键是审计链路。", 1950, F_H2, fill=(203, 213, 225))


def render_frame(t: float, videos: dict[str, LoopVideo], ext: LoopVideo | None, stats: dict) -> Image.Image:
    seg = next((s for s in SEGMENTS if s.contains(t)), SEGMENTS[-1])
    img, draw = base_canvas(t, seg)
    u = seg.u(t)
    if seg.key == "three_open":
        draw_three_view(img, draw, videos, t, headline=True)
    elif seg.key == "arena":
        draw_arena_map(img, draw, t, u)
    elif seg.key == "physics":
        draw_physics(img, draw, t, u)
    elif seg.key == "shooting":
        draw_shooting(draw, t, u)
    elif seg.key == "system":
        draw_system(draw, t, u)
    elif seg.key == "world_model":
        draw_world_model(img, draw, t, u, ext)
    elif seg.key == "flow":
        draw_flow(img, draw, t, u, ext)
    elif seg.key == "microaim":
        draw_microaim(draw, t, u)
    elif seg.key == "eval":
        draw_eval(draw, stats)
    else:
        draw_three_view(img, draw, videos, t, headline=False)
        draw.text((260, 1540), "规则可信 · 物理可信 · 训练可信 · 回放可信", font=F_TITLE, fill=WHITE)
    if seg.key not in {"three_open", "ending"}:
        draw_footer_caption(draw, seg)
    return img.convert("RGB")


def mux_audio(video_path: Path, wav_path: Path, out_path: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(wav_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    log("Generating Chinese narration...")
    audio_duration = generate_tts(NARRATION_WAV)
    duration = max(MIN_DURATION, audio_duration + 8.0)
    log(f"Narration duration: {audio_duration:.2f}s; video duration: {duration:.2f}s")
    stats = load_stats()
    available_ai = [p for p in EXTERNAL_AI_CLIPS if p.exists() and p.stat().st_size > 50000]
    log(f"External AI clips available: {[str(p.relative_to(ROOT)) for p in available_ai]}")

    with tempfile.TemporaryDirectory(prefix="rcvrl_expert_video_") as tmp_name:
        tmp = Path(tmp_name)
        copied = copy_inputs_to_temp(tmp)
        videos = {key: LoopVideo(path) for key, path in copied.items()}
        ext = LoopVideo(available_ai[0]) if available_ai else None
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            str(TMP_VIDEO),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        total = int(math.ceil(duration * FPS))
        try:
            for i in range(total):
                t = i / FPS
                frame = render_frame(t, videos, ext, stats)
                assert proc.stdin is not None
                proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                if i % (FPS * 10) == 0:
                    log(f"Rendered {t:.1f}s / {duration:.1f}s")
        finally:
            if proc.stdin:
                proc.stdin.close()
            rc = proc.wait()
            for v in videos.values():
                v.close()
            if ext is not None:
                ext.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg video render failed with code {rc}")

    log("Muxing narration...")
    if NARRATION_WAV.exists() and NARRATION_WAV.stat().st_size > 1000:
        mux_audio(TMP_VIDEO, NARRATION_WAV, OUTPUT)
    else:
        shutil.copy2(TMP_VIDEO, OUTPUT)
    log(f"Saved: {OUTPUT}")

    # Save a small provenance note for reviewers.
    provenance = {
        "output": str(OUTPUT),
        "duration_s_target": duration,
        "fps": FPS,
        "resolution": [W, H],
        "source_replays": {k: str(v) for k, v in INPUTS.items()},
        "external_ai_clips_used": [str(p) for p in available_ai],
        "external_ai_note": "LTX-2.3 / Hugging Face generated clips are used only as short abstract no-text transition backgrounds; factual claims use project replay and audit data.",
        "reference_videos": [
            "https://www.bilibili.com/video/BV1fj6vBfEnu/",
            "https://www.bilibili.com/video/BV1NCS4BkEt7/",
            "https://www.bilibili.com/video/BV1buxDzzE9P/",
        ],
    }
    (MEDIA_DIR / "reviewer_expert_5min_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
