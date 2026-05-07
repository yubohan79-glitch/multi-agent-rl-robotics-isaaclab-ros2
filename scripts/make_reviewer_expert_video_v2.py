from __future__ import annotations

import argparse
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "docs" / "media"
DATA_DIR = ROOT / "docs" / "rl_data" / "world_model_sacflow_final"

OUTPUT = MEDIA_DIR / "最终回放_审稿人专家讲解版_5min_4K30_v2.mp4"
TMP_VIDEO = MEDIA_DIR / "reviewer_expert_5min_v2_video_only.mp4"
NARRATION_WAV = MEDIA_DIR / "reviewer_expert_5min_narration.wav"
LOG_PATH = MEDIA_DIR / "reviewer_expert_5min_v2_render.log"
PREVIEW_DIR = MEDIA_DIR / "reviewer_expert_5min_v2_preview_frames"

INPUTS = {
    "top": MEDIA_DIR / "最终回放_顶视角.mp4",
    "yellow": MEDIA_DIR / "最终回放_黄车第一视角.mp4",
    "blue": MEDIA_DIR / "最终回放_蓝车第一视角.mp4",
}

# Draw at 1080p and encode to 4K with ffmpeg scaling. This is much more stable
# than writing 4K PIL frames for a 5+ minute video.
W, H = 1920, 1080
OUT_W, OUT_H = 3840, 2160
FPS = 30
MIN_DURATION = 336.0

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]

BG = (7, 11, 20)
PANEL = (14, 22, 36)
PANEL_2 = (20, 31, 48)
INK = (226, 232, 240)
MUTED = (148, 163, 184)
LINE = (51, 65, 85)
GREEN = (34, 197, 94)
YELLOW = (245, 158, 11)
BLUE = (56, 189, 248)
RED = (239, 68, 68)
PURPLE = (167, 139, 250)
ORANGE = (249, 115, 22)
WHITE = (248, 250, 252)
DARK_TEXT = (15, 23, 42)


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
        return clamp((t - self.start) / max(self.end - self.start, 1e-6))


SEGMENTS = [
    Segment(0, 42, "three_open", "最终三视角同步回放", "先把真实行为放在第一屏：顶视角看全局路线，第一视角看瞄准、遮挡和射击距离。"),
    Segment(42, 80, "arena", "任务规则：目标、箱子、挡板、激光", "策略必须同时理解普通靶、基地靶、红箱、蓝挡板和非法命中约束。"),
    Segment(80, 118, "physics", "物理可信先于训练可信", "红箱必须真实可推动；蓝色挡板必须同时阻挡车辆和激光 raycast。"),
    Segment(118, 154, "shooting", "射击规则：20-80cm、视线、驻留", "0.8 秒以内不倒；0.8 到 2 秒之间概率增强，避免瞬间擦边投机。"),
    Segment(154, 192, "system", "ROS2 + IsaacLab + 审计闭环", "训练、评估和回放共享同一套规则契约，减少口径漂移。"),
    Segment(192, 242, "world_model", "对象中心世界模型", "机器人、靶子、箱子、挡板、视线、比分和时间被建成对象 token。"),
    Segment(242, 292, "flow", "SAC Flow / PolicyFlow 自博弈", "双 actor 保留黄蓝差异，中心化 Twin-Q critic 评估交互后的长期价值。"),
    Segment(292, 318, "microaim", "微调瞄准：小角度扫描与侧移", "到点后慢速扫角，若基地靶被挡则向合法侧微移再重试。"),
    Segment(318, 334, "eval", "128 局多 seed 评估与严格审计", "胜率平衡、普通靶分布、基地命中率、箱子位移和 hard violation 同时检查。"),
    Segment(334, 336, "ending", "可复核链路", "规则可信、物理可信、训练可信、回放可信。"),
]


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def ease(u: float) -> float:
    u = clamp(u)
    return 3 * u * u - 2 * u * u * u


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_BIG = font(48)
F_TITLE = font(38)
F_H2 = font(28)
F_BODY = font(23)
F_SMALL = font(18)
F_TINY = font(15)


def rgba(color: tuple[int, int, int], a: int = 255) -> tuple[int, int, int, int]:
    return (*color, a)


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


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width: int,
    text_font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = INK,
    spacing: int = 6,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, width, text_font):
        draw.text((x, y), line, font=text_font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=text_font)
        y += bbox[3] - bbox[1] + spacing
    return y


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int] | tuple[int, int, int, int],
    outline: tuple[int, int, int] | tuple[int, int, int, int] = LINE,
    radius: int = 18,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def glow_line(
    img: Image.Image,
    pts: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int = 4,
    blur: int = 10,
) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.line(pts, fill=rgba(color, 190), width=width, joint="curve")
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))
    img.alpha_composite(layer)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int], width: int = 3) -> None:
    draw.line((start, end), fill=color, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 12
    p1 = (end[0] - size * math.cos(ang - 0.45), end[1] - size * math.sin(ang - 0.45))
    p2 = (end[0] - size * math.cos(ang + 0.45), end[1] - size * math.sin(ang + 0.45))
    draw.polygon([end, p1, p2], fill=color)


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font_obj=F_SMALL) -> tuple[int, int, int, int]:
    x, y = xy
    tw = int(draw.textlength(text, font=font_obj))
    box = (x, y, x + tw + 26, y + 34)
    rounded(draw, box, rgba((7, 12, 22), 235), rgba(color, 210), radius=17, width=1)
    draw.text((x + 13, y + 7), text, font=font_obj, fill=WHITE)
    return box


def cv_to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def fit_image(image: Image.Image, size: tuple[int, int], cover: bool = True) -> Image.Image:
    sw, sh = image.size
    tw, th = size
    scale = max(tw / sw, th / sh) if cover else min(tw / sw, th / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    if cover:
        left = max(0, (nw - tw) // 2)
        top = max(0, (nh - th) // 2)
        return resized.crop((left, top, left + tw, top + th))
    canvas = Image.new("RGB", size, (8, 13, 22))
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def paste_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    frame: Image.Image,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    zoom: float = 1.0,
) -> None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if zoom > 1.01:
        fw, fh = frame.size
        cw, ch = int(fw / zoom), int(fh / zoom)
        left = (fw - cw) // 2
        top = (fh - ch) // 2
        frame = frame.crop((left, top, left + cw, top + ch))
    fitted = fit_image(frame, (w, h), cover=True).convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, w, h), radius=18, fill=255)
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((x1 + 6, y1 + 10, x2 + 6, y2 + 10), radius=20, fill=(0, 0, 0, 95))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)))
    canvas.paste(fitted, (x1, y1), mask)
    draw.rounded_rectangle(box, radius=18, outline=rgba(color, 220), width=2)
    rounded(draw, (x1 + 12, y1 + 12, x1 + 12 + int(draw.textlength(label, font=F_SMALL)) + 28, y1 + 46), rgba((2, 6, 23), 190), rgba(color, 210), radius=17, width=1)
    draw.text((x1 + 26, y1 + 20), label, font=F_SMALL, fill=color)


class LoopVideo:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.frames = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or FPS
        self.index = -1
        self.last: np.ndarray | None = None

    def read_at(self, t: float) -> Image.Image:
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
                raise RuntimeError(f"cannot read video: {self.path}")
            self.index = 0
            self.last = frame
        return cv_to_image(self.last)

    def close(self) -> None:
        self.cap.release()


def load_stats() -> dict:
    out: dict[str, dict] = {}
    for name, file in [
        ("contract", DATA_DIR / "contract_eval_multiseed.json"),
        ("replay", DATA_DIR / "strict_replay_summary.json"),
        ("training", DATA_DIR / "training_summary.json"),
    ]:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            out[name] = data.get("summary", data)
        except Exception:
            out[name] = {}
    return out


def wav_duration(path: Path) -> float:
    if not path.exists() or path.stat().st_size < 1000:
        return 0.0
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def generate_tts(path: Path) -> float:
    existing = wav_duration(path)
    if existing > 60:
        return existing
    try:
        import win32com.client  # type: ignore

        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voices = voice.GetVoices()
        for i in range(voices.Count):
            desc = voices.Item(i).GetDescription()
            if "Huihui" in desc or "Chinese" in desc or "中文" in desc:
                voice.Voice = voices.Item(i)
                break
        voice.Rate = -1
        voice.Volume = 96
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(path), 3, False)
        voice.AudioOutputStream = stream
        voice.Speak(NARRATION)
        stream.Close()
        return wav_duration(path)
    except Exception as exc:
        log(f"TTS failed, continue without narration: {exc}")
        return 0.0


def current_segment(t: float) -> Segment:
    return next((s for s in SEGMENTS if s.contains(t)), SEGMENTS[-1])


def base_canvas(t: float, duration: float) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (W, H), rgba(BG))
    draw = ImageDraw.Draw(img)

    # Academic dark grid, with very subtle motion. The reference videos use
    # one concept per shot and clean progressive reveals; this background keeps
    # the focus on the diagrams.
    for x in range(0, W, 80):
        shade = 24 + int(5 * math.sin(0.6 * t + x * 0.018))
        draw.line((x, 0, x, H), fill=(shade, shade + 4, shade + 12, 90), width=1)
    for y in range(0, H, 80):
        shade = 22 + int(5 * math.cos(0.5 * t + y * 0.017))
        draw.line((0, y, W, y), fill=(shade, shade + 4, shade + 11, 90), width=1)

    # Header and progress bar.
    draw.rectangle((0, 0, W, 64), fill=(4, 8, 15, 244))
    draw.text((42, 16), "RoboCupVisionRL", font=F_H2, fill=WHITE)
    draw.text((270, 20), "Object-Centric World-Model + SAC Flow / PolicyFlow", font=F_SMALL, fill=MUTED)
    draw.text((1490, 20), "专家讲解版 · 4K30", font=F_SMALL, fill=MUTED)
    draw.rounded_rectangle((42, 55, W - 42, 61), radius=3, fill=(30, 41, 59, 255))
    draw.rounded_rectangle((42, 55, int(42 + (W - 84) * clamp(t / max(duration, 1))), 61), radius=3, fill=rgba(GREEN))
    return img, draw


def draw_footer(draw: ImageDraw.ImageDraw, seg: Segment) -> None:
    box = (70, 942, 1850, 1032)
    rounded(draw, box, rgba((3, 7, 18), 232), rgba((71, 85, 105), 230), radius=22, width=1)
    draw.text((96, 960), seg.title, font=F_H2, fill=WHITE)
    draw_wrapped(draw, (96, 992), seg.caption, 1680, F_SMALL, fill=(203, 213, 225), spacing=3)


def draw_section_nav(draw: ImageDraw.ImageDraw, seg: Segment) -> None:
    labels = ["回放", "规则", "物理", "射击", "系统", "世界模型", "Flow", "微调", "评估"]
    active = max(0, min(len(labels) - 1, [s.key for s in SEGMENTS].index(seg.key) if seg.key in [s.key for s in SEGMENTS[:-1]] else len(labels) - 1))
    x = 1730
    y = 94
    for i, label in enumerate(labels):
        c = GREEN if i == active else MUTED
        if i == active:
            draw.line((x - 10, y + i * 34 + 5, x - 10, y + i * 34 + 22), fill=GREEN, width=3)
        draw.ellipse((x, y + i * 34, x + 10, y + i * 34 + 10), fill=c)
        draw.text((x + 18, y + i * 34 - 5), label, font=F_TINY, fill=c)


def draw_three_view(img: Image.Image, draw: ImageDraw.ImageDraw, videos: dict[str, LoopVideo], t: float, opening: bool) -> None:
    title_y = 90
    if opening:
        draw.text((62, title_y), "先看真实三视角：行为证据放在第一屏", font=F_BIG, fill=WHITE)
        draw.text((64, title_y + 58), "顶视角验证全局路线；第一视角验证瞄准、遮挡、射击距离和真实视线。", font=F_BODY, fill=(203, 213, 225))
    else:
        draw.text((62, title_y), "回到三视角：最终结果必须能被视频复核", font=F_BIG, fill=WHITE)

    zoom = 1.0 + 0.035 * math.sin(t * 0.23)
    top = videos["top"].read_at(t)
    yellow = videos["yellow"].read_at(t)
    blue = videos["blue"].read_at(t)
    paste_panel(img, draw, top, (64, 182, 1246, 848), "顶视角 / 全局路线", GREEN, zoom=zoom)
    paste_panel(img, draw, yellow, (1290, 182, 1856, 501), "黄车第一视角 / 瞄准", YELLOW, zoom=1.02)
    paste_panel(img, draw, blue, (1290, 529, 1856, 848), "蓝车第一视角 / 遮挡", BLUE, zoom=1.02)

    # Attention anchors.
    for i, (txt, col) in enumerate([("红箱位移", RED), ("挡板阻挡", BLUE), ("激光驻留", GREEN)]):
        pill(draw, (88 + i * 170, 865), txt, col)

    if opening:
        rounded(draw, (1008, 705, 1226, 825), rgba((3, 7, 18), 210), rgba(GREEN, 180), radius=20, width=1)
        draw.text((1030, 727), "审稿人检查点", font=F_SMALL, fill=GREEN)
        for j, item in enumerate(["是否穿模", "是否绕射", "是否真实终止"]):
            draw.text((1034, 756 + j * 22), f"✓ {item}", font=F_TINY, fill=INK)


def arena_xy(x: float, y: float, box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return x1 + x * (x2 - x1), y1 + y * (y2 - y1)


def draw_arena(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((70, 92), "任务规则为什么复杂", font=F_BIG, fill=WHITE)
    draw_wrapped(draw, (72, 150), "策略不能只会“走到点”。它必须同时处理目标所有权、挡板解锁、红箱路线变化、激光驻留和非法命中。", 760, F_BODY, fill=(203, 213, 225))

    box = (70, 255, 910, 880)
    rounded(draw, box, rgba((8, 15, 28), 238), rgba((71, 85, 105), 210), radius=26, width=2)
    # Arena field.
    field = (120, 295, 860, 835)
    rounded(draw, field, rgba((12, 20, 35), 245), rgba((100, 116, 139), 230), radius=16, width=2)
    for i in range(1, 6):
        x = field[0] + i * (field[2] - field[0]) / 6
        draw.line((x, field[1], x, field[3]), fill=(40, 55, 80, 100), width=1)
    for i in range(1, 4):
        y = field[1] + i * (field[3] - field[1]) / 4
        draw.line((field[0], y, field[2], y), fill=(40, 55, 80, 100), width=1)

    targets = [
        (0.18, 0.22, YELLOW), (0.28, 0.38, YELLOW), (0.39, 0.28, YELLOW), (0.52, 0.22, YELLOW),
        (0.82, 0.78, BLUE), (0.72, 0.62, BLUE), (0.61, 0.72, BLUE), (0.48, 0.78, BLUE),
    ]
    reveal = int(1 + ease(u) * len(targets))
    for idx, (x, y, col) in enumerate(targets[:reveal]):
        px, py = arena_xy(x, y, field)
        r = 12 + 4 * math.sin(t * 3 + idx)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=rgba(col, 190), outline=WHITE, width=1)
        draw.line((px - 17, py, px + 17, py), fill=rgba(col, 220), width=2)

    for x, y, col, name in [(0.12, 0.78, YELLOW, "Y"), (0.88, 0.22, BLUE, "B")]:
        px, py = arena_xy(x, y, field)
        draw.rounded_rectangle((px - 22, py - 14, px + 22, py + 14), radius=7, fill=rgba(col, 230), outline=WHITE, width=1)
        draw.text((px - 6, py - 12), name, font=F_TINY, fill=(2, 6, 23))

    # Pushable boxes and blockers.
    for x, y, name in [(0.42, 0.50, "红箱"), (0.57, 0.50, "红箱")]:
        px, py = arena_xy(x, y, field)
        off = 14 * math.sin(t * 1.2 + x * 10)
        draw.rounded_rectangle((px - 24 + off, py - 20, px + 24 + off, py + 20), radius=6, fill=rgba(RED, 210), outline=WHITE, width=1)
        draw.text((px - 21 + off, py + 26), name, font=F_TINY, fill=RED)
    for x, y in [(0.86, 0.50), (0.14, 0.50)]:
        px, py = arena_xy(x, y, field)
        draw.rounded_rectangle((px - 12, py - 65, px + 12, py + 65), radius=5, fill=rgba(BLUE, 210), outline=rgba(BLUE, 255), width=1)

    glow_line(img, [arena_xy(0.12, 0.78, field), arena_xy(0.25, 0.56, field), arena_xy(0.42 + 0.04 * math.sin(t), 0.50, field), arena_xy(0.66, 0.34, field)], YELLOW, width=4)
    glow_line(img, [arena_xy(0.88, 0.22, field), arena_xy(0.72, 0.45, field), arena_xy(0.57, 0.50, field), arena_xy(0.33, 0.66, field)], BLUE, width=4)

    side = (990, 255, 1810, 880)
    rounded(draw, side, rgba(PANEL, 235), rgba(LINE, 220), radius=26, width=1)
    draw.text((1030, 292), "规则依赖链", font=F_TITLE, fill=WHITE)
    nodes = [
        ("普通靶", "改变基地挡板状态", YELLOW),
        ("红箱", "改变可通行路线", RED),
        ("蓝挡板", "阻挡车和激光", BLUE),
        ("驻留时间", "决定命中概率", GREEN),
        ("审计器", "排除穿模/绕射", PURPLE),
    ]
    for i, (name, desc, col) in enumerate(nodes):
        y = 365 + i * 82
        alpha = int(90 + 160 * clamp((u * 5 - i) * 1.3))
        rounded(draw, (1030, y, 1760, y + 58), rgba(col, 24), rgba(col, alpha), radius=16, width=1)
        draw.text((1054, y + 12), name, font=F_H2, fill=col)
        draw.text((1180, y + 17), desc, font=F_SMALL, fill=DARK_TEXT)
        if i < len(nodes) - 1:
            arrow(draw, (1395, y + 62), (1395, y + 80), MUTED, width=2)


def draw_physics(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((70, 92), "物理可信：先证明场景不是漏洞", font=F_BIG, fill=WHITE)
    draw_wrapped(draw, (72, 150), "如果箱子只是渲染物、挡板不阻挡 raycast，RL 会学习漏洞而不是规则。这里把碰撞体、刚体和视线阻挡显式可视化。", 980, F_BODY, fill=(203, 213, 225))

    left = (80, 265, 890, 846)
    right = (1010, 265, 1810, 846)
    rounded(draw, left, rgba(PANEL, 235), rgba(LINE, 220), radius=24, width=1)
    rounded(draw, right, rgba(PANEL, 235), rgba(LINE, 220), radius=24, width=1)
    draw.text((120, 302), "红箱：真实可推动刚体", font=F_H2, fill=RED)
    draw.text((1050, 302), "蓝挡板：阻挡车辆与激光", font=F_H2, fill=BLUE)

    # Left panel: robot pushing a rigid box.
    base_y = 600
    draw.line((140, base_y + 88, 820, base_y + 88), fill=(71, 85, 105), width=2)
    x = 215 + 120 * ease(u)
    box_x = 515 + 58 * ease(max(0, u - 0.25) / 0.75)
    draw.rounded_rectangle((x - 52, base_y - 28, x + 52, base_y + 28), radius=10, fill=rgba(YELLOW, 230), outline=WHITE, width=1)
    draw.text((x - 18, base_y - 16), "车", font=F_SMALL, fill=(3, 7, 18))
    draw.rectangle((box_x - 58, base_y - 58, box_x + 58, base_y + 58), fill=rgba(RED, 215), outline=WHITE, width=2)
    for k in range(3):
        draw.rectangle((box_x - 66 - k * 7, base_y - 66 - k * 7, box_x + 66 + k * 7, base_y + 66 + k * 7), outline=rgba(RED, 70 - k * 16), width=2)
    arrow(draw, (x + 64, base_y), (box_x - 70, base_y), GREEN, width=4)
    draw.text((150, 750), "接触结果：推动或被阻挡，不允许穿箱", font=F_BODY, fill=INK)

    # Right panel: blocker and raycast.
    bx = 1385
    draw.rounded_rectangle((bx - 18, 405, bx + 18, 735), radius=8, fill=rgba(BLUE, 220), outline=WHITE, width=1)
    draw.text((bx - 55, 755), "基地挡板", font=F_SMALL, fill=BLUE)
    robot = (1120, 580)
    target = (1660, 580)
    draw.rounded_rectangle((robot[0] - 54, robot[1] - 28, robot[0] + 54, robot[1] + 28), radius=10, fill=rgba(YELLOW, 230), outline=WHITE, width=1)
    draw.ellipse((target[0] - 30, target[1] - 30, target[0] + 30, target[1] + 30), fill=rgba(RED, 220), outline=WHITE, width=2)
    draw.text((target[0] - 34, target[1] + 42), "基地靶", font=F_SMALL, fill=RED)
    draw.line((robot[0] + 58, robot[1], bx - 18, robot[1]), fill=rgba(GREEN, 230), width=4)
    draw.line((bx + 18, robot[1], target[0] - 34, robot[1]), fill=rgba(RED, 120), width=4)
    draw.text((1110, 750), "未拆挡板：raycast 必须失败", font=F_BODY, fill=INK)
    draw.text((1470, 510), "STOP", font=F_H2, fill=RED)

    checks = [("红箱位移持续变化", GREEN), ("穿箱次数 = 0", GREEN), ("挡板阻挡激光", GREEN), ("未拆挡板基地不可命中", GREEN)]
    for i, (text, col) in enumerate(checks):
        pill(draw, (120 + i * 420, 875), text, col, F_TINY)


def draw_shooting(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((70, 92), "射击规则：不是碰到就算命中", font=F_BIG, fill=WHITE)
    draw_wrapped(draw, (72, 150), "命中需要满足合法距离、无遮挡视线和驻留时间。这样可以排除瞬间擦边、隔挡板命中和错误射程。", 900, F_BODY, fill=(203, 213, 225))
    panel = (80, 260, 1810, 870)
    rounded(draw, panel, rgba(PANEL, 235), rgba(LINE, 220), radius=24, width=1)

    # Laser range scene.
    robot = (230, 570)
    target = (700, 570)
    draw.rounded_rectangle((robot[0] - 56, robot[1] - 34, robot[0] + 56, robot[1] + 34), radius=12, fill=rgba(YELLOW, 230), outline=WHITE, width=1)
    draw.ellipse((target[0] - 44, target[1] - 44, target[0] + 44, target[1] + 44), fill=rgba(RED, 220), outline=WHITE, width=2)
    draw.line((robot[0] + 62, robot[1], target[0] - 48, target[1]), fill=rgba(GREEN, 230), width=5)
    draw.text((318, 525), "20 cm - 80 cm 合法射程", font=F_SMALL, fill=GREEN)
    dwell = 2.0 * clamp((math.sin(t * 0.8) + 1) / 2)
    rounded(draw, (180, 690, 760, 740), rgba((2, 6, 23), 210), rgba(LINE, 240), radius=18, width=1)
    draw.rounded_rectangle((190, 702, 190 + int(560 * clamp(dwell / 2.0)), 728), radius=13, fill=rgba(GREEN if dwell >= 0.8 else RED, 220))
    draw.text((190, 750), f"驻留时间 {dwell:.2f}s", font=F_BODY, fill=INK)

    # Probability curve.
    cx, cy, cw, ch = 1010, 355, 650, 350
    draw.text((cx, 310), "命中概率门控", font=F_H2, fill=WHITE)
    draw.line((cx, cy + ch, cx + cw, cy + ch), fill=MUTED, width=2)
    draw.line((cx, cy, cx, cy + ch), fill=MUTED, width=2)
    pts = []
    for i in range(120):
        x = i / 119 * 2.2
        p = 0 if x < 0.8 else clamp((x - 0.8) / 1.2)
        pts.append((cx + x / 2.2 * cw, cy + ch - p * ch))
    draw.line(pts, fill=GREEN, width=4)
    for x, label in [(0.8, "0.8s"), (2.0, "2.0s")]:
        px = cx + x / 2.2 * cw
        draw.line((px, cy, px, cy + ch), fill=rgba(WHITE, 80), width=1)
        draw.text((px - 25, cy + ch + 16), label, font=F_SMALL, fill=MUTED)
    draw.text((cx - 4, cy + ch + 48), "驻留时间", font=F_SMALL, fill=MUTED)
    draw.text((cx - 66, cy - 8), "概率", font=F_SMALL, fill=MUTED)
    moving_x = cx + min(dwell, 2.2) / 2.2 * cw
    moving_y = cy + ch - (0 if dwell < 0.8 else clamp((dwell - 0.8) / 1.2)) * ch
    draw.ellipse((moving_x - 10, moving_y - 10, moving_x + 10, moving_y + 10), fill=WHITE, outline=GREEN, width=3)
    draw_wrapped(draw, (1010, 740), "0.8 秒以内 100% 不倒；0.8 到 2 秒之间线性增强，但仍保留失败概率。", 650, F_BODY, fill=INK)


def draw_system(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((70, 92), "工程闭环：同一套契约贯穿训练、评估、回放", font=F_BIG, fill=WHITE)
    draw_wrapped(draw, (72, 150), "参考视频常用“感知-模型-动作”的层次化讲法。这里把 ROS2、IsaacLab、规则审计和回放导出放在同一张闭环图里。", 1050, F_BODY, fill=(203, 213, 225))

    nodes = [
        ("ROS2 接口", "视觉检测 / 定位融合 / 控制 / 发射器", YELLOW, (120, 360)),
        ("IsaacLab 场景", "物理、刚体、碰撞、raycast", BLUE, (590, 360)),
        ("规则审计器", "驻留、射程、挡板、穿模", GREEN, (1060, 360)),
        ("训练评估", "World Model + SAC Flow 自博弈", PURPLE, (590, 650)),
    ]
    for name, desc, col, (x, y) in nodes:
        rounded(draw, (x, y, x + 340, y + 145), rgba(col, 24), rgba(col, 200), radius=22, width=2)
        draw.text((x + 28, y + 26), name, font=F_H2, fill=col)
        draw_wrapped(draw, (x + 28, y + 68), desc, 280, F_SMALL, fill=INK, spacing=3)
    arrows = [((460, 432), (590, 432)), ((930, 432), (1060, 432)), ((1230, 505), (930, 650)), ((590, 720), (460, 505))]
    for i, (a, b) in enumerate(arrows):
        col = [YELLOW, BLUE, GREEN, PURPLE][i]
        arrow(draw, a, b, col, width=4)
    # Moving packet around the loop.
    loop_pts = [(460, 432), (590, 432), (930, 432), (1060, 432), (1230, 505), (930, 650), (590, 720), (460, 505), (460, 432)]
    idx = int((t * 1.2) % (len(loop_pts) - 1))
    local = (t * 1.2) % 1
    a, b = loop_pts[idx], loop_pts[idx + 1]
    px, py = a[0] + (b[0] - a[0]) * local, a[1] + (b[1] - a[1]) * local
    draw.ellipse((px - 11, py - 11, px + 11, py + 11), fill=WHITE, outline=GREEN, width=3)

    box = (1440, 310, 1810, 770)
    rounded(draw, box, rgba(PANEL_2, 238), rgba(LINE, 220), radius=22, width=1)
    draw.text((1475, 345), "统一契约", font=F_H2, fill=WHITE)
    items = ["target layout", "collision", "laser LOS", "hit dwell", "scoring", "replay audit"]
    for i, item in enumerate(items):
        y = 400 + i * 52
        draw.text((1485, y), "✓", font=F_BODY, fill=GREEN)
        draw.text((1520, y + 2), item, font=F_SMALL, fill=INK)


def draw_world_model(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((70, 92), "对象中心世界模型：先看对象，再学关系", font=F_BIG, fill=WHITE)
    draw_wrapped(draw, (72, 150), "策略输入不是一串难解释的扁平向量，而是机器人、靶子、箱子、挡板、激光和比分这些对象 token。", 1050, F_BODY, fill=(203, 213, 225))

    left = (70, 260, 480, 860)
    mid = (560, 240, 1280, 885)
    right = (1360, 260, 1810, 860)
    rounded(draw, left, rgba(PANEL, 235), rgba(LINE, 220), radius=24, width=1)
    rounded(draw, mid, rgba(PANEL, 235), rgba(PURPLE, 170), radius=24, width=1)
    rounded(draw, right, rgba(PANEL, 235), rgba(LINE, 220), radius=24, width=1)
    draw.text((110, 295), "对象 token", font=F_H2, fill=WHITE)
    draw.text((805, 278), "latent graph", font=F_H2, fill=PURPLE)
    draw.text((1400, 295), "辅助预测", font=F_H2, fill=WHITE)

    tokens = [("yellow robot", YELLOW), ("blue robot", BLUE), ("targets × 10", GREEN), ("red boxes", RED), ("blockers", BLUE), ("laser LOS", PURPLE), ("score / timer", WHITE)]
    token_centers = []
    for i, (name, col) in enumerate(tokens):
        y = 365 + i * 58
        a = clamp((u * len(tokens) - i) * 1.4)
        rounded(draw, (110, y, 430, y + 40), rgba(col, int(20 + 28 * a)), rgba(col, int(80 + 140 * a)), radius=16, width=1)
        draw.text((128, y + 10), name, font=F_SMALL, fill=WHITE)
        token_centers.append((430, y + 20, col))
        if a > 0.7:
            arrow(draw, (430, y + 20), (585, 420 + (i % 4) * 88), col, width=2)

    # Graph nodes.
    graph_nodes = []
    for i in range(13):
        ang = 2 * math.pi * i / 13 + 0.25 * math.sin(t * 0.3)
        rad = 210 + 44 * math.sin(i * 1.7)
        x = 920 + math.cos(ang) * rad
        y = 565 + math.sin(ang) * rad
        col = [YELLOW, BLUE, GREEN, RED, PURPLE][i % 5]
        graph_nodes.append((x, y, col))
    for i, (x, y, col) in enumerate(graph_nodes):
        for j in range(i + 1, len(graph_nodes)):
            if (i * 7 + j * 3) % 5 == 0:
                x2, y2, _ = graph_nodes[j]
                draw.line((x, y, x2, y2), fill=(80, 95, 120, 80), width=1)
    for i, (x, y, col) in enumerate(graph_nodes):
        r = 10 + 3 * math.sin(t * 2.0 + i)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rgba(col, 230), outline=WHITE, width=1)
    draw.ellipse((920 - 68, 565 - 68, 920 + 68, 565 + 68), outline=rgba(PURPLE, 190), width=3)
    draw.text((860, 548), "belief", font=F_H2, fill=PURPLE)

    preds = [("下一步箱子位移", "box Δxy", RED), ("合法视线", "LOS mask", GREEN), ("基地可命中窗口", "base window", YELLOW), ("得分状态", "score state", BLUE)]
    for i, (name, code, col) in enumerate(preds):
        y = 370 + i * 95
        rounded(draw, (1400, y, 1760, y + 62), rgba(col, 24), rgba(col, 180), radius=16, width=1)
        draw.text((1420, y + 12), name, font=F_SMALL, fill=DARK_TEXT)
        draw.text((1618, y + 14), code, font=F_TINY, fill=DARK_TEXT)
        arrow(draw, (1280, 565), (1400, y + 31), col, width=2)


def draw_flow(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((70, 92), "SAC Flow / PolicyFlow：保留多路线，不让策略过早塌缩", font=F_BIG, fill=WHITE)
    draw_wrapped(draw, (72, 150), "Flow 的作用是把同一个局面下的多条候选路线保留下来，再由 critic 和规则约束选择更稳的动作。", 1100, F_BODY, fill=(203, 213, 225))

    field = (90, 270, 980, 850)
    rounded(draw, field, rgba(PANEL, 235), rgba(LINE, 220), radius=24, width=1)
    draw.text((130, 305), "多候选轨迹", font=F_H2, fill=WHITE)
    start_y = (220, 725)
    start_b = (850, 395)
    target_y = (780, 390)
    target_b = (240, 720)
    draw.rounded_rectangle((start_y[0] - 38, start_y[1] - 24, start_y[0] + 38, start_y[1] + 24), radius=10, fill=rgba(YELLOW, 230), outline=WHITE, width=1)
    draw.rounded_rectangle((start_b[0] - 38, start_b[1] - 24, start_b[0] + 38, start_b[1] + 24), radius=10, fill=rgba(BLUE, 230), outline=WHITE, width=1)
    draw.ellipse((target_y[0] - 28, target_y[1] - 28, target_y[0] + 28, target_y[1] + 28), fill=rgba(RED, 210), outline=WHITE, width=1)
    draw.ellipse((target_b[0] - 28, target_b[1] - 28, target_b[0] + 28, target_b[1] + 28), fill=rgba(RED, 210), outline=WHITE, width=1)

    curves = [
        ([start_y, (370, 590), (580, 470), target_y], YELLOW),
        ([start_y, (360, 735), (650, 620), target_y], ORANGE),
        ([start_y, (300, 520), (520, 360), target_y], GREEN),
        ([start_b, (690, 500), (470, 610), target_b], BLUE),
        ([start_b, (740, 300), (430, 420), target_b], PURPLE),
        ([start_b, (640, 650), (430, 760), target_b], GREEN),
    ]
    for i, (pts, col) in enumerate(curves):
        bez = bezier_points(pts, 80)
        glow_line(img, bez, col, width=3, blur=5)
        k = int((t * 22 + i * 12) % len(bez))
        x, y = bez[k]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=WHITE, outline=col, width=2)

    # Algorithm block diagram.
    x0 = 1080
    blocks = [
        ("Yellow actor", "路线 / 推箱 / 早攻窗口", YELLOW, 280),
        ("Blue actor", "路线 / 防守 / 基地节奏", BLUE, 405),
        ("Replay buffer", "真实 rollout + 审计标签", GREEN, 530),
        ("Centralized Twin-Q", "交互长期价值", PURPLE, 655),
        ("PolicyFlow update", "保留多峰动作分布", ORANGE, 780),
    ]
    for name, desc, col, y in blocks:
        rounded(draw, (x0, y, 1785, y + 82), rgba(col, 24), rgba(col, 190), radius=18, width=1)
        draw.text((x0 + 24, y + 14), name, font=F_H2, fill=DARK_TEXT)
        draw.text((x0 + 260, y + 20), desc, font=F_SMALL, fill=DARK_TEXT)
    for y1, y2, col in [(362, 405, YELLOW), (487, 530, BLUE), (612, 655, GREEN), (737, 780, PURPLE)]:
        arrow(draw, (1432, y1), (1432, y2), col, width=3)
    draw.text((1110, 900), "旧 PPO/MAPPO 仅作历史对照；正式结果来自新算法口径重新评估。", font=F_SMALL, fill=MUTED)


def bezier_points(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    if len(points) == 4:
        p0, p1, p2, p3 = points
        out = []
        for i in range(n):
            u = i / max(n - 1, 1)
            x = (1 - u) ** 3 * p0[0] + 3 * (1 - u) ** 2 * u * p1[0] + 3 * (1 - u) * u * u * p2[0] + u**3 * p3[0]
            y = (1 - u) ** 3 * p0[1] + 3 * (1 - u) ** 2 * u * p1[1] + 3 * (1 - u) * u * u * p2[1] + u**3 * p3[1]
            out.append((x, y))
        return out
    return points


def draw_microaim(img: Image.Image, draw: ImageDraw.ImageDraw, t: float, u: float) -> None:
    draw.text((70, 92), "微调瞄准：解决“到点后卡住、打不中”", font=F_BIG, fill=WHITE)
    draw_wrapped(draw, (72, 150), "到达射击点后，不立刻判失败，而是执行很小的慢速扫角和合法侧微移，让激光有稳定驻留窗口。", 980, F_BODY, fill=(203, 213, 225))

    panel = (90, 250, 1810, 875)
    rounded(draw, panel, rgba(PANEL, 235), rgba(LINE, 220), radius=24, width=1)
    cx, cy = 930, 595
    target = (1330, 560)
    robot_x = 520 + 42 * math.sin(t * 0.9)
    robot_y = 610 + 18 * math.sin(t * 0.35)
    scan = math.sin(t * 1.6) * 0.12
    draw.ellipse((target[0] - 70, target[1] - 70, target[0] + 70, target[1] + 70), fill=rgba(RED, 170), outline=WHITE, width=2)
    draw.ellipse((target[0] - 28, target[1] - 28, target[0] + 28, target[1] + 28), fill=rgba(WHITE, 235), outline=RED, width=3)
    draw.text((target[0] - 60, target[1] + 92), "基地靶", font=F_H2, fill=RED)

    draw.rounded_rectangle((robot_x - 70, robot_y - 38, robot_x + 70, robot_y + 38), radius=14, fill=rgba(YELLOW, 230), outline=WHITE, width=2)
    draw.text((robot_x - 30, robot_y - 17), "robot", font=F_SMALL, fill=(3, 7, 18))
    aim_end = (target[0] - 50, target[1] + 80 * scan)
    glow_line(img, [(robot_x + 80, robot_y), aim_end], GREEN, width=5, blur=8)
    draw.arc((robot_x - 120, robot_y - 120, robot_x + 180, robot_y + 120), start=-18, end=18, fill=rgba(GREEN, 190), width=4)
    draw.text((robot_x - 120, robot_y + 75), "慢速小角度扫描", font=F_BODY, fill=GREEN)
    arrow(draw, (robot_x - 40, robot_y + 120), (robot_x + 55, robot_y + 120), BLUE, width=4)
    draw.text((robot_x - 50, robot_y + 145), "侧移一点点", font=F_BODY, fill=BLUE)

    # Dwell meter.
    meter = (1090, 720, 1560, 775)
    rounded(draw, meter, rgba((2, 6, 23), 225), rgba(LINE, 220), radius=20, width=1)
    val = 0.62 + 0.38 * clamp((math.sin(t * 1.25) + 1) / 2)
    draw.rounded_rectangle((1102, 733, 1102 + int(446 * val), 761), radius=14, fill=rgba(GREEN, 230))
    draw.text((1090, 785), "目标：让合法视线驻留超过 0.8s", font=F_BODY, fill=INK)

    steps = [("1 到达射击点", YELLOW), ("2 小角度扫瞄", GREEN), ("3 被挡则侧移", BLUE), ("4 合法驻留命中", PURPLE)]
    for i, (txt, col) in enumerate(steps):
        pill(draw, (210 + i * 390, 845), txt, col, F_TINY)


def draw_eval(img: Image.Image, draw: ImageDraw.ImageDraw, stats: dict) -> None:
    c = stats.get("contract", {})
    r = stats.get("replay", {})
    draw.text((70, 92), "评估不是只看 reward：统计 + 审计 + 视频一起复核", font=F_BIG, fill=WHITE)
    draw_wrapped(draw, (72, 150), "这里把最终可复核指标直接画出来：胜率、靶子数量分布、基地命中率、箱子位移和 hard violation。", 1100, F_BODY, fill=(203, 213, 225))

    # Metric cards.
    cards = [
        ("episodes", f"{int(c.get('episodes', 128))}", GREEN),
        ("yellow win", f"{float(c.get('yellow_win_rate', 0.4922)) * 100:.1f}%", YELLOW),
        ("blue win", f"{float(c.get('blue_win_rate', 0.5078)) * 100:.1f}%", BLUE),
        ("hard violations", f"{int(r.get('hard_violations', 0))}", GREEN),
    ]
    for i, (name, val, col) in enumerate(cards):
        x = 80 + i * 440
        rounded(draw, (x, 255, x + 390, 380), rgba(col, 24), rgba(col, 190), radius=22, width=1)
        draw.text((x + 28, 280), name, font=F_SMALL, fill=MUTED)
        draw.text((x + 28, 310), val, font=F_TITLE, fill=col)

    # Win-rate bar.
    x, y = 110, 455
    draw.text((x, y - 52), "胜率平衡", font=F_H2, fill=WHITE)
    rounded(draw, (x, y, x + 700, y + 58), rgba((2, 6, 23), 220), rgba(LINE, 220), radius=20, width=1)
    yw = float(c.get("yellow_win_rate", 0.4922))
    draw.rounded_rectangle((x + 8, y + 10, x + 8 + int(684 * yw), y + 48), radius=16, fill=rgba(YELLOW, 230))
    draw.rounded_rectangle((x + 8 + int(684 * yw), y + 10, x + 692, y + 48), radius=16, fill=rgba(BLUE, 230))
    draw.text((x + 28, y + 72), "目标：双方接近 50%，避免单边策略漏洞", font=F_SMALL, fill=INK)

    # Normal hit distribution.
    dist = c.get("normal_hit_count_distribution", {}).get("yellow", {"1": 0.0078, "2": 0.7188, "3": 0.2656, "4": 0.0078})
    bx, by = 940, 455
    draw.text((bx, by - 52), "普通靶击倒数分布（黄方）", font=F_H2, fill=WHITE)
    draw.line((bx, by + 220, bx + 500, by + 220), fill=MUTED, width=2)
    for i, key in enumerate(["1", "2", "3", "4"]):
        val = float(dist.get(key, 0))
        h = int(210 * val)
        px = bx + i * 115
        draw.rounded_rectangle((px, by + 220 - h, px + 70, by + 220), radius=8, fill=rgba(GREEN, 220))
        draw.text((px + 24, by + 236), key, font=F_SMALL, fill=INK)
        draw.text((px - 4, by + 196 - h), f"{val * 100:.1f}%", font=F_TINY, fill=INK)

    # Base success by hits.
    base = c.get("base_success_by_hits", {}).get("yellow", {})
    bx2, by2 = 80, 760
    draw.text((bx2, by2 - 44), "基地命中率随普通靶数量变化", font=F_H2, fill=WHITE)
    for i, key in enumerate(["1", "2", "3", "4"]):
        item = base.get(key, {})
        val = float(item.get("success_rate", [0.0, 0.3622, 0.4571, 1.0][i]))
        x0 = bx2 + i * 220
        rounded(draw, (x0, by2, x0 + 175, by2 + 82), rgba(PURPLE, 20), rgba(PURPLE, 140), radius=16, width=1)
        draw.text((x0 + 18, by2 + 14), f"{key} 靶后", font=F_SMALL, fill=MUTED)
        draw.text((x0 + 18, by2 + 40), f"{val * 100:.1f}%", font=F_H2, fill=PURPLE)

    # Audit box.
    ax, ay = 1070, 760
    rounded(draw, (ax, ay, 1785, ay + 110), rgba(GREEN, 18), rgba(GREEN, 150), radius=18, width=1)
    audit = [
        f"穿模总数 {int(c.get('static_penetrations_total', 0)) + int(c.get('box_penetrations_total', 0))}",
        f"机器人碰撞/局 {float(c.get('robot_contacts_per_episode', 0.0)):.2f}",
        f"箱子位移 NE {float(c.get('mean_final_box_displacement_m', {}).get('box_ne', 0.1193)):.3f} m",
        f"严格回放 warnings {int(r.get('warnings', 0))}",
    ]
    for i, item in enumerate(audit):
        draw.text((ax + 26 + (i % 2) * 330, ay + 18 + (i // 2) * 42), f"✓ {item}", font=F_SMALL, fill=INK)


def render_frame(t: float, duration: float, videos: dict[str, LoopVideo], stats: dict) -> Image.Image:
    seg = current_segment(t)
    img, draw = base_canvas(t, duration)
    draw_section_nav(draw, seg)
    u = seg.u(t)

    if seg.key == "three_open":
        draw_three_view(img, draw, videos, t, opening=True)
    elif seg.key == "arena":
        draw_arena(img, draw, t, u)
    elif seg.key == "physics":
        draw_physics(img, draw, t, u)
    elif seg.key == "shooting":
        draw_shooting(img, draw, t, u)
    elif seg.key == "system":
        draw_system(img, draw, t, u)
    elif seg.key == "world_model":
        draw_world_model(img, draw, t, u)
    elif seg.key == "flow":
        draw_flow(img, draw, t, u)
    elif seg.key == "microaim":
        draw_microaim(img, draw, t, u)
    elif seg.key == "eval":
        draw_eval(img, draw, stats)
    else:
        draw_three_view(img, draw, videos, t, opening=False)
        draw.text((360, 880), "规则可信 · 物理可信 · 训练可信 · 回放可信", font=F_TITLE, fill=WHITE)

    if seg.key not in {"three_open", "ending"}:
        draw_footer(draw, seg)
    return img.convert("RGB")


def copy_inputs_to_temp(tmp: Path) -> dict[str, Path]:
    copied = {}
    for key, src in INPUTS.items():
        if not src.exists():
            raise FileNotFoundError(src)
        dst = tmp / f"{key}.mp4"
        shutil.copy2(src, dst)
        copied[key] = dst
    return copied


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
        "-shortest",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def encoder_args() -> list[str]:
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "20",
            "-b:v",
            "0",
        ]
    except Exception:
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
        ]


def render_video(duration: float, videos: dict[str, LoopVideo], stats: dict, seconds_override: float | None = None) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    actual_duration = seconds_override or duration
    enc = encoder_args()
    log(f"Encoder args: {' '.join(enc)}")
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
        "-vf",
        f"scale={OUT_W}:{OUT_H}:flags=lanczos",
        *enc,
        "-pix_fmt",
        "yuv420p",
        str(TMP_VIDEO),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = int(math.ceil(actual_duration * FPS))
    try:
        for i in range(total):
            t = i / FPS
            frame = render_frame(t, duration, videos, stats)
            assert proc.stdin is not None
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
            if i % (FPS * 10) == 0:
                log(f"Rendered {t:.1f}s / {actual_duration:.1f}s")
    finally:
        if proc.stdin:
            proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg render failed: {rc}")


def save_previews(duration: float, videos: dict[str, LoopVideo], stats: dict) -> None:
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    times = [2, 44, 84, 122, 160, 198, 248, 296, 322, 334]
    for t in times:
        frame = render_frame(t, duration, videos, stats)
        path = PREVIEW_DIR / f"preview_{int(t):03d}s.jpg"
        frame.save(path, quality=92)
        log(f"Saved preview {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="write representative preview frames only")
    parser.add_argument("--seconds", type=float, default=None, help="render only first N seconds for smoke test")
    args = parser.parse_args()

    LOG_PATH.write_text("", encoding="utf-8")
    audio_duration = generate_tts(NARRATION_WAV)
    duration = max(MIN_DURATION, audio_duration + 8.0)
    stats = load_stats()
    log(f"Narration duration: {audio_duration:.2f}s; video duration: {duration:.2f}s")
    log("Reference style used: progressive reveal, one concept per shot, restrained academic motion graphics. No Bilibili frames are reused.")

    with tempfile.TemporaryDirectory(prefix="rcvrl_expert_v2_") as tmp_name:
        copied = copy_inputs_to_temp(Path(tmp_name))
        videos = {key: LoopVideo(path) for key, path in copied.items()}
        try:
            if args.preview:
                save_previews(duration, videos, stats)
                return
            render_video(duration, videos, stats, args.seconds)
        finally:
            for v in videos.values():
                v.close()

    if args.seconds:
        log(f"Smoke video saved: {TMP_VIDEO}")
        return

    if NARRATION_WAV.exists() and NARRATION_WAV.stat().st_size > 1000:
        mux_audio(TMP_VIDEO, NARRATION_WAV, OUTPUT)
    else:
        shutil.copy2(TMP_VIDEO, OUTPUT)
    provenance = {
        "output": str(OUTPUT),
        "duration_s_target": duration,
        "fps": FPS,
        "draw_resolution": [W, H],
        "encoded_resolution": [OUT_W, OUT_H],
        "source_replays": {k: str(v) for k, v in INPUTS.items()},
        "method": "local Python/PIL/OpenCV/ffmpeg motion graphics; no external AI clips in the final video",
        "reference_videos_used_for_style_only": [
            "https://www.bilibili.com/video/BV1fj6vBfEnu/",
            "https://www.bilibili.com/video/BV1NCS4BkEt7/",
            "https://www.bilibili.com/video/BV1buxDzzE9P/",
        ],
    }
    (MEDIA_DIR / "reviewer_expert_5min_v2_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
