from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "docs" / "media"
FIG_DIR = ROOT / "docs" / "figures" / "paper"
DEEP_DIVE = ROOT / "docs" / "project_deep_dive.md"
LOG_PATH = MEDIA_DIR / "reviewer_4k_render.log"
OUTPUT = MEDIA_DIR / "最终回放_审稿人版_4K30_中文讲解.mp4"

INPUTS = {
    "top": MEDIA_DIR / "最终回放_顶视角.mp4",
    "yellow": MEDIA_DIR / "最终回放_黄车第一视角.mp4",
    "blue": MEDIA_DIR / "最终回放_蓝车第一视角.mp4",
}

FIGURES = {
    "overview": FIG_DIR / "fig01_project_overview.png",
    "method": FIG_DIR / "fig02_method_architecture.png",
    "training": FIG_DIR / "fig03_training_and_results.png",
    "safety": FIG_DIR / "fig04_ablation_and_safety.png",
    "pipeline": FIG_DIR / "fig05_sim2real_replay_pipeline.png",
}

W, H = 3840, 2160
FPS = 30
DURATION = 210.0

FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
FALLBACK_FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")

BG = (248, 250, 252)
PAPER = (255, 255, 255)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
LINE = (203, 213, 225)
NAVY = (15, 23, 42)
GREEN = (22, 163, 74)
YELLOW = (230, 159, 0)
BLUE = (0, 114, 178)
ORANGE = (213, 94, 0)
PURPLE = (124, 58, 237)
RED = (220, 38, 38)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    name: str

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end

    def u(self, t: float) -> float:
        return min(1.0, max(0.0, (t - self.start) / max(self.end - self.start, 1e-6)))


SEGMENTS = [
    Segment(0, 12, "开场"),
    Segment(12, 32, "项目总览"),
    Segment(32, 60, "规则与物理"),
    Segment(60, 86, "ROS2 与 Sim2Real"),
    Segment(86, 120, "对象中心世界模型"),
    Segment(120, 164, "三视角完整回放"),
    Segment(164, 184, "微调瞄准拆解"),
    Segment(184, 204, "评估与审计"),
    Segment(204, 210, "结论"),
]


def font(size: int) -> ImageFont.FreeTypeFont:
    path = FONT_PATH if FONT_PATH.exists() else FALLBACK_FONT_PATH
    if not path.exists():
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size=size)


F_TITLE = font(92)
F_H1 = font(68)
F_H2 = font(50)
F_BODY = font(42)
F_BODY2 = font(36)
F_SMALL = font(30)
F_TINY = font(24)


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * min(1.0, max(0.0, u))


def ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 3 * u * u - 2 * u * u * u


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, text_font: ImageFont.ImageFont) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        current = ""
        for char in para:
            candidate = current + char
            if draw.textlength(candidate, font=text_font) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
    return lines


def draw_multiline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    max_width: int,
    text_font: ImageFont.ImageFont,
    fill=INK,
    spacing: int = 12,
) -> int:
    x, y = xy
    for line in wrap_text(draw, text, max_width, text_font):
        draw.text((x, y), line, font=text_font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=text_font)
        y += (bbox[3] - bbox[1]) + spacing
    return y


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=LINE, radius=28, width=3) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color=INK, width=5) -> None:
    draw.line((a[0], a[1], b[0], b[1]), fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 22
    p1 = (b[0] - size * math.cos(ang - 0.45), b[1] - size * math.sin(ang - 0.45))
    p2 = (b[0] - size * math.cos(ang + 0.45), b[1] - size * math.sin(ang + 0.45))
    draw.polygon([b, p1, p2], fill=color)


def fit_image(image: Image.Image, size: tuple[int, int], bg=(241, 245, 249)) -> Image.Image:
    image = image.convert("RGB")
    sw, sh = image.size
    tw, th = size
    scale = min(tw / sw, th / sh)
    new = (max(1, int(sw * scale)), max(1, int(sh * scale)))
    resized = image.resize(new, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    canvas.paste(resized, ((tw - new[0]) // 2, (th - new[1]) // 2))
    return canvas


def video_frame_to_image(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return fit_image(Image.fromarray(rgb), size, bg=(226, 232, 240))


class LoopVideo:
    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open source video: {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.index = -1
        self.last: np.ndarray | None = None

    def read_at(self, t: float) -> np.ndarray:
        if self.frames <= 0:
            raise RuntimeError("empty video")
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
                raise RuntimeError("failed reading first frame")
            self.index = 0
            self.last = frame
        return self.last

    def close(self) -> None:
        self.cap.release()


def copy_inputs_to_ascii(tmp: Path) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for key, src in INPUTS.items():
        if not src.exists():
            raise FileNotFoundError(src)
        dst = tmp / f"{key}.mp4"
        shutil.copy2(src, dst)
        copied[key] = dst
    return copied


def load_figures() -> dict[str, Image.Image]:
    images = {}
    for key, path in FIGURES.items():
        images[key] = Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", (1920, 1080), BG)
    return images


def base_frame(t: float, title: str, section: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 176), fill=NAVY)
    draw.text((92, 42), title, font=F_H1, fill=(255, 255, 255))
    draw.text((92, 118), section, font=F_SMALL, fill=(203, 213, 225))
    draw.rounded_rectangle((3150, 52, 3728, 124), radius=22, fill=(30, 41, 59), outline=(71, 85, 105), width=2)
    draw.text((3190, 70), "4K30 Reviewer Cut", font=F_SMALL, fill=(226, 232, 240))
    progress = int((W - 184) * min(1.0, max(0.0, t / DURATION)))
    draw.rounded_rectangle((92, 2020, W - 92, 2040), radius=10, fill=(226, 232, 240))
    draw.rounded_rectangle((92, 2020, 92 + progress, 2040), radius=10, fill=GREEN)
    draw.text((92, 2064), f"{t:05.1f}s / {DURATION:05.1f}s", font=F_TINY, fill=MUTED)
    return img, draw


def bullet_card(draw: ImageDraw.ImageDraw, xy: tuple[int, int], title: str, bullets: list[str], color, width=820, height=430) -> None:
    x, y = xy
    rounded(draw, (x, y, x + width, y + height), PAPER, color, 28, 4)
    draw.text((x + 42, y + 34), title, font=F_H2, fill=INK)
    yy = y + 112
    for item in bullets:
        draw.ellipse((x + 46, yy + 14, x + 66, yy + 34), fill=color)
        yy = draw_multiline(draw, (x + 88, yy), item, max_width=width - 130, text_font=F_BODY2, fill=INK, spacing=10)
        yy += 20


def section_at(t: float) -> Segment:
    for seg in SEGMENTS:
        if seg.contains(t):
            return seg
    return SEGMENTS[-1]


def draw_title(t: float, docs_text: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    u = ease(t / 12.0)
    draw.rectangle((0, 0, W, H), fill=BG)
    rounded(draw, (220, 210, 3620, 1780), PAPER, LINE, 46, 4)
    draw.text((330, 360), "RoboCupVisionRL", font=font(130), fill=INK)
    draw.text((330, 520), "对象中心世界模型 + SAC Flow 多智能体机器人", font=F_H1, fill=MUTED)
    draw.line((330, 660, int(330 + 1200 * u), 660), fill=GREEN, width=14)
    cards = [
        ("规则可信", "靶子、挡板、推箱、激光驻留全部可审计", GREEN),
        ("系统可信", "ROS2/Nav2/IsaacLab 共用同一规则口径", BLUE),
        ("训练可信", "128 局多 seed 评估 + 严格 replay", ORANGE),
        ("展示可信", "三视角完整回放 + 可复现图表", PURPLE),
    ]
    for i, (title, body, color) in enumerate(cards):
        x = 330 + i * 790
        y = 840 + int(50 * (1 - u))
        rounded(draw, (x, y, x + 700, y + 320), (248, 250, 252), color, 28, 4)
        draw.text((x + 44, y + 44), title, font=F_H2, fill=INK)
        draw_multiline(draw, (x + 44, y + 130), body, max_width=600, text_font=F_BODY2, fill=MUTED)
    quote = "讲解素材来自 docs/project_deep_dive.md：规则、ROS2、IsaacLab、世界模型训练、评估审计与 Sim2Real 部署。"
    draw_multiline(draw, (330, 1320), quote, max_width=2960, text_font=F_BODY, fill=INK)
    draw.text((330, 1590), "输出规格：3840 x 2160 / 30 fps / 时长超过 3 分钟", font=F_BODY2, fill=MUTED)
    return img


def draw_project_overview(t: float, fig: Image.Image) -> Image.Image:
    img, draw = base_frame(t, "1. 项目总览", "从比赛规则到可复现实验的端到端机器人学习系统")
    u = SEGMENTS[1].u(t)
    fig_img = fit_image(fig, (1720, 968))
    rounded(draw, (92, 260, 1872, 1296), PAPER, LINE, 32, 3)
    img.paste(fig_img, (122, 294))
    bullet_card(
        draw,
        (1990, 280),
        "项目覆盖范围",
        [
            "两车多智能体对抗策略学习：靶子选择、推箱路线、基地攻坚和早攻窗口。",
            "ROS2/Nav2/视觉/发射器形成真实机器人闭环，IsaacLab 负责物理回放与展示。",
            "正式结果必须来自完整比赛 rollout，不使用短片段冒充成功。",
        ],
        GREEN,
        width=1760,
        height=620,
    )
    bullet_card(
        draw,
        (1990, 980),
        "可信结果原则",
        [
            "不只看 reward，同时检查胜率、碰撞、穿模、普通靶数量、基地命中率和视频行为。",
            "README、论文图、GIF、PPTX 和 JSON/CSV 指标均从同一套结果数据生成。",
        ],
        BLUE,
        width=1760,
        height=470,
    )
    draw.text((2100, 1548), f"当前段落进度 {int(u * 100):02d}%", font=F_SMALL, fill=MUTED)
    return img


def draw_arena_scene(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, u: float) -> None:
    rounded(draw, (x, y, x + s, y + s), (255, 255, 255), LINE, 32, 4)
    pad = 80
    ax0, ay0 = x + pad, y + pad
    ax1, ay1 = x + s - pad, y + s - pad
    draw.rectangle((ax0, ay0, ax1, ay1), fill=(241, 245, 249), outline=INK, width=5)
    draw.line((ax0 + (ax1 - ax0) // 2, ay0, ax0 + (ax1 - ax0) // 2, ay1), fill=(203, 213, 225), width=3)
    draw.line((ax0, ay0 + (ay1 - ay0) // 2, ax1, ay0 + (ay1 - ay0) // 2), fill=(203, 213, 225), width=3)

    def map_xy(px: float, py: float) -> tuple[int, int]:
        mx = ax0 + int((px + 1.5) / 3.0 * (ax1 - ax0))
        my = ay1 - int((py + 1.5) / 3.0 * (ay1 - ay0))
        return mx, my

    # Bases and starts.
    for px, py, color, label in [(1.25, -1.25, YELLOW, "黄方基地"), (-1.25, 1.25, BLUE, "蓝方基地")]:
        cx, cy = map_xy(px, py)
        draw.rounded_rectangle((cx - 92, cy - 92, cx + 92, cy + 92), radius=18, fill=(255, 255, 255), outline=color, width=5)
        draw.text((cx - 72, cy + 104), label, font=F_TINY, fill=INK)

    boxes = [(0.8, 0.8), (-0.8, -0.8)]
    for bx, by in boxes:
        cx, cy = map_xy(bx, by)
        offset = int(20 * math.sin(u * math.pi * 2))
        draw.rectangle((cx - 56 + offset, cy - 56, cx + 56 + offset, cy + 56), fill=(248, 113, 113), outline=RED, width=4)

    targets = [
        (0.18, 1.26, BLUE), (1.26, 1.26, BLUE), (-1.26, 0.24, BLUE), (1.26, 0.24, BLUE),
        (-1.26, -0.24, YELLOW), (1.26, -0.24, YELLOW), (-1.26, -1.26, YELLOW), (-0.18, -1.26, YELLOW),
    ]
    for i, (tx, ty, color) in enumerate(targets):
        cx, cy = map_xy(tx, ty)
        r = 18 + int(5 * math.sin(u * 8 + i))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=INK, width=2)

    # Robot trajectories.
    yellow_path = [(0.25, -1.25), (0.55, -0.55), (0.15, 0.75), (-0.72, 1.02), (-1.05, 1.05)]
    blue_path = [(-0.25, 1.25), (-0.52, 0.50), (-0.05, -0.75), (0.72, -1.02), (1.05, -1.05)]
    for path, color in [(yellow_path, YELLOW), (blue_path, BLUE)]:
        pts = [map_xy(*p) for p in path]
        draw.line(pts, fill=color, width=9, joint="curve")
        idx = min(len(pts) - 1, int(u * (len(pts) - 1)))
        cx, cy = pts[idx]
        draw.rounded_rectangle((cx - 34, cy - 24, cx + 34, cy + 24), radius=12, fill=color, outline=INK, width=3)


def draw_rules(t: float) -> Image.Image:
    img, draw = base_frame(t, "2. 规则与物理拆解", "靶子、箱子、基地挡板、激光驻留必须真实可信")
    u = SEGMENTS[2].u(t)
    draw_arena_scene(draw, 110, 270, 1320, u)
    bullet_card(
        draw,
        (1530, 280),
        "比赛规则合同",
        [
            "黄车只能攻击蓝方靶，蓝车只能攻击黄方靶；自己靶和自己基地在发射前被安全门拒绝。",
            "普通靶命中后移除对方一块基地装甲；基地靶必须在合法窗口、合法距离、合法视线下命中。",
            "激光驻留不足 0.80 秒时命中概率为 0，0.80 到 2.00 秒之间概率逐步增强。",
        ],
        BLUE,
        width=2160,
        height=620,
    )
    bullet_card(
        draw,
        (1530, 980),
        "物理可信要求",
        [
            "红色箱子是真实可推动物体，推后位置要持续变化，不能只是视觉位移。",
            "蓝色基地挡板要真实阻挡车辆和激光，未拆挡板前基地靶不可被击中。",
            "严格审计检查穿墙、穿箱、穿挡板、非法自靶、异常差速步长和规则事件一致性。",
        ],
        ORANGE,
        width=2160,
        height=620,
    )
    return img


def draw_ros2(t: float, fig: Image.Image) -> Image.Image:
    img, draw = base_frame(t, "3. ROS2 与 Sim2Real 闭环", "训练策略不绕过真实机器人接口，部署层仍是 ROS2 合同")
    u = SEGMENTS[3].u(t)
    fig_img = fit_image(fig, (1620, 912))
    rounded(draw, (92, 286, 1780, 1268), PAPER, LINE, 28, 3)
    img.paste(fig_img, (126, 320))

    nodes = [
        ("AprilTag 视觉", 2050, 350, BLUE),
        ("EKF 融合定位", 2750, 350, GREEN),
        ("Nav2 路径控制", 2050, 700, ORANGE),
        ("行为状态机", 2750, 700, PURPLE),
        ("发射器服务", 2400, 1050, RED),
    ]
    for name, x, y, color in nodes:
        rounded(draw, (x, y, x + 540, y + 170), (255, 255, 255), color, 26, 4)
        draw.text((x + 46, y + 54), name, font=F_BODY, fill=INK)
    arrows = [
        ((2590, 435), (2750, 435)),
        ((2320, 520), (2320, 700)),
        ((3020, 520), (3020, 700)),
        ((2590, 785), (2750, 785)),
        ((3020, 870), (2670, 1050)),
    ]
    for a, b in arrows:
        arrow(draw, a, b, color=INK, width=6)
    draw_multiline(
        draw,
        (2010, 1360),
        "Sim2Real 迁移层不是仿真全状态，而是 /cmd_vel、/target_detection、Nav2 goal、TF、EKF 和 shooter service。策略只输出高层战术，安全门仍由规则层执行。",
        max_width=1620,
        text_font=F_BODY,
        fill=INK,
    )
    draw.text((2010, 1675), f"动画强调：接口一致性 > 单次仿真成功  ({int(u * 100)}%)", font=F_SMALL, fill=MUTED)
    return img


def draw_algorithm(t: float, fig: Image.Image) -> Image.Image:
    img, draw = base_frame(t, "4. 对象中心世界模型 + SAC Flow", "让策略显式看到机器人、靶子、箱子、挡板和规则状态")
    u = SEGMENTS[4].u(t)
    left = fit_image(fig, (1580, 889))
    rounded(draw, (92, 285, 1740, 1244), PAPER, LINE, 28, 3)
    img.paste(left, (126, 320))

    x0, y0 = 1880, 300
    stages = [
        ("对象 token", "机器人 / 靶子 / 箱子 / 挡板 / 分数 / 时间", GREEN),
        ("Flow Actor", "6 维高层战术动作，表达多路线和早攻窗口", PURPLE),
        ("Twin-Q Critic", "训练时使用对象中心全局状态降低过估计", BLUE),
        ("World Model", "学习对象动态、reward 和 done，为想象 rollout 留接口", ORANGE),
        ("Rule Shield", "对手靶、视线、距离、驻留、装甲门统一约束", RED),
    ]
    for i, (name, body, color) in enumerate(stages):
        y = y0 + i * 260
        rounded(draw, (x0, y, x0 + 1580, y + 180), (255, 255, 255), color, 26, 4)
        draw.text((x0 + 48, y + 26), name, font=F_H2, fill=INK)
        draw_multiline(draw, (x0 + 410, y + 36), body, max_width=1060, text_font=F_BODY2, fill=MUTED)
        if i < len(stages) - 1:
            arrow(draw, (x0 + 790, y + 180), (x0 + 790, y + 250), color=INK, width=5)
    pulse_x = int(1880 + 1580 * (0.08 + 0.84 * ((u * 2) % 1.0)))
    draw.ellipse((pulse_x - 22, 1616, pulse_x + 22, 1660), fill=GREEN)
    draw.text((1880, 1710), "动作含义：target_selector / base_rush_gate / block_interference_gate / recovery_gate / fire_gate / risk_preference", font=F_SMALL, fill=MUTED)
    return img


def draw_replay(t: float, videos: dict[str, LoopVideo]) -> Image.Image:
    img, draw = base_frame(t, "5. 三视角同步回放", "顶视角展示完整比赛，第一视角检查真实瞄准与遮挡")
    u = SEGMENTS[5].u(t)
    replay_t = (t - SEGMENTS[5].start)
    top = video_frame_to_image(videos["top"].read_at(replay_t), (2410, 1356))
    yellow = video_frame_to_image(videos["yellow"].read_at(replay_t), (1180, 664))
    blue = video_frame_to_image(videos["blue"].read_at(replay_t), (1180, 664))

    rounded(draw, (78, 258, 2562, 1688), PAPER, GREEN, 30, 5)
    img.paste(top, (116, 306))
    draw.text((130, 264), "顶视角：完整赛场轨迹 / 推箱 / 基地挡板 / 靶子状态", font=F_BODY2, fill=INK)

    rounded(draw, (2660, 258, 3780, 950), PAPER, YELLOW, 28, 5)
    img.paste(yellow, (2630, 306))
    draw.text((2678, 264), "黄车第一视角：射击点、微扫、合法靶", font=F_BODY2, fill=INK)

    rounded(draw, (2660, 1000, 3780, 1692), PAPER, BLUE, 28, 5)
    img.paste(blue, (2630, 1048))
    draw.text((2678, 1006), "蓝车第一视角：路线、基地窗口、视线遮挡", font=F_BODY2, fill=INK)

    captions = [
        "两车从出发区同步出发，策略只允许攻击对方靶。",
        "普通靶命中后移除基地装甲，未拆挡板时基地靶不会被合法命中。",
        "红色箱子被推动后，位置进入对象状态并在严格回放中持续更新。",
        "到达点位后执行小角度微扫，增加 0.80 秒驻留期间的合法命中概率。",
        "最终胜负来自完整比赛 trace，不是截取短片段。",
    ]
    idx = min(len(captions) - 1, int(u * len(captions)))
    rounded(draw, (92, 1765, 3748, 1950), PAPER, LINE, 26, 3)
    draw_multiline(draw, (136, 1798), captions[idx], max_width=3400, text_font=F_BODY, fill=INK)
    return img


def draw_micro_aim(t: float) -> Image.Image:
    img, draw = base_frame(t, "6. 微调瞄准动画拆解", "解决到达基地附近却差一点打不中的问题")
    u = SEGMENTS[6].u(t)
    cx, cy = 1160, 1020
    target = (2480, 1020)
    rounded(draw, (360, 360, 3320, 1600), PAPER, LINE, 36, 4)
    draw.text((520, 450), "合法开火需要同时满足：距离、视线、角度、驻留、装甲窗口", font=F_H2, fill=INK)
    # Base blocker and target.
    draw.rectangle((2350, 760, 2620, 1280), fill=(191, 219, 254), outline=BLUE, width=6)
    draw.rectangle((2600, 890, 2680, 1150), fill=(255, 255, 255), outline=RED, width=6)
    draw.text((2320, 1320), "基地挡板", font=F_SMALL, fill=MUTED)
    draw.text((2580, 1195), "基地靶", font=F_SMALL, fill=MUTED)
    # Robot and scan rays.
    scan = math.sin(u * math.pi * 8) * 0.16
    rx = int(cx + 260 * ease(u))
    ry = int(cy + 70 * math.sin(u * math.pi * 2))
    yaw = math.atan2(target[1] - ry, target[0] - rx) + scan
    draw.rounded_rectangle((rx - 110, ry - 70, rx + 110, ry + 70), radius=24, fill=(254, 243, 199), outline=YELLOW, width=6)
    draw.text((rx - 68, ry - 24), "机器人", font=F_SMALL, fill=INK)
    for k, alpha in enumerate([-0.12, 0.0, 0.12]):
        ang = yaw + alpha
        end = (int(rx + 1250 * math.cos(ang)), int(ry + 1250 * math.sin(ang)))
        color = GREEN if k == 1 else (251, 191, 36)
        draw.line((rx, ry, end[0], end[1]), fill=color, width=5)
    draw.arc((rx - 180, ry - 180, rx + 180, ry + 180), start=-22, end=22, fill=ORANGE, width=8)
    bullet_card(
        draw,
        (550, 1370),
        "为什么需要微扫",
        [
            "小车到达射击点后，若只静止等待，轻微角度误差会导致驻留期间命中不稳定。",
            "策略在安全余量内慢速小角度扫描，并尝试厘米级侧向/径向候选点。",
        ],
        ORANGE,
        width=2660,
        height=420,
    )
    return img


def draw_metrics(t: float, training_fig: Image.Image, safety_fig: Image.Image) -> Image.Image:
    img, draw = base_frame(t, "7. 多 seed 评估与严格审计", "正式结论绑定 JSON/CSV 指标和 replay audit，不只看 reward")
    u = SEGMENTS[7].u(t)
    rounded(draw, (92, 276, 1780, 1228), PAPER, LINE, 28, 3)
    img.paste(fit_image(training_fig, (1600, 900)), (136, 306))
    rounded(draw, (1940, 276, 3748, 1228), PAPER, LINE, 28, 3)
    img.paste(fit_image(safety_fig, (1710, 962)), (1988, 302))

    cards = [
        ("128 局", "多 seed 随机评估", GREEN),
        ("49.22%", "黄方胜率", YELLOW),
        ("50.78%", "蓝方胜率", BLUE),
        ("0", "静态/箱子穿模", RED),
        ("0", "严格回放 hard violation", ORANGE),
    ]
    for i, (num, label, color) in enumerate(cards):
        x = 120 + i * 735
        y = 1370
        rounded(draw, (x, y, x + 620, y + 260), PAPER, color, 28, 5)
        draw.text((x + 44, y + 40), num, font=font(72), fill=color)
        draw_multiline(draw, (x + 44, y + 145), label, max_width=520, text_font=F_BODY2, fill=INK)
    draw_multiline(
        draw,
        (132, 1745),
        "审稿人应关注：胜率是否平衡、基地命中率是否随普通靶数量合理变化、推箱是否真实产生位移、回放是否没有穿模和非法自靶事件。",
        max_width=3500,
        text_font=F_BODY,
        fill=INK,
    )
    # Animated underline.
    draw.line((132, 1910, int(132 + 3400 * ease(u)), 1910), fill=GREEN, width=12)
    return img


def draw_conclusion(t: float, pipeline_fig: Image.Image) -> Image.Image:
    img, draw = base_frame(t, "8. 结论与交付物", "代码、训练、评估、图表、回放共同构成可审计证据链")
    rounded(draw, (120, 260, 1820, 1220), PAPER, LINE, 28, 3)
    img.paste(fit_image(pipeline_fig, (1600, 900)), (168, 294))
    bullet_card(
        draw,
        (1990, 320),
        "最终交付",
        [
            "三视角完整回放：顶视角、黄车第一视角、蓝车第一视角。",
            "算法口径：对象中心世界模型 + SAC Flow/PolicyFlow 自博弈。",
            "结果数据：训练曲线、128 局评估、严格 replay、规则几何审计。",
            "复现入口：训练、评估、导出、回放、图表生成脚本。",
        ],
        GREEN,
        width=1760,
        height=820,
    )
    rounded(draw, (360, 1450, 3480, 1745), (240, 253, 244), GREEN, 32, 4)
    draw.text((440, 1508), "本视频用途", font=F_H2, fill=INK)
    draw_multiline(
        draw,
        (850, 1518),
        "面向审稿人快速理解：项目不是单一策略视频，而是一个规则可信、训练可信、回放可信、可复现的机器人学习系统。",
        max_width=2500,
        text_font=F_BODY,
        fill=INK,
    )
    return img


def render_frame(t: float, figs: dict[str, Image.Image], videos: dict[str, LoopVideo], docs_text: str) -> Image.Image:
    seg = section_at(t)
    if seg.name == "开场":
        return draw_title(t, docs_text)
    if seg.name == "项目总览":
        return draw_project_overview(t, figs["overview"])
    if seg.name == "规则与物理":
        return draw_rules(t)
    if seg.name == "ROS2 与 Sim2Real":
        return draw_ros2(t, figs["pipeline"])
    if seg.name == "对象中心世界模型":
        return draw_algorithm(t, figs["method"])
    if seg.name == "三视角完整回放":
        return draw_replay(t, videos)
    if seg.name == "微调瞄准拆解":
        return draw_micro_aim(t)
    if seg.name == "评估与审计":
        return draw_metrics(t, figs["training"], figs["safety"])
    return draw_conclusion(t, figs["pipeline"])


def open_ffmpeg() -> subprocess.Popen:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
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
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUTPUT),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    log(f"[START] rendering {OUTPUT}")
    docs_text = DEEP_DIVE.read_text(encoding="utf-8") if DEEP_DIVE.exists() else ""
    figs = load_figures()
    total_frames = int(DURATION * FPS)

    with tempfile.TemporaryDirectory(prefix="rcvrl_reviewer4k_") as temp_name:
        temp = Path(temp_name)
        copied = copy_inputs_to_ascii(temp)
        videos = {key: LoopVideo(path) for key, path in copied.items()}
        proc = open_ffmpeg()
        assert proc.stdin is not None
        try:
            for idx in range(total_frames):
                t = idx / FPS
                frame = render_frame(t, figs, videos, docs_text)
                proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
                if idx % (FPS * 5) == 0:
                    log(f"[PROGRESS] {idx}/{total_frames} frames, t={t:.1f}s")
            proc.stdin.close()
            return_code = proc.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed with code {return_code}")
        finally:
            for video in videos.values():
                video.close()
            if proc.poll() is None:
                proc.kill()
    log(f"[DONE] wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    print(f"[OK] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
