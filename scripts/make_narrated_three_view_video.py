from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "docs" / "media"
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")
FALLBACK_FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")

INPUTS = {
    "top": MEDIA_DIR / "最终回放_顶视角.mp4",
    "yellow": MEDIA_DIR / "最终回放_黄车第一视角.mp4",
    "blue": MEDIA_DIR / "最终回放_蓝车第一视角.mp4",
}
OUTPUT = MEDIA_DIR / "最终回放_三视角中文讲解.mp4"


@dataclass(frozen=True)
class Panel:
    x: int
    y: int
    w: int
    h: int
    label: str
    color: tuple[int, int, int]


W, H = 1600, 900
FPS_OUT = 24.0
TITLE_SECONDS = 3.0
ENDING_SECONDS = 3.0

BG = (248, 250, 252)
INK = (15, 23, 42)
MUTED = (71, 85, 105)
LINE = (203, 213, 225)
NAVY = (15, 23, 42)
GREEN = (22, 163, 74)
YELLOW = (230, 159, 0)
BLUE = (0, 114, 178)
ORANGE = (213, 94, 0)

TOP_PANEL = Panel(36, 112, 1000, 562, "主视角：顶视角 / 完整赛场轨迹", GREEN)
YELLOW_PANEL = Panel(1070, 112, 494, 278, "黄车第一视角", YELLOW)
BLUE_PANEL = Panel(1070, 420, 494, 278, "蓝车第一视角", BLUE)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_PATH if FONT_PATH.exists() else FALLBACK_FONT_PATH
    if not path.exists():
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size=size)


FONT_TITLE = font(34)
FONT_SUBTITLE = font(20)
FONT_LABEL = font(20)
FONT_SMALL = font(15)
FONT_BODY = font(25)
FONT_BODY_SMALL = font(20)


class SequentialVideoReader:
    def __init__(self, cap: cv2.VideoCapture, meta: tuple[int, float, int, int]):
        self.cap = cap
        self.frames, self.fps, _w, _h = meta
        self.index = -1
        self.last_frame: np.ndarray | None = None

    def frame_at(self, t: float) -> np.ndarray:
        target = min(max(0, int(t * self.fps)), max(0, self.frames - 1))
        while self.index < target:
            ok, frame = self.cap.read()
            if not ok:
                if self.last_frame is not None:
                    return self.last_frame
                raise RuntimeError(f"failed reading source frame {target}")
            self.index += 1
            self.last_frame = frame
        if self.last_frame is None:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("failed reading first source frame")
            self.index = 0
            self.last_frame = frame
        return self.last_frame


def copy_to_ascii_temp(tmp: Path) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for key, source in INPUTS.items():
        if not source.exists():
            raise FileNotFoundError(source)
        dest = tmp / f"{key}.mp4"
        shutil.copy2(source, dest)
        copied[key] = dest
    return copied


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    return cap


def video_meta(cap: cv2.VideoCapture) -> tuple[int, float, int, int]:
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or FPS_OUT
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frames, fps, width, height


def fit_frame(frame: np.ndarray, width: int, height: int) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    src_w, src_h = image.size
    scale = min(width / src_w, height / src_h)
    new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (226, 232, 240))
    ox = (width - new_size[0]) // 2
    oy = (height - new_size[1]) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def rounded_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill, outline=None, radius: int = 18, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    text_font: ImageFont.ImageFont,
    fill=INK,
    line_spacing: int = 8,
) -> None:
    x, y = xy
    for line in text.split("\n"):
        draw.text((x, y), line, font=text_font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=text_font)
        y += (bbox[3] - bbox[1]) + line_spacing


def draw_header(draw: ImageDraw.ImageDraw, subtitle: str) -> None:
    draw.rectangle((0, 0, W, 78), fill=NAVY)
    draw.text((36, 18), "RoboCupVisionRL 最终三视角回放", font=FONT_TITLE, fill=(255, 255, 255))
    draw.text((630, 31), subtitle, font=FONT_SUBTITLE, fill=(203, 213, 225))
    draw.rounded_rectangle((1338, 20, 1564, 60), radius=14, fill=(30, 41, 59), outline=(71, 85, 105), width=1)
    draw.text((1358, 30), "World-Model SAC Flow", font=FONT_SMALL, fill=(226, 232, 240))


def draw_panel(draw: ImageDraw.ImageDraw, canvas: Image.Image, panel: Panel, frame: np.ndarray) -> None:
    rounded_rect(draw, (panel.x - 4, panel.y - 34, panel.x + panel.w + 4, panel.y + panel.h + 4), (255, 255, 255), LINE, 14, 2)
    draw.rounded_rectangle((panel.x + 12, panel.y - 28, panel.x + panel.w - 12, panel.y + 8), radius=12, fill=(255, 255, 255), outline=panel.color, width=2)
    draw.text((panel.x + 28, panel.y - 24), panel.label, font=FONT_LABEL, fill=INK)
    image = fit_frame(frame, panel.w, panel.h)
    canvas.paste(image, (panel.x, panel.y))
    draw.rectangle((panel.x, panel.y, panel.x + panel.w, panel.y + panel.h), outline=panel.color, width=3)


def caption_for_time(t: float, total: float) -> tuple[str, tuple[int, int, int]]:
    ratio = t / max(total, 1e-6)
    if ratio < 0.18:
        return (
            "双车从各自起点同步出发。顶视角用于检查全局路线、箱子、挡板和靶子状态；两侧第一视角用于确认真实瞄准视线。",
            GREEN,
        )
    if ratio < 0.38:
        return (
            "策略只允许攻击对方靶。普通靶命中后按顺序移除对方基地挡板，未拆挡板前基地靶不会被合法击中。",
            BLUE,
        )
    if ratio < 0.58:
        return (
            "红色箱子是真实可推动刚体。推箱后地图位置持续更新，严格审计统计箱子位移、穿模次数和机器人接触事件。",
            ORANGE,
        )
    if ratio < 0.80:
        return (
            "到达射击点后，小车会做安全的小角度微扫与侧向微调，提高 0.80 秒激光驻留期间的合法命中概率。",
            YELLOW,
        )
    return (
        "最终基地命中来自完整比赛 trace。该回放通过严格规则审计：0 hard violation、0 own-target penalty、0 穿模。",
        GREEN,
    )


def draw_bottom_caption(draw: ImageDraw.ImageDraw, t: float, total: float) -> None:
    caption, color = caption_for_time(t, total)
    x0, y0, x1, y1 = 36, 722, 1564, 870
    rounded_rect(draw, (x0, y0, x1, y1), (255, 255, 255), LINE, 20, 2)
    draw.rounded_rectangle((x0 + 20, y0 + 22, x0 + 176, y0 + 58), radius=12, fill=(248, 250, 252), outline=color, width=2)
    draw.text((x0 + 38, y0 + 28), "中文讲解", font=FONT_BODY_SMALL, fill=INK)
    draw_text_box(draw, (x0 + 200, y0 + 18), caption, text_font=FONT_BODY, fill=INK, line_spacing=6)

    metrics = "128 局评估：黄方胜率 49.22% | 蓝方胜率 50.78% | draw 0.00% | 静态/箱子穿模 0"
    draw.text((x0 + 200, y0 + 88), metrics, font=FONT_BODY_SMALL, fill=MUTED)
    progress_w = int((x1 - x0 - 48) * min(1.0, max(0.0, t / max(total, 1e-6))))
    draw.rounded_rectangle((x0 + 24, y1 - 24, x1 - 24, y1 - 14), radius=5, fill=(226, 232, 240))
    draw.rounded_rectangle((x0 + 24, y1 - 24, x0 + 24 + progress_w, y1 - 14), radius=5, fill=color)


def title_card() -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, W, H), fill=BG)
    rounded_rect(draw, (90, 90, 1510, 760), (255, 255, 255), LINE, 28, 2)
    draw.text((150, 152), "RoboCupVisionRL 最终回放", font=font(52), fill=INK)
    draw.text((150, 230), "三视角同步剪辑 + 中文规则讲解", font=font(34), fill=MUTED)
    items = [
        ("顶视角", "检查全局路线、红色箱子、基地挡板和完整比赛 trace。", GREEN),
        ("黄车第一视角", "确认黄车瞄准、微调和合法对方靶攻击行为。", YELLOW),
        ("蓝车第一视角", "确认蓝车路线、基地攻坚窗口和视线遮挡逻辑。", BLUE),
        ("严格审计", "0 hard violation、0 own-target penalty、0 穿模。", ORANGE),
    ]
    y = 330
    for title, body, color in items:
        draw.rounded_rectangle((150, y, 360, y + 48), radius=15, fill=(248, 250, 252), outline=color, width=2)
        draw.text((174, y + 8), title, font=FONT_BODY_SMALL, fill=INK)
        draw.text((395, y + 8), body, font=FONT_BODY_SMALL, fill=MUTED)
        y += 72
    draw.text((150, 690), "方法口径：对象中心世界模型 + SAC Flow / PolicyFlow 自博弈", font=FONT_BODY, fill=INK)
    return canvas


def ending_card() -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    rounded_rect(draw, (90, 120, 1510, 760), (255, 255, 255), LINE, 28, 2)
    draw.text((150, 180), "回放结论", font=font(50), fill=INK)
    lines = [
        "1. 三个视角来自同一条严格 replay trace。",
        "2. 红色箱子、基地挡板、靶子状态和激光命中均按规则环境审计。",
        "3. 多 seed 评估胜率接近均衡，穿模与非法自靶惩罚为 0。",
        "4. 视频用于展示完整比赛行为，不替代 JSON/CSV 审计数据。",
    ]
    y = 295
    for line in lines:
        draw.text((155, y), line, font=FONT_BODY, fill=INK)
        y += 65
    draw.rounded_rectangle((150, 645, 1450, 705), radius=18, fill=(240, 253, 244), outline=GREEN, width=2)
    draw.text((185, 662), "最终文件：docs/media/最终回放_三视角中文讲解.mp4", font=FONT_BODY_SMALL, fill=INK)
    return canvas


def bgr_from_pil(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def write_still(writer: cv2.VideoWriter, image: Image.Image, seconds: float) -> None:
    frame = bgr_from_pil(image)
    for _ in range(int(round(seconds * FPS_OUT))):
        writer.write(frame)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rcvrl_three_view_") as temp_name:
        temp_dir = Path(temp_name)
        ascii_inputs = copy_to_ascii_temp(temp_dir)
        caps = {key: open_video(path) for key, path in ascii_inputs.items()}
        try:
            metas = {key: video_meta(cap) for key, cap in caps.items()}
            durations = {key: frames / max(fps, 1e-6) for key, (frames, fps, _w, _h) in metas.items()}
            total_seconds = min(durations.values())
            total_frames = int(math.floor(total_seconds * FPS_OUT))
            temp_output = temp_dir / "narrated_three_view.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_output), fourcc, FPS_OUT, (W, H))
            if not writer.isOpened():
                raise RuntimeError("cannot open mp4 writer")
            try:
                write_still(writer, title_card(), TITLE_SECONDS)
                readers = {key: SequentialVideoReader(caps[key], metas[key]) for key in caps}
                for out_index in range(total_frames):
                    t = out_index / FPS_OUT
                    frames = {key: reader.frame_at(t) for key, reader in readers.items()}

                    canvas = Image.new("RGB", (W, H), BG)
                    draw = ImageDraw.Draw(canvas)
                    draw_header(draw, f"同步时间 {t:05.1f}s / {total_seconds:05.1f}s")
                    draw_panel(draw, canvas, TOP_PANEL, frames["top"])
                    draw_panel(draw, canvas, YELLOW_PANEL, frames["yellow"])
                    draw_panel(draw, canvas, BLUE_PANEL, frames["blue"])
                    draw_bottom_caption(draw, t, total_seconds)
                    writer.write(bgr_from_pil(canvas))

                write_still(writer, ending_card(), ENDING_SECONDS)
            finally:
                writer.release()
        finally:
            for cap in caps.values():
                cap.release()

        shutil.copy2(temp_output, OUTPUT)
    print(f"[OK] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
