from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"

SOURCES = [
    "最终回放_顶视角",
    "最终回放_黄车第一视角",
    "最终回放_蓝车第一视角",
]


def make_gif(stem: str) -> None:
    src = MEDIA / f"{stem}.mp4"
    dst = MEDIA / f"{stem}.gif"
    if not src.exists():
        raise FileNotFoundError(src)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        palette = Path(tmp) / "palette.png"
        vf = "fps=4,scale=540:-1:flags=lanczos"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-vf",
                f"{vf},palettegen=stats_mode=diff",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-i",
                str(palette),
                "-filter_complex",
                f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
                "-loop",
                "0",
                str(dst),
            ],
            check=True,
        )
    print(f"wrote {dst.relative_to(ROOT)}")


def main() -> None:
    for stem in SOURCES:
        make_gif(stem)


if __name__ == "__main__":
    main()
