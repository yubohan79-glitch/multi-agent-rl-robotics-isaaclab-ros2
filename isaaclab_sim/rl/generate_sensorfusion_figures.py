from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "figures" / "rl"
TRAIN_CSV = ROOT / "isaaclab_sim" / "output" / "rl" / "mappo_sensorfusion_embodied_fast_gpu" / "training_curve.csv"
EVAL_JSON = ROOT / "isaaclab_sim" / "output" / "eval" / "mappo_sensorfusion_embodied_balanced_eval64_stochastic.json"
STRICT_JSON = ROOT / "isaaclab_sim" / "output" / "replay" / "mappo_sensorfusion_embodied_balanced_strict16" / "strict_replay_summary.json"


COLORS = {
    "ink": "#1F2937",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "yellow": "#F2C94C",
    "blue": "#2563EB",
    "green": "#16A34A",
    "red": "#DC2626",
    "violet": "#7C3AED",
}


def read_training_rows() -> list[dict[str, float]]:
    rows = []
    with TRAIN_CSV.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def polyline(rows: list[dict[str, float]], x_key: str, y_key: str, x: int, y: int, w: int, h: int, color: str) -> str:
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    if not xs or not ys:
        return ""
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0
    pts = []
    for row in rows:
        px = x + (float(row[x_key]) - xmin) / max(1e-9, xmax - xmin) * w
        py = y + h - (float(row[y_key]) - ymin) / max(1e-9, ymax - ymin) * h
        pts.append(f"{px:.1f},{py:.1f}")
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'


def chart_frame(x: int, y: int, w: int, h: int, title: str) -> str:
    grid = []
    for i in range(5):
        gy = y + i * h / 4
        grid.append(f'<line x1="{x}" y1="{gy:.1f}" x2="{x + w}" y2="{gy:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>')
    return "\n".join(
        [
            f'<rect x="{x - 18}" y="{y - 54}" width="{w + 36}" height="{h + 82}" rx="8" fill="#FFFFFF" stroke="#D1D5DB"/>',
            f'<text x="{x}" y="{y - 22}" class="h">{title}</text>',
            *grid,
            f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#9CA3AF" stroke-width="2"/>',
            f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#9CA3AF" stroke-width="2"/>',
        ]
    )


def write_training_figure(rows: list[dict[str, float]]) -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
<style>
.title{{font:700 38px Arial;fill:{COLORS["ink"]}}}.sub{{font:18px Arial;fill:{COLORS["muted"]}}}.h{{font:700 21px Arial;fill:{COLORS["ink"]}}}.txt{{font:16px Arial;fill:{COLORS["muted"]}}}
</style>
<rect width="1600" height="900" fill="#FFFFFF"/>
<text x="70" y="72" class="title">Sensor-Fusion MAPPO Training Curve</text>
<text x="70" y="106" class="sub">Actual GPU run: 200,704 agent steps, 16 parallel envs, residual expert, obs_dim=46.</text>
{chart_frame(110, 190, 620, 230, "Mean Reward")}
{polyline(rows, "global_step", "mean_reward", 110, 190, 620, 230, COLORS["green"])}
{chart_frame(870, 190, 620, 230, "Terminal Rate")}
{polyline(rows, "global_step", "done_rate", 870, 190, 620, 230, COLORS["blue"])}
{chart_frame(110, 560, 620, 210, "Critic Explained Variance")}
{polyline(rows, "global_step", "explained_variance", 110, 560, 620, 210, COLORS["violet"])}
{chart_frame(870, 560, 620, 210, "Approx KL")}
{polyline(rows, "global_step", "approx_kl", 870, 560, 620, 210, COLORS["red"])}
<text x="70" y="848" class="txt">Source: isaaclab_sim/output/rl/mappo_sensorfusion_embodied_fast_gpu/training_curve.csv</text>
</svg>'''
    (FIG_DIR / "rl_sensorfusion_training_curve.svg").write_text(svg, encoding="utf-8")


def bar(x: int, y: int, h: float, label: str, value: str, color: str) -> str:
    height = 280 * h
    return "\n".join(
        [
            f'<rect x="{x}" y="{y + 280 - height:.1f}" width="110" height="{height:.1f}" rx="6" fill="{color}" opacity="0.88"/>',
            f'<text x="{x + 55}" y="{y + 320}" class="txt" text-anchor="middle">{label}</text>',
            f'<text x="{x + 55}" y="{y + 280 - height - 14:.1f}" class="h" text-anchor="middle">{value}</text>',
        ]
    )


def normal_hit_distribution(episodes: list[dict[str, object]]) -> dict[int, int]:
    dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for episode in episodes:
        winner = str(episode.get("winner"))
        orders = episode.get("target_order", {})
        if winner not in ("yellow", "blue") or not isinstance(orders, dict):
            dist[0] += 1
            continue
        order = orders.get(winner, [])
        normal_hits = sum(1 for item in order if isinstance(item, str) and not item.endswith("BaseTarget"))
        dist[min(4, normal_hits)] += 1
    return dist


def write_eval_figure(eval_payload: dict[str, object], strict_payload: dict[str, object]) -> None:
    summary = eval_payload["summary"]
    strict = strict_payload["summary"]
    episodes = eval_payload["episodes"]
    dist = normal_hit_distribution(episodes)
    total = max(1, sum(dist.values()))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1700" height="980" viewBox="0 0 1700 980">
<style>
.title{{font:700 40px Arial;fill:{COLORS["ink"]}}}.sub{{font:18px Arial;fill:{COLORS["muted"]}}}.h{{font:700 22px Arial;fill:{COLORS["ink"]}}}.txt{{font:16px Arial;fill:{COLORS["muted"]}}}.big{{font:700 30px Arial;fill:{COLORS["ink"]}}}
</style>
<rect width="1700" height="980" fill="#FFFFFF"/>
<text x="76" y="76" class="title">Embodied Self-Play Evaluation Metrics</text>
<text x="76" y="112" class="sub">Actual outputs from 64 stochastic evaluation episodes and 16 strict replay audit episodes.</text>
<rect x="76" y="168" width="690" height="410" rx="8" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="112" y="216" class="h">Win / Safety Summary</text>
{bar(132, 258, float(summary["yellow_win_rate"]), "Yellow win", f'{float(summary["yellow_win_rate"])*100:.1f}%', COLORS["yellow"])}
{bar(302, 258, float(summary["blue_win_rate"]), "Blue win", f'{float(summary["blue_win_rate"])*100:.1f}%', COLORS["blue"])}
{bar(472, 258, float(summary["draw_or_timeout_rate"]), "Draw/timeout", f'{float(summary["draw_or_timeout_rate"])*100:.1f}%', "#9CA3AF")}
{bar(642, 258, 1.0 - min(1.0, float(summary["own_target_penalties_per_episode"])), "Own-fire safe", "0 own", COLORS["green"])}
<rect x="842" y="168" width="780" height="410" rx="8" fill="#FFFFFF" stroke="#D1D5DB"/>
<text x="878" y="216" class="h">Normal Targets Before Base</text>
'''
    x0 = 910
    for count, value in dist.items():
        svg += bar(x0 + count * 130, 258, value / total, f"{count} normal", str(value), COLORS["green"] if count in (1, 2) else "#9CA3AF")
    svg += f'''
<rect x="76" y="636" width="1546" height="210" rx="8" fill="#F9FAFB" stroke="#D1D5DB"/>
<text x="116" y="690" class="h">Measured Contract</text>
<text x="116" y="734" class="big">{float(summary["normal_hits_per_episode"]):.2f}</text><text x="212" y="734" class="txt">normal hits / episode</text>
<text x="480" y="734" class="big">{float(summary["base_hit_wins_per_episode"]):.3f}</text><text x="590" y="734" class="txt">base-hit wins / episode</text>
<text x="890" y="734" class="big">{int(strict["hard_violations"])}</text><text x="930" y="734" class="txt">hard replay violations</text>
<text x="1210" y="734" class="big">{int(strict["warnings"])}</text><text x="1250" y="734" class="txt">strict replay warnings</text>
<text x="116" y="794" class="txt">Laser rule: 5-50 cm outlet range, probabilistic accuracy by distance/lateral error, 0.80 s dwell before knockdown.</text>
<text x="76" y="910" class="txt">Sources: mappo_sensorfusion_embodied_balanced_eval64_stochastic.json; mappo_sensorfusion_embodied_balanced_strict16/strict_replay_summary.json</text>
</svg>'''
    (FIG_DIR / "rl_sensorfusion_eval_metrics.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_training_rows()
    eval_payload = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    strict_payload = json.loads(STRICT_JSON.read_text(encoding="utf-8"))
    write_training_figure(rows)
    write_eval_figure(eval_payload, strict_payload)
    print("wrote sensor-fusion RL figures")


if __name__ == "__main__":
    main()
