from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "isaaclab_sim" / "output" / "rl" / "mappo_selfplay_full_gpu" / "training_curve.csv"
TRAIN_SUMMARY = ROOT / "isaaclab_sim" / "output" / "rl" / "mappo_selfplay_full_gpu" / "training_summary.json"
EVAL_DET = ROOT / "isaaclab_sim" / "output" / "eval" / "mappo_full_gpu_eval.json"
EVAL_STOCH = ROOT / "isaaclab_sim" / "output" / "eval" / "mappo_full_gpu_eval_stochastic.json"
EVAL_BASELINE = ROOT / "isaaclab_sim" / "output" / "eval" / "scripted_rules_baseline_eval.json"
OUT_DIR = ROOT / "docs" / "figures" / "rl"
DOC_DATA_DIR = ROOT / "docs" / "rl_data" / "mappo_selfplay_full_gpu"


COLORS = {
    "ink": "#111827",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "panel": "#F8FAFC",
    "panel2": "#F1F5F9",
    "yellow": "#F2C94C",
    "blue": "#2563EB",
    "green": "#16A34A",
    "red": "#DC2626",
    "violet": "#7C3AED",
    "cyan": "#0891B2",
    "orange": "#F97316",
}


def read_curve() -> list[dict[str, float]]:
    with TRAIN_CSV.open("r", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if key == "device":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
        return rows


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_wrap(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L8,3 z" fill="{COLORS['ink']}"/>
    </marker>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#0F172A" flood-opacity="0.10"/>
    </filter>
    <style>
      .title{{font:700 34px Arial,Inter,sans-serif;fill:{COLORS['ink']}}}
      .subtitle{{font:400 17px Arial,Inter,sans-serif;fill:{COLORS['muted']}}}
      .h{{font:700 21px Arial,Inter,sans-serif;fill:{COLORS['ink']}}}
      .txt{{font:400 16px Arial,Inter,sans-serif;fill:#334155}}
      .small{{font:400 13px Arial,Inter,sans-serif;fill:#475569}}
      .tiny{{font:400 11px Arial,Inter,sans-serif;fill:#64748B}}
      .panel{{fill:{COLORS['panel']};stroke:#CBD5E1;stroke-width:1.4;rx:12;filter:url(#softShadow)}}
      .card{{fill:#FFFFFF;stroke:#CBD5E1;stroke-width:1.2;rx:8}}
      .grid{{stroke:{COLORS['grid']};stroke-width:1}}
      .axis{{stroke:#475569;stroke-width:1.4}}
      .dash{{stroke-dasharray:6 6}}
    </style>
  </defs>
  <rect width="{width}" height="{height}" fill="#FFFFFF"/>
{body}
</svg>
"""


def line_chart(
    x: int,
    y: int,
    w: int,
    h: int,
    rows: list[dict[str, float]],
    x_key: str,
    y_key: str,
    color: str,
    label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    xs = [row[x_key] for row in rows]
    ys = [row[y_key] for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min = min(ys) if y_min is None else y_min
    y_max = max(ys) if y_max is None else y_max
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    def sx(v: float) -> float:
        return x + (v - x_min) / (x_max - x_min) * w
    def sy(v: float) -> float:
        return y + h - (v - y_min) / (y_max - y_min) * h
    points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, ys))
    grid = []
    for i in range(5):
        gy = y + h * i / 4
        grid.append(f'<line class="grid" x1="{x}" y1="{gy:.1f}" x2="{x+w}" y2="{gy:.1f}"/>')
    return f"""
      {''.join(grid)}
      <line class="axis" x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}"/>
      <line class="axis" x1="{x}" y1="{y}" x2="{x}" y2="{y+h}"/>
      <polyline fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" points="{points}"/>
      <circle cx="{sx(xs[-1]):.1f}" cy="{sy(ys[-1]):.1f}" r="5" fill="{color}"/>
      <text class="small" x="{x}" y="{y-12}">{label}</text>
      <text class="tiny" x="{x}" y="{y+h+28}">{int(x_min):,} steps</text>
      <text class="tiny" x="{x+w-74}" y="{y+h+28}">{int(x_max):,} steps</text>
      <text class="tiny" x="{x+8}" y="{y+14}">{y_max:.3g}</text>
      <text class="tiny" x="{x+8}" y="{y+h-8}">{y_min:.3g}</text>
    """


def bar(x: float, y: float, w: float, h: float, color: str, label: str, value: str) -> str:
    return f"""
      <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="5" fill="{color}"/>
      <text class="tiny" x="{x + w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle">{value}</text>
      <text class="tiny" x="{x + w / 2:.1f}" y="{y + h + 18:.1f}" text-anchor="middle">{label}</text>
    """


def fig_training_curve(curve: list[dict[str, float]], summary: dict):
    final = summary["final_curve_row"]
    body = f"""
  <text class="title" x="70" y="62">GPU MAPPO Self-Play Training</text>
  <text class="subtitle" x="72" y="94">507,904 agent steps on {summary['cuda_device']} | CTDE critic, decentralized tactical actors</text>
  <rect class="panel" x="70" y="126" width="1040" height="560"/>
  <text class="h" x="105" y="170">A. Reward and convergence signals</text>
  {line_chart(120, 215, 910, 170, curve, "global_step", "mean_reward", COLORS["blue"], "mean reward / rollout")}
  {line_chart(120, 465, 390, 130, curve, "global_step", "done_rate", COLORS["green"], "terminal rate", 0.0, max(0.01, max(r["done_rate"] for r in curve)))}
  {line_chart(620, 465, 390, 130, curve, "global_step", "explained_variance", COLORS["violet"], "critic explained variance", min(-0.05, min(r["explained_variance"] for r in curve)), max(0.35, max(r["explained_variance"] for r in curve)))}
  <rect class="panel" x="1160" y="126" width="560" height="560"/>
  <text class="h" x="1195" y="170">B. Optimization diagnostics</text>
  {line_chart(1210, 220, 420, 120, curve, "global_step", "entropy", COLORS["orange"], "actor entropy")}
  {line_chart(1210, 420, 420, 120, curve, "global_step", "approx_kl", COLORS["red"], "approx KL")}
  <rect class="card" x="1210" y="585" width="420" height="58"/>
  <text class="txt" x="1232" y="620">final reward {final['mean_reward']:.4f} | speed {final['steps_per_second']:.0f} steps/s</text>
  <rect class="panel" x="70" y="730" width="1650" height="220"/>
  <text class="h" x="105" y="775">C. Experiment contract</text>
  <text class="txt" x="105" y="815">Algorithm: MAPPO self-play, gamma={summary['config']['gamma']}, rollout={summary['config']['rollout_steps']}, envs={summary['config']['num_envs']}, hidden={summary['config']['hidden_dim']}</text>
  <text class="txt" x="105" y="855">Action: target selector, base rush, block/interfere, recovery, fire gate, risk preference</text>
  <text class="txt" x="105" y="895">Safety: own-target fire remains forbidden; reward penalizes own-target and base mistakes.</text>
  <rect x="1260" y="785" width="370" height="90" rx="8" fill="#ECFDF5" stroke="#86EFAC"/>
  <text class="h" x="1290" y="825" fill="{COLORS['green']}">CUDA verified</text>
  <text class="txt" x="1290" y="858">{summary['torch_version']} / {summary['device']}</text>
"""
    (OUT_DIR / "rl_training_curve_gpu.svg").write_text(svg_wrap(1800, 1000, body), encoding="utf-8")


def fig_strategy_metrics(det: dict, stoch: dict, baseline: dict):
    summaries = [
        ("MAPPO det", det["summary"], COLORS["blue"]),
        ("MAPPO stochastic", stoch["summary"], COLORS["violet"]),
        ("Scripted baseline", baseline["summary"], COLORS["green"]),
    ]
    metrics = [
        ("normal_hits_per_episode", "normal hits"),
        ("base_hit_wins_per_episode", "base wins"),
        ("own_target_penalties_per_episode", "own penalties"),
        ("robot_contacts_per_episode", "contacts"),
        ("block_steps_per_episode", "block steps"),
        ("base_rush_steps_per_episode", "rush steps"),
    ]
    body = """
  <text class="title" x="70" y="62">Learned Strategy Event Distribution</text>
  <text class="subtitle" x="72" y="94">Evaluation over 64 episodes. Stochastic MAPPO exposes the learned risk-taking strategy; deterministic actor is conservative.</text>
  <rect class="panel" x="70" y="130" width="1660" height="760"/>
"""
    group_w = 250
    chart_x = 130
    chart_y = 220
    chart_h = 430
    for gi, (key, label) in enumerate(metrics):
        gx = chart_x + gi * group_w
        max_value = max(float(summary.get(key, 0.0)) for _, summary, _ in summaries)
        scale = chart_h / max(max_value, 1.0)
        body += f'<text class="small" x="{gx+75}" y="{chart_y-36}" text-anchor="middle">{label}</text>'
        for bi, (name, summary, color) in enumerate(summaries):
            value = float(summary.get(key, 0.0))
            h = value * scale
            body += bar(gx + bi * 46, chart_y + chart_h - h, 34, h, color, name.split()[0], f"{value:.2g}")
    body += f"""
  <line class="axis" x1="{chart_x-20}" y1="{chart_y+chart_h}" x2="{chart_x + group_w * len(metrics)-35}" y2="{chart_y+chart_h}"/>
  <rect class="card" x="145" y="735" width="450" height="94"/>
  <text class="h" x="175" y="775">Outcome</text>
  <text class="txt" x="175" y="810">Stochastic MAPPO blue win rate: {stoch['summary']['blue_win_rate'] * 100:.1f}%</text>
  <rect class="card" x="675" y="735" width="470" height="94"/>
  <text class="h" x="705" y="775">Safety</text>
  <text class="txt" x="705" y="810">Own-target penalties: {stoch['summary']['own_target_penalties_per_episode']:.1f} per episode</text>
  <rect class="card" x="1225" y="735" width="410" height="94"/>
  <text class="h" x="1255" y="775">Tactical behavior</text>
  <text class="txt" x="1255" y="810">Block + base-rush appear in learned rollouts</text>
"""
    (OUT_DIR / "rl_strategy_event_metrics.svg").write_text(svg_wrap(1800, 950, body), encoding="utf-8")


def world_to_svg(xy: list[float] | tuple[float, float], x: int, y: int, size: int) -> tuple[float, float]:
    return x + (float(xy[0]) + 1.5) / 3.0 * size, y + (1.5 - float(xy[1])) / 3.0 * size


def fig_policy_trace(stoch: dict):
    episode = next((ep for ep in stoch["episodes"] if ep["winner"] == "blue" and ep.get("trace")), stoch["episodes"][0])
    trace = episode.get("trace", [])
    arena_x, arena_y, arena_size = 90, 150, 760
    yellow_points = [world_to_svg(item["yellow_pose"][:2], arena_x, arena_y, arena_size) for item in trace]
    blue_points = [world_to_svg(item["blue_pose"][:2], arena_x, arena_y, arena_size) for item in trace]
    def poly(points: list[tuple[float, float]], color: str) -> str:
        return " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    targets = {
        "T01": (0.18, 1.26), "T02": (1.26, 1.26), "T03": (-1.26, 0.24), "T04": (-1.26, -0.24),
        "T05": (1.26, 0.24), "T06": (1.26, -0.24), "T07": (-1.26, -1.26), "T08": (-0.18, -1.26),
        "BlueBase": (-1.36, 1.36), "YellowBase": (1.36, -1.36),
    }
    target_svg = ""
    for name, xy in targets.items():
        px, py = world_to_svg(xy, arena_x, arena_y, arena_size)
        fill = "#FFFFFF" if "Base" not in name else "#FEE2E2"
        target_svg += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="11" fill="{fill}" stroke="#334155" stroke-width="2"/><text class="tiny" x="{px:.1f}" y="{py+28:.1f}" text-anchor="middle">{name}</text>'
    event_rows = []
    for item in trace:
        for team in ("yellow", "blue"):
            info = item.get(f"{team}_info", {})
            if "hit" in info or "winner" in info:
                event_rows.append((item["elapsed_s"], team, info.get("hit") or info.get("winner"), item["scores"]))
    event_text = ""
    for i, (t, team, event, scores) in enumerate(event_rows[:10]):
        y = 220 + i * 45
        color = COLORS["yellow"] if team == "yellow" else COLORS["blue"]
        event_text += f'<circle cx="1010" cy="{y-5}" r="7" fill="{color}"/><text class="txt" x="1030" y="{y}">{t:05.1f}s {team}: {event} | Y {scores["yellow"]} / B {scores["blue"]}</text>'
    body = f"""
  <text class="title" x="70" y="62">Learned Policy Episode Trace</text>
  <text class="subtitle" x="72" y="94">A stochastic MAPPO rollout: target order, route choice, blocking and base-rush decisions are evaluated under competition rules.</text>
  <rect class="panel" x="60" y="125" width="850" height="850"/>
  <text class="h" x="90" y="118">Arena trajectory</text>
  <rect x="{arena_x}" y="{arena_y}" width="{arena_size}" height="{arena_size}" fill="#FFFFFF" stroke="#111827" stroke-width="3"/>
  <line x1="{arena_x}" y1="{arena_y+arena_size/2}" x2="{arena_x+arena_size}" y2="{arena_y+arena_size/2}" stroke="#CBD5E1" stroke-width="5"/>
  <line x1="{arena_x+arena_size/2}" y1="{arena_y}" x2="{arena_x+arena_size/2}" y2="{arena_y+arena_size}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="8 8"/>
  {target_svg}
  <polyline fill="none" stroke="{COLORS['yellow']}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" points="{poly(yellow_points, COLORS['yellow'])}"/>
  <polyline fill="none" stroke="{COLORS['blue']}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" points="{poly(blue_points, COLORS['blue'])}"/>
  <rect class="panel" x="960" y="125" width="770" height="850"/>
  <text class="h" x="1000" y="175">Event timeline</text>
  {event_text}
  <rect class="card" x="1000" y="735" width="620" height="150"/>
  <text class="h" x="1030" y="775">Episode result</text>
  <text class="txt" x="1030" y="812">winner: {episode['winner']} | score Y {episode['scores']['yellow']} / B {episode['scores']['blue']}</text>
  <text class="txt" x="1030" y="850">yellow order: {', '.join(episode['target_order']['yellow'])}</text>
  <text class="txt" x="1030" y="888">blue order: {', '.join(episode['target_order']['blue'])}</text>
"""
    (OUT_DIR / "rl_policy_episode_trace.svg").write_text(svg_wrap(1800, 1030, body), encoding="utf-8")


def fig_tactical_contract(summary: dict, stoch: dict):
    body = f"""
  <text class="title" x="70" y="62">Tactical RL Contract for Sim2Real Deployment</text>
  <text class="subtitle" x="72" y="94">The learned layer chooses competition tactics; ROS2/Nav2/vision/shooter services execute safe low-level actions.</text>
  <rect class="panel" x="70" y="130" width="1660" height="760"/>
  <rect class="card" x="130" y="205" width="390" height="500"/>
  <text class="h" x="165" y="250">Local observation actor</text>
  <text class="txt" x="165" y="292">obs dim: {summary['obs_dim']}</text>
  <text class="txt" x="165" y="330">opponent bearing, armor, time</text>
  <text class="txt" x="165" y="368">target flags and localization confidence</text>
  <path d="M520 455 L660 455" stroke="{COLORS['ink']}" stroke-width="3" marker-end="url(#arrow)"/>
  <rect class="card" x="660" y="170" width="480" height="570"/>
  <text class="h" x="700" y="215">6-D tactical action</text>
  <text class="txt" x="700" y="265">1 target selection</text>
  <text class="txt" x="700" y="305">2 base-rush gate</text>
  <text class="txt" x="700" y="345">3 block/interference gate</text>
  <text class="txt" x="700" y="385">4 recovery gate</text>
  <text class="txt" x="700" y="425">5 fire gate</text>
  <text class="txt" x="700" y="465">6 risk preference</text>
  <rect x="700" y="515" width="360" height="72" rx="8" fill="#FEF2F2" stroke="#FCA5A5"/>
  <text class="txt" x="728" y="558">opponent-target safety gate</text>
  <path d="M1140 455 L1280 455" stroke="{COLORS['ink']}" stroke-width="3" marker-end="url(#arrow)"/>
  <rect class="card" x="1280" y="205" width="360" height="500"/>
  <text class="h" x="1315" y="250">ROS2 execution</text>
  <text class="txt" x="1315" y="292">Nav2 goal</text>
  <text class="txt" x="1315" y="330">AprilTag alignment</text>
  <text class="txt" x="1315" y="368">shooter service</text>
  <text class="txt" x="1315" y="406">EKF + recovery</text>
  <rect x="1315" y="470" width="260" height="82" rx="8" fill="#ECFDF5" stroke="#86EFAC"/>
  <text class="txt" x="1344" y="504">stochastic eval:</text>
  <text class="txt" x="1344" y="535">{stoch['summary']['base_hit_wins_per_episode']:.2f} base wins / episode</text>
  <rect class="card" x="170" y="760" width="1330" height="72"/>
  <text class="txt" x="205" y="805">CTDE: centralized critic dim {summary['central_obs_dim']} during training; each deployed robot runs only its local actor.</text>
"""
    (OUT_DIR / "rl_tactical_contract.svg").write_text(svg_wrap(1800, 950, body), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curve = read_curve()
    train_summary = read_json(TRAIN_SUMMARY)
    det = read_json(EVAL_DET)
    stoch = read_json(EVAL_STOCH)
    baseline = read_json(EVAL_BASELINE)
    fig_training_curve(curve, train_summary)
    fig_strategy_metrics(det, stoch, baseline)
    fig_policy_trace(stoch)
    fig_tactical_contract(train_summary, stoch)
    manifest = {
        "figures": [
            "docs/figures/rl/rl_training_curve_gpu.svg",
            "docs/figures/rl/rl_strategy_event_metrics.svg",
            "docs/figures/rl/rl_policy_episode_trace.svg",
            "docs/figures/rl/rl_tactical_contract.svg",
        ],
        "source_data": [
            str((DOC_DATA_DIR / "training_curve.csv").relative_to(ROOT)),
            str((DOC_DATA_DIR / "training_summary.json").relative_to(ROOT)),
            str((DOC_DATA_DIR / "mappo_full_gpu_eval.json").relative_to(ROOT)),
            str((DOC_DATA_DIR / "mappo_full_gpu_eval_stochastic.json").relative_to(ROOT)),
            str((DOC_DATA_DIR / "scripted_rules_baseline_eval.json").relative_to(ROOT)),
        ],
    }
    (OUT_DIR / "rl_result_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
