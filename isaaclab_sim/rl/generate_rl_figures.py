from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "figures" / "rl"
DOC_DATA_DIR = ROOT / "docs" / "rl_data" / "world_model_sacflow_final"

TRAIN_CSV_CANDIDATES = [
    DOC_DATA_DIR / "training_curve.csv",
    ROOT / "isaaclab_sim" / "output" / "rl" / "world_model_sacflow_seed260707_rerun" / "training_curve.csv",
]
TRAIN_SUMMARY_CANDIDATES = [
    DOC_DATA_DIR / "training_summary.json",
    ROOT / "isaaclab_sim" / "output" / "rl" / "world_model_sacflow_seed260707_rerun" / "training_summary.json",
]
EVAL_CANDIDATES = [
    DOC_DATA_DIR / "contract_eval_multiseed.json",
    ROOT / "isaaclab_sim" / "output" / "eval" / "world_model_sacflow_microaim_contract_eval256.json",
    ROOT / "isaaclab_sim" / "output" / "eval" / "world_model_sacflow_rs004_multiseed_contract_eval128.json",
]
STRICT_CANDIDATES = [
    DOC_DATA_DIR / "strict_replay_summary.json",
    ROOT / "isaaclab_sim" / "output" / "replay" / "world_model_sacflow_strict_replay_abs" / "strict_replay_summary.json",
]


COLORS = {
    "ink": "#111827",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "panel": "#F8FAFC",
    "yellow": "#E5B82E",
    "blue": "#2563EB",
    "green": "#16A34A",
    "red": "#DC2626",
    "violet": "#7C3AED",
    "cyan": "#0891B2",
    "orange": "#F97316",
}


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of these inputs exist: " + ", ".join(str(path) for path in paths))


def read_curve(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items() if value not in ("", None)})
    return rows


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_wrap(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <style>
      .title{{font:700 32px Arial,Inter,sans-serif;fill:{COLORS['ink']}}}
      .subtitle{{font:400 16px Arial,Inter,sans-serif;fill:{COLORS['muted']}}}
      .h{{font:700 19px Arial,Inter,sans-serif;fill:{COLORS['ink']}}}
      .txt{{font:400 15px Arial,Inter,sans-serif;fill:#334155}}
      .small{{font:400 12px Arial,Inter,sans-serif;fill:#475569}}
      .tiny{{font:400 10px Arial,Inter,sans-serif;fill:#64748B}}
      .panel{{fill:{COLORS['panel']};stroke:#CBD5E1;stroke-width:1.2;rx:8}}
      .card{{fill:#FFFFFF;stroke:#CBD5E1;stroke-width:1.0;rx:7}}
      .grid{{stroke:{COLORS['grid']};stroke-width:1}}
      .axis{{stroke:#475569;stroke-width:1.2}}
    </style>
  </defs>
  <rect width="{width}" height="{height}" fill="#FFFFFF"/>
{body}
</svg>
"""


def points_for_line(rows: list[dict[str, float]], x_key: str, y_key: str, x: int, y: int, w: int, h: int) -> tuple[str, float, float]:
    xs = [row[x_key] for row in rows]
    ys = [row[y_key] for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return x + (value - x_min) / (x_max - x_min) * w

    def sy(value: float) -> float:
        return y + h - (value - y_min) / (y_max - y_min) * h

    return " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, ys)), y_min, y_max


def line_chart(x: int, y: int, w: int, h: int, rows: list[dict[str, float]], y_key: str, color: str, label: str) -> str:
    points, y_min, y_max = points_for_line(rows, "env_step", y_key, x, y, w, h)
    grid = "".join(
        f'<line class="grid" x1="{x}" y1="{y + h * i / 4:.1f}" x2="{x + w}" y2="{y + h * i / 4:.1f}"/>'
        for i in range(5)
    )
    return f"""
      {grid}
      <line class="axis" x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}"/>
      <line class="axis" x1="{x}" y1="{y}" x2="{x}" y2="{y+h}"/>
      <polyline fill="none" stroke="{color}" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" points="{points}"/>
      <text class="small" x="{x}" y="{y-12}">{label}</text>
      <text class="tiny" x="{x+8}" y="{y+14}">{y_max:.3g}</text>
      <text class="tiny" x="{x+8}" y="{y+h-8}">{y_min:.3g}</text>
    """


def bar(x: float, baseline: float, width: float, height: float, color: str, label: str, value: str) -> str:
    return f"""
      <rect x="{x:.1f}" y="{baseline - height:.1f}" width="{width:.1f}" height="{height:.1f}" rx="4" fill="{color}"/>
      <text class="tiny" x="{x + width / 2:.1f}" y="{baseline - height - 8:.1f}" text-anchor="middle">{value}</text>
      <text class="tiny" x="{x + width / 2:.1f}" y="{baseline + 18:.1f}" text-anchor="middle">{label}</text>
    """


def fig_training_curve(curve: list[dict[str, float]], summary: dict) -> None:
    config = summary["config"]
    final = curve[-1]
    body = f"""
  <text class="title" x="70" y="60">Object-Centric World-Model SAC Flow Training</text>
  <text class="subtitle" x="72" y="92">Flow actor + centralized twin-Q + auxiliary object dynamics model | {config['num_envs']} envs, {config['timesteps']:,} env steps</text>
  <rect class="panel" x="70" y="125" width="760" height="610"/>
  <text class="h" x="105" y="168">Reward and termination</text>
  {line_chart(120, 215, 620, 180, curve, "mean_reward", COLORS["blue"], "mean reward")}
  {line_chart(120, 470, 620, 150, curve, "done_rate", COLORS["green"], "done rate")}
  <rect class="panel" x="880" y="125" width="790" height="610"/>
  <text class="h" x="915" y="168">Optimization diagnostics</text>
  {line_chart(930, 215, 620, 150, curve, "critic_loss", COLORS["red"], "critic loss")}
  {line_chart(930, 455, 620, 150, curve, "actor_loss", COLORS["violet"], "actor loss")}
  <rect class="card" x="930" y="635" width="610" height="58"/>
  <text class="txt" x="952" y="670">final reward {final['mean_reward']:.4f} | alpha {final['alpha']:.3f} | throughput {final['steps_per_second']:.1f} steps/s</text>
  <rect class="panel" x="70" y="775" width="1600" height="145"/>
  <text class="h" x="105" y="818">Experiment contract</text>
  <text class="txt" x="105" y="852">actor_mode={config['actor_mode']} | policy_mode={config['policy_mode']} | residual_scale={config['residual_scale']} | world_model_coef={config['world_model_coef']}</text>
  <text class="txt" x="105" y="884">Object state dimension {summary['object_state_dim']} encodes robots, targets, armor blockers and pushable boxes.</text>
"""
    (OUT_DIR / "rl_training_curve_gpu.svg").write_text(svg_wrap(1740, 980, body), encoding="utf-8")


def fig_strategy_metrics(eval_payload: dict, strict_payload: dict) -> None:
    summary = eval_payload["summary"]
    strict = strict_payload["summary"]
    outcome = [
        ("yellow", summary["yellow_win_rate"], COLORS["yellow"]),
        ("blue", summary["blue_win_rate"], COLORS["blue"]),
        ("draw", summary["draw_rate"], COLORS["muted"]),
    ]
    safety = [
        ("static pen", float(summary["static_penetrations_total"]), COLORS["red"]),
        ("box pen", float(summary["box_penetrations_total"]), COLORS["orange"]),
        ("contacts", float(summary["robot_contacts_per_episode"]), COLORS["violet"]),
        ("spin Y", float(summary["abnormal_spin_steps_per_episode"]["yellow"]), COLORS["yellow"]),
        ("spin B", float(summary["abnormal_spin_steps_per_episode"]["blue"]), COLORS["blue"]),
    ]
    body = f"""
  <text class="title" x="70" y="60">Self-Play Strategy Contract Evaluation</text>
  <text class="subtitle" x="72" y="92">{summary['episodes']} stochastic games plus {strict['episodes']} strict replay audits</text>
  <rect class="panel" x="70" y="130" width="760" height="650"/>
  <text class="h" x="105" y="175">Match outcome</text>
  <line class="axis" x1="130" y1="650" x2="690" y2="650"/>
"""
    for index, (label, value, color) in enumerate(outcome):
        body += bar(180 + index * 150, 650, 78, 420 * float(value), color, label, f"{float(value) * 100:.1f}%")
    body += f"""
  <rect class="card" x="120" y="700" width="610" height="48"/>
  <text class="txt" x="145" y="731">mean score: yellow {summary['mean_yellow_score']:.2f}, blue {summary['mean_blue_score']:.2f}; mean time {summary['mean_episode_time_s']:.2f}s</text>
  <rect class="panel" x="880" y="130" width="790" height="650"/>
  <text class="h" x="915" y="175">Safety and behavior checks</text>
  <line class="axis" x1="930" y1="650" x2="1580" y2="650"/>
"""
    max_safety = max(max(value for _, value, _ in safety), 1.0)
    for index, (label, value, color) in enumerate(safety):
        body += bar(965 + index * 118, 650, 58, 380 * value / max_safety, color, label, f"{value:.2g}")
    body += f"""
  <rect class="card" x="930" y="700" width="610" height="48"/>
  <text class="txt" x="955" y="731">strict replay: {strict['hard_violations']} hard violations, {strict['warnings']} warnings, {strict['base_wins_per_episode']:.2f} base wins/game</text>
"""
    (OUT_DIR / "rl_strategy_event_metrics.svg").write_text(svg_wrap(1740, 860, body), encoding="utf-8")


def fig_target_and_base_metrics(eval_payload: dict) -> None:
    summary = eval_payload["summary"]
    dist = summary["normal_hit_count_distribution"]
    base = summary["base_success_by_hits"]
    body = """
  <text class="title" x="70" y="60">Target Clearing and Base-Rush Timing</text>
  <text class="subtitle" x="72" y="92">The policy should use multiple target-count routes instead of collapsing to a single script.</text>
  <rect class="panel" x="70" y="130" width="760" height="650"/>
  <text class="h" x="105" y="175">Normal targets knocked down per game</text>
  <line class="axis" x1="130" y1="650" x2="725" y2="650"/>
"""
    for team_index, team in enumerate(("yellow", "blue")):
        color = COLORS[team]
        for hits in range(1, 5):
            value = float(dist[team][str(hits)])
            body += bar(155 + (hits - 1) * 135 + team_index * 50, 650, 42, 380 * value, color, f"{team[0].upper()}{hits}", f"{value * 100:.0f}%")
    body += """
  <rect class="panel" x="880" y="130" width="790" height="650"/>
  <text class="h" x="915" y="175">Base hit success by normal-target count</text>
  <line class="axis" x1="930" y1="650" x2="1600" y2="650"/>
"""
    for team_index, team in enumerate(("yellow", "blue")):
        color = COLORS[team]
        for hits in range(1, 5):
            item = base[team][str(hits)]
            value = float(item["success_rate"])
            label = f"{team[0].upper()}{hits}"
            text = f"{value * 100:.0f}%/{int(item['attempts'])}"
            body += bar(970 + (hits - 1) * 145 + team_index * 52, 650, 42, 380 * value, color, label, text)
    (OUT_DIR / "rl_target_base_metrics.svg").write_text(svg_wrap(1740, 860, body), encoding="utf-8")


def fig_box_metrics(eval_payload: dict) -> None:
    summary = eval_payload["summary"]
    final_disp = summary["mean_final_box_displacement_m"]
    max_disp = summary["mean_max_box_displacement_m"]
    pushes = summary["push_events_per_episode"]
    body = """
  <text class="title" x="70" y="60">Pushable Box Interaction</text>
  <text class="subtitle" x="72" y="92">Rigid boxes must move in map state and remain collision-safe in strict replay.</text>
  <rect class="panel" x="70" y="130" width="760" height="550"/>
  <text class="h" x="105" y="175">Mean box displacement</text>
  <line class="axis" x1="130" y1="565" x2="700" y2="565"/>
"""
    max_value = max(max(final_disp.values()), max(max_disp.values()), 0.15)
    entries = [
        ("box_ne final", final_disp["box_ne"], COLORS["red"]),
        ("box_ne max", max_disp["box_ne"], COLORS["orange"]),
        ("box_sw final", final_disp["box_sw"], COLORS["blue"]),
        ("box_sw max", max_disp["box_sw"], COLORS["cyan"]),
    ]
    for index, (label, value, color) in enumerate(entries):
        body += bar(155 + index * 130, 565, 58, 330 * float(value) / max_value, color, label, f"{float(value):.3f}m")
    body += f"""
  <rect class="panel" x="880" y="130" width="790" height="550"/>
  <text class="h" x="915" y="175">Push events per game</text>
  <line class="axis" x1="930" y1="565" x2="1540" y2="565"/>
  {bar(1030, 565, 82, 330 * float(pushes['yellow']) / max(max(pushes.values()), 1.0), COLORS['yellow'], 'yellow', f"{float(pushes['yellow']):.2f}")}
  {bar(1220, 565, 82, 330 * float(pushes['blue']) / max(max(pushes.values()), 1.0), COLORS['blue'], 'blue', f"{float(pushes['blue']):.2f}")}
"""
    (OUT_DIR / "rl_box_push_metrics.svg").write_text(svg_wrap(1740, 760, body), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = first_existing(TRAIN_CSV_CANDIDATES)
    train_summary = first_existing(TRAIN_SUMMARY_CANDIDATES)
    eval_path = first_existing(EVAL_CANDIDATES)
    strict_path = first_existing(STRICT_CANDIDATES)

    curve = read_curve(train_csv)
    summary = read_json(train_summary)
    eval_payload = read_json(eval_path)
    strict_payload = read_json(strict_path)

    fig_training_curve(curve, summary)
    fig_strategy_metrics(eval_payload, strict_payload)
    fig_target_and_base_metrics(eval_payload)
    fig_box_metrics(eval_payload)

    manifest = {
        "generated": [
            "docs/figures/rl/rl_training_curve_gpu.svg",
            "docs/figures/rl/rl_strategy_event_metrics.svg",
            "docs/figures/rl/rl_target_base_metrics.svg",
            "docs/figures/rl/rl_box_push_metrics.svg",
        ],
        "sources": [str(train_csv.relative_to(ROOT)), str(train_summary.relative_to(ROOT)), str(eval_path.relative_to(ROOT)), str(strict_path.relative_to(ROOT))],
    }
    (OUT_DIR / "rl_result_figures_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
