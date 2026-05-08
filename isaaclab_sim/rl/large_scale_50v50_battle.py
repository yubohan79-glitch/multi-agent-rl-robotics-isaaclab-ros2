from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs" / "rl_data" / "large_scale_50v50"
FIG_DIR = ROOT / "docs" / "figures" / "large_scale_50v50"
MEDIA_DIR = ROOT / "docs" / "media"


@dataclass
class BattleConfig:
    width_m: float = 80.0
    height_m: float = 50.0
    agents_per_team: int = 50
    dt_s: float = 0.20
    max_steps: int = 420
    max_speed_mps: float = 3.0
    fire_range_m: float = 6.5
    base_fire_range_m: float = 10.0
    fire_cooldown_s: float = 1.20
    agent_hp: float = 3.0
    agent_damage: float = 0.16
    base_hp: float = 45.0
    base_damage: float = 1.10
    blue_base_damage_multiplier: float = 1.0
    capture_radius_m: float = 6.0
    capture_rate: float = 0.055
    shield_progress_to_open: float = 9.0
    obstacle_margin_m: float = 1.1
    contact_radius_m: float = 0.70
    separation_radius_m: float = 1.35
    sensor_range_m: float = 14.0


DEFAULT_THETA = np.array(
    [2.0, 5.0, -1.0, -1.0, 2.5, 0.0, -2.0, 2.0, 1.0, -2.0],
    dtype=np.float64,
)


def config_from_args(args: argparse.Namespace) -> BattleConfig:
    cfg = BattleConfig()
    if hasattr(args, "agents_per_team"):
        cfg.agents_per_team = int(args.agents_per_team)
    if hasattr(args, "max_steps"):
        cfg.max_steps = int(args.max_steps)
    if hasattr(args, "base_hp") and args.base_hp is not None:
        cfg.base_hp = float(args.base_hp)
    if hasattr(args, "base_damage") and args.base_damage is not None:
        cfg.base_damage = float(args.base_damage)
    if hasattr(args, "blue_base_damage_multiplier") and args.blue_base_damage_multiplier is not None:
        cfg.blue_base_damage_multiplier = float(args.blue_base_damage_multiplier)
    if hasattr(args, "capture_rate") and args.capture_rate is not None:
        cfg.capture_rate = float(args.capture_rate)
    if hasattr(args, "shield_progress_to_open") and args.shield_progress_to_open is not None:
        cfg.shield_progress_to_open = float(args.shield_progress_to_open)
    if hasattr(args, "contact_radius") and args.contact_radius is not None:
        cfg.contact_radius_m = float(args.contact_radius)
    if hasattr(args, "separation_radius") and args.separation_radius is not None:
        cfg.separation_radius_m = float(args.separation_radius)
    return cfg


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def policy_params(theta: np.ndarray) -> dict[str, float]:
    theta = np.asarray(theta, dtype=np.float64)
    return {
        "zone_weight": 0.8 + 1.7 * float(sigmoid(theta[0])),
        "base_weight": 0.4 + 2.2 * float(sigmoid(theta[1])),
        "enemy_weight": 0.2 + 1.6 * float(sigmoid(theta[2])),
        "cohesion_weight": 0.1 + 1.1 * float(sigmoid(theta[3])),
        "separation_weight": 0.8 + 2.0 * float(sigmoid(theta[4])),
        "flank_bias_m": 9.0 * float(np.tanh(theta[5])),
        "defense_weight": 0.2 + 1.8 * float(sigmoid(theta[6])),
        "aggression": float(sigmoid(theta[7])),
        "spread_m": 1.0 + 4.0 * float(sigmoid(theta[8])),
        "retreat_health": 0.12 + 0.55 * float(sigmoid(theta[9])),
    }


class LargeScaleBattle50v50:
    def __init__(self, config: BattleConfig | None = None):
        self.cfg = config or BattleConfig()
        self.zones = np.array([[34.0, 13.0], [40.0, 25.0], [46.0, 37.0]], dtype=np.float64)
        self.yellow_base = np.array([4.5, 25.0], dtype=np.float64)
        self.blue_base = np.array([75.5, 25.0], dtype=np.float64)
        self.obstacles = np.array(
            [
                [25.0, 6.0, 28.0, 18.5],
                [52.0, 31.5, 55.0, 44.0],
                [37.6, 21.0, 42.4, 29.0],
            ],
            dtype=np.float64,
        )

    def _initial_positions(self, team: str, rng: np.random.Generator) -> np.ndarray:
        n = self.cfg.agents_per_team
        rows = min(5, n)
        cols = int(math.ceil(n / rows))
        grid = []
        for r in range(rows):
            for c in range(cols):
                grid.append((r, c))
        grid = np.array(grid[:n], dtype=np.float64)
        y = 7.0 + grid[:, 0] * 8.5 + rng.normal(0.0, 0.25, size=n)
        if team == "yellow":
            x = 7.0 + grid[:, 1] * 0.9 + rng.normal(0.0, 0.15, size=n)
        else:
            y = self.cfg.height_m - y
            x = 73.0 - grid[:, 1] * 0.9 + rng.normal(0.0, 0.15, size=n)
        return np.stack([x, y], axis=1)

    def _obstacle_repulsion(self, pos: np.ndarray) -> tuple[np.ndarray, int]:
        force = np.zeros_like(pos)
        contacts = 0
        margin = self.cfg.obstacle_margin_m
        for rect in self.obstacles:
            xmin, ymin, xmax, ymax = rect
            closest_x = np.clip(pos[:, 0], xmin, xmax)
            closest_y = np.clip(pos[:, 1], ymin, ymax)
            diff = pos - np.stack([closest_x, closest_y], axis=1)
            dist = np.linalg.norm(diff, axis=1)
            inside = (pos[:, 0] >= xmin) & (pos[:, 0] <= xmax) & (pos[:, 1] >= ymin) & (pos[:, 1] <= ymax)
            contacts += int(np.count_nonzero(inside))
            if np.any(inside):
                left = np.abs(pos[:, 0] - xmin)
                right = np.abs(xmax - pos[:, 0])
                down = np.abs(pos[:, 1] - ymin)
                up = np.abs(ymax - pos[:, 1])
                nearest = np.stack([left, right, down, up], axis=1).argmin(axis=1)
                push = np.zeros_like(pos)
                push[nearest == 0, 0] = -1.0
                push[nearest == 1, 0] = 1.0
                push[nearest == 2, 1] = -1.0
                push[nearest == 3, 1] = 1.0
                diff[inside] = push[inside]
                dist[inside] = 0.01
            active = dist < margin
            force[active] += diff[active] / (dist[active, None] + 1e-6) * (margin - dist[active, None])
        return force, contacts

    def _separation(self, pos: np.ndarray, alive: np.ndarray) -> np.ndarray:
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-6
        weight = np.clip(self.cfg.separation_radius_m - dist, 0.0, None)
        weight *= alive[None, :] * alive[:, None]
        np.fill_diagonal(weight, 0.0)
        return np.sum(delta / dist[:, :, None] * weight[:, :, None], axis=1)

    def _nearest_enemy(self, own_pos: np.ndarray, own_alive: np.ndarray, enemy_pos: np.ndarray, enemy_alive: np.ndarray):
        delta = enemy_pos[None, :, :] - own_pos[:, None, :]
        dist = np.linalg.norm(delta, axis=2)
        dist = np.where(enemy_alive[None, :] & own_alive[:, None], dist, 1e9)
        idx = np.argmin(dist, axis=1)
        nearest_dist = dist[np.arange(len(own_pos)), idx]
        nearest_vec = enemy_pos[idx] - own_pos
        nearest_vec = np.where(nearest_dist[:, None] < 1e8, nearest_vec, 0.0)
        return idx, nearest_dist, nearest_vec

    def _segment_blocked(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if len(src) == 0:
            return np.zeros((0,), dtype=bool)
        t = np.linspace(0.15, 0.85, 5, dtype=np.float64)
        points = src[:, None, :] * (1.0 - t[None, :, None]) + dst[:, None, :] * t[None, :, None]
        blocked = np.zeros((len(src),), dtype=bool)
        for rect in self.obstacles:
            xmin, ymin, xmax, ymax = rect
            inside = (
                (points[:, :, 0] >= xmin)
                & (points[:, :, 0] <= xmax)
                & (points[:, :, 1] >= ymin)
                & (points[:, :, 1] <= ymax)
            )
            blocked |= inside.any(axis=1)
        return blocked

    def _zone_update(self, yellow_pos: np.ndarray, yellow_alive: np.ndarray, blue_pos: np.ndarray, blue_alive: np.ndarray, state: np.ndarray) -> np.ndarray:
        ydist = np.linalg.norm(yellow_pos[:, None, :] - self.zones[None, :, :], axis=2)
        bdist = np.linalg.norm(blue_pos[:, None, :] - self.zones[None, :, :], axis=2)
        yc = np.sum((ydist <= self.cfg.capture_radius_m) & yellow_alive[:, None], axis=0)
        bc = np.sum((bdist <= self.cfg.capture_radius_m) & blue_alive[:, None], axis=0)
        influence = (yc - bc) / (yc + bc + 4.0)
        return np.clip(state + self.cfg.capture_rate * influence, -1.0, 1.0)

    def _policy_velocity(
        self,
        team: str,
        pos: np.ndarray,
        alive: np.ndarray,
        hp: np.ndarray,
        enemy_pos: np.ndarray,
        enemy_alive: np.ndarray,
        zone_state: np.ndarray,
        shield_open: bool,
        progress_ratio: float,
        theta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        p = policy_params(theta)
        n = self.cfg.agents_per_team
        squad = np.floor(np.arange(n) * 5 / max(1, n)).astype(int)
        raw_zone_idx = squad % 3
        zone_idx = raw_zone_idx if team == "yellow" else 2 - raw_zone_idx
        flank = np.where((np.arange(n) % 2) == 0, 1.0, -1.0)
        flank_dir = flank if team == "yellow" else -flank
        side = 1.0 if team == "yellow" else -1.0
        own_base = self.yellow_base if team == "yellow" else self.blue_base
        enemy_base = self.blue_base if team == "yellow" else self.yellow_base
        zone_targets = self.zones[zone_idx].copy()
        zone_targets[:, 1] += flank_dir * (0.4 * p["spread_m"] + 0.35 * p["flank_bias_m"])
        base_targets = np.repeat(enemy_base[None, :], n, axis=0)
        base_targets[:, 0] -= side * 3.5
        base_targets[:, 1] += flank_dir * p["flank_bias_m"]

        idx, nearest_dist, nearest_vec = self._nearest_enemy(pos, alive, enemy_pos, enemy_alive)
        enemy_dir = nearest_vec / (nearest_dist[:, None] + 1e-6)
        enemy_active = nearest_dist < self.cfg.sensor_range_m

        centroids = np.zeros_like(pos)
        for s in range(5):
            mask = (squad == s) & alive
            if np.any(mask):
                centroids[squad == s] = pos[mask].mean(axis=0)
            else:
                centroids[squad == s] = own_base

        low_hp = hp < p["retreat_health"]
        controlled = np.count_nonzero(zone_state > 0.35) if team == "yellow" else np.count_nonzero(zone_state < -0.35)
        base_gate = shield_open or controlled >= 2 or progress_ratio > 0.78
        base_weight = p["base_weight"] * (6.5 if base_gate else 1.2)
        assault_mask = (squad >= 3) | (base_gate & (squad >= 2))
        defend = ((squad == 4) | low_hp) & (~assault_mask)
        defense_targets = np.repeat(own_base[None, :], n, axis=0)
        defense_targets[:, 0] += side * 8.0
        defense_targets[:, 1] += flank_dir * 6.0

        desired = np.zeros_like(pos)
        desired += p["zone_weight"] * (zone_targets - pos) * np.where(assault_mask[:, None], 0.12, 1.0)
        desired += base_weight * assault_mask[:, None] * (base_targets - pos)
        desired += p["enemy_weight"] * p["aggression"] * enemy_active[:, None] * enemy_dir * (~assault_mask)[:, None]
        desired += p["cohesion_weight"] * (centroids - pos)
        desired += p["defense_weight"] * defend[:, None] * (defense_targets - pos)
        desired += p["separation_weight"] * self._separation(pos, alive)
        obstacle_force, _ = self._obstacle_repulsion(pos)
        desired += 11.0 * obstacle_force
        desired += rng.normal(0.0, 0.08, size=desired.shape)
        desired[~alive] = 0.0

        norm = np.linalg.norm(desired, axis=1, keepdims=True)
        direction = desired / (norm + 1e-6)
        speed = self.cfg.max_speed_mps * (0.68 + 0.32 * p["aggression"])
        speed_scale = np.where((nearest_dist < self.cfg.fire_range_m * 0.9) & (~assault_mask), 0.42, 1.0)
        speed_scale = np.where(low_hp, 0.72, speed_scale)
        return direction * speed * speed_scale[:, None]

    def _apply_shots(
        self,
        shooter_team: str,
        shooter_pos: np.ndarray,
        shooter_alive: np.ndarray,
        shooter_cd: np.ndarray,
        target_pos: np.ndarray,
        target_alive: np.ndarray,
        target_hp: np.ndarray,
        base_open: bool,
        base_hp: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
        enemy_base = self.blue_base if shooter_team == "yellow" else self.yellow_base
        base_dist = np.linalg.norm(shooter_pos - enemy_base[None, :], axis=1)
        base_candidates = shooter_alive & (shooter_cd <= 0.0) & (base_dist <= self.cfg.base_fire_range_m)
        base_ids = np.flatnonzero(base_candidates)
        base_blocked = self._segment_blocked(shooter_pos[base_ids], np.repeat(enemy_base[None, :], len(base_ids), axis=0)) if len(base_ids) else np.zeros((0,), dtype=bool)
        base_legal = base_ids[~base_blocked]
        shielded = 0
        base_damage = 0.0
        if len(base_legal):
            if base_open:
                chance = 0.55 + 0.30 * (1.0 - base_dist[base_legal] / self.cfg.base_fire_range_m)
                base_hits = base_legal[rng.random(len(base_legal)) < chance]
                damage_multiplier = self.cfg.blue_base_damage_multiplier if shooter_team == "blue" else 1.0
                base_damage = float(len(base_hits) * self.cfg.base_damage * damage_multiplier)
                base_hp = max(0.0, base_hp - base_damage)
                shooter_cd[base_hits] = self.cfg.fire_cooldown_s
            else:
                shielded = int(len(base_legal))
                shooter_cd[base_legal] = self.cfg.fire_cooldown_s

        idx, nearest_dist, _ = self._nearest_enemy(shooter_pos, shooter_alive, target_pos, target_alive)
        can_fire = shooter_alive & (shooter_cd <= 0.0) & (nearest_dist <= self.cfg.fire_range_m)
        shooter_ids = np.flatnonzero(can_fire)
        blocked = self._segment_blocked(shooter_pos[shooter_ids], target_pos[idx[shooter_ids]]) if len(shooter_ids) else np.zeros((0,), dtype=bool)
        legal_ids = shooter_ids[~blocked]
        hit_chance = 0.62 + 0.26 * (1.0 - nearest_dist[legal_ids] / self.cfg.fire_range_m)
        hits = legal_ids[rng.random(len(legal_ids)) < hit_chance]
        damage = np.zeros_like(target_hp)
        if len(hits):
            np.add.at(damage, idx[hits], self.cfg.agent_damage)
            shooter_cd[hits] = self.cfg.fire_cooldown_s
        target_hp = np.maximum(0.0, target_hp - damage)

        stats = {
            "agent_shots": float(len(legal_ids)),
            "agent_hits": float(len(hits)),
            "base_shots": float(len(base_legal)),
            "base_damage": float(base_damage),
            "shielded_base_shots": float(shielded),
        }
        return shooter_cd, target_hp, base_hp, stats

    def run_episode(
        self,
        theta_yellow: np.ndarray,
        theta_blue: np.ndarray,
        seed: int,
        collect_trace: bool = False,
        trace_stride: int = 2,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        c = self.cfg
        yp = self._initial_positions("yellow", rng)
        bp = self._initial_positions("blue", rng)
        yhp = np.full(c.agents_per_team, c.agent_hp, dtype=np.float64)
        bhp = np.full(c.agents_per_team, c.agent_hp, dtype=np.float64)
        ycd = np.zeros(c.agents_per_team, dtype=np.float64)
        bcd = np.zeros(c.agents_per_team, dtype=np.float64)
        ybase = c.base_hp
        bbase = c.base_hp
        zone_state = np.zeros(3, dtype=np.float64)
        yellow_shield_progress = 0.0
        blue_shield_progress = 0.0

        stats = {
            "yellow_agent_hits": 0.0,
            "blue_agent_hits": 0.0,
            "yellow_base_damage": 0.0,
            "blue_base_damage": 0.0,
            "yellow_shielded_base_shots": 0.0,
            "blue_shielded_base_shots": 0.0,
            "robot_contacts": 0,
            "obstacle_contacts": 0,
            "yellow_zone_steps": 0,
            "blue_zone_steps": 0,
            "yellow_base_open_steps": 0,
            "blue_base_open_steps": 0,
        }
        trace = []

        for step in range(c.max_steps):
            ya = yhp > 0.0
            ba = bhp > 0.0
            if not np.any(ya) or not np.any(ba) or ybase <= 0.0 or bbase <= 0.0:
                break
            zone_state = self._zone_update(yp, ya, bp, ba, zone_state)
            yellow_control = int(np.count_nonzero(zone_state > 0.35))
            blue_control = int(np.count_nonzero(zone_state < -0.35))
            stats["yellow_zone_steps"] += yellow_control
            stats["blue_zone_steps"] += blue_control
            yellow_shield_progress = min(c.shield_progress_to_open, yellow_shield_progress + yellow_control * c.dt_s)
            blue_shield_progress = min(c.shield_progress_to_open, blue_shield_progress + blue_control * c.dt_s)
            yellow_base_open = yellow_shield_progress >= c.shield_progress_to_open
            blue_base_open = blue_shield_progress >= c.shield_progress_to_open
            stats["yellow_base_open_steps"] += int(yellow_base_open)
            stats["blue_base_open_steps"] += int(blue_base_open)

            yv = self._policy_velocity("yellow", yp, ya, yhp, bp, ba, zone_state, yellow_base_open, yellow_shield_progress / c.shield_progress_to_open, theta_yellow, rng)
            bv = self._policy_velocity("blue", bp, ba, bhp, yp, ya, zone_state, blue_base_open, blue_shield_progress / c.shield_progress_to_open, theta_blue, rng)
            yp = yp + yv * c.dt_s
            bp = bp + bv * c.dt_s
            yp[:, 0] = np.clip(yp[:, 0], 1.0, c.width_m - 1.0)
            yp[:, 1] = np.clip(yp[:, 1], 1.0, c.height_m - 1.0)
            bp[:, 0] = np.clip(bp[:, 0], 1.0, c.width_m - 1.0)
            bp[:, 1] = np.clip(bp[:, 1], 1.0, c.height_m - 1.0)
            yobs, yc = self._obstacle_repulsion(yp)
            bobs, bc = self._obstacle_repulsion(bp)
            yp += 0.95 * yobs
            bp += 0.95 * bobs
            stats["obstacle_contacts"] += yc + bc

            pair_dist = np.linalg.norm(yp[:, None, :] - bp[None, :, :], axis=2)
            contacts = (pair_dist < c.contact_radius_m) & ya[:, None] & ba[None, :]
            stats["robot_contacts"] += int(np.count_nonzero(contacts))

            ycd = np.maximum(0.0, ycd - c.dt_s)
            bcd = np.maximum(0.0, bcd - c.dt_s)
            ycd, bhp, bbase, yshot = self._apply_shots("yellow", yp, ya, ycd, bp, ba, bhp, yellow_base_open, bbase, rng)
            bcd, yhp, ybase, bshot = self._apply_shots("blue", bp, ba, bcd, yp, ya, yhp, blue_base_open, ybase, rng)
            stats["yellow_agent_hits"] += yshot["agent_hits"]
            stats["blue_agent_hits"] += bshot["agent_hits"]
            stats["yellow_base_damage"] += yshot["base_damage"]
            stats["blue_base_damage"] += bshot["base_damage"]
            stats["yellow_shielded_base_shots"] += yshot["shielded_base_shots"]
            stats["blue_shielded_base_shots"] += bshot["shielded_base_shots"]

            if collect_trace and step % trace_stride == 0:
                trace.append(
                    {
                        "step": step,
                        "yellow_pos": yp.copy(),
                        "blue_pos": bp.copy(),
                        "yellow_alive": (yhp > 0.0).copy(),
                        "blue_alive": (bhp > 0.0).copy(),
                        "zone_state": zone_state.copy(),
                        "yellow_base_hp": float(ybase),
                        "blue_base_hp": float(bbase),
                        "yellow_base_open": bool(yellow_base_open),
                        "blue_base_open": bool(blue_base_open),
                    }
                )

        ya = yhp > 0.0
        ba = bhp > 0.0
        yellow_kills = int(c.agents_per_team - np.count_nonzero(ba))
        blue_kills = int(c.agents_per_team - np.count_nonzero(ya))
        yellow_score = yellow_kills * 1.2 + (c.base_hp - bbase) * 5.0 + stats["yellow_zone_steps"] * 0.03 + stats["yellow_base_open_steps"] * 0.05 + np.count_nonzero(ya) * 0.03
        blue_score = blue_kills * 1.2 + (c.base_hp - ybase) * 5.0 + stats["blue_zone_steps"] * 0.03 + stats["blue_base_open_steps"] * 0.05 + np.count_nonzero(ba) * 0.03
        if bbase <= 0.0 and ybase > 0.0:
            winner = "yellow"
        elif ybase <= 0.0 and bbase > 0.0:
            winner = "blue"
        elif abs(yellow_score - blue_score) < 1e-6:
            winner = "draw"
        else:
            winner = "yellow" if yellow_score > blue_score else "blue"

        result: dict[str, Any] = {
            "winner": winner,
            "elapsed_s": round(step * c.dt_s, 3),
            "steps": int(step),
            "yellow_score": float(yellow_score),
            "blue_score": float(blue_score),
            "yellow_alive": int(np.count_nonzero(ya)),
            "blue_alive": int(np.count_nonzero(ba)),
            "yellow_kills": yellow_kills,
            "blue_kills": blue_kills,
            "yellow_base_hp": float(ybase),
            "blue_base_hp": float(bbase),
            "final_zone_state": zone_state.tolist(),
            "robot_contacts": int(stats["robot_contacts"]),
            "obstacle_contacts": int(stats["obstacle_contacts"]),
            "yellow_agent_hits": float(stats["yellow_agent_hits"]),
            "blue_agent_hits": float(stats["blue_agent_hits"]),
            "yellow_base_damage": float(stats["yellow_base_damage"]),
            "blue_base_damage": float(stats["blue_base_damage"]),
            "yellow_shielded_base_shots": float(stats["yellow_shielded_base_shots"]),
            "blue_shielded_base_shots": float(stats["blue_shielded_base_shots"]),
            "yellow_base_open_rate": float(stats["yellow_base_open_steps"] / max(1, step + 1)),
            "blue_base_open_rate": float(stats["blue_base_open_steps"] / max(1, step + 1)),
        }
        if collect_trace:
            result["trace"] = trace
        return result


def side_fitness(metrics: dict[str, Any], side: str) -> float:
    sign = 1.0 if side == "yellow" else -1.0
    score_diff = sign * (metrics["yellow_score"] - metrics["blue_score"])
    if metrics["winner"] == side:
        win_bonus = 35.0
    elif metrics["winner"] == "draw":
        win_bonus = 0.0
    else:
        win_bonus = -35.0
    alive = metrics[f"{side}_alive"]
    base_damage = metrics[f"{side}_base_damage"]
    opp = "blue" if side == "yellow" else "yellow"
    open_rate = metrics[f"{side}_base_open_rate"]
    shielded = metrics[f"{side}_shielded_base_shots"]
    contacts = metrics["robot_contacts"]
    obstacle = metrics["obstacle_contacts"]
    return float(score_diff + win_bonus + 0.12 * alive + 5.0 * base_damage + 8.0 * open_rate - 0.012 * contacts - 0.015 * obstacle - 0.05 * shielded)


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    winners = [e["winner"] for e in episodes]
    summary = {
        "episodes": n,
        "yellow_win_rate": winners.count("yellow") / n,
        "blue_win_rate": winners.count("blue") / n,
        "draw_rate": winners.count("draw") / n,
    }
    keys = [
        "elapsed_s",
        "yellow_score",
        "blue_score",
        "yellow_alive",
        "blue_alive",
        "yellow_kills",
        "blue_kills",
        "yellow_base_hp",
        "blue_base_hp",
        "robot_contacts",
        "obstacle_contacts",
        "yellow_base_damage",
        "blue_base_damage",
        "yellow_shielded_base_shots",
        "blue_shielded_base_shots",
        "yellow_base_open_rate",
        "blue_base_open_rate",
    ]
    for key in keys:
        summary[f"mean_{key}"] = float(np.mean([e[key] for e in episodes]))
    zone = np.array([e["final_zone_state"] for e in episodes], dtype=np.float64)
    summary["mean_final_zone_state"] = zone.mean(axis=0).round(4).tolist()
    summary["p95_robot_contacts"] = float(np.percentile([e["robot_contacts"] for e in episodes], 95))
    summary["p95_obstacle_contacts"] = float(np.percentile([e["obstacle_contacts"] for e in episodes], 95))
    return summary


def train(args: argparse.Namespace) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = LargeScaleBattle50v50(config_from_args(args))
    rng = np.random.default_rng(args.seed)
    init_checkpoint = getattr(args, "init_checkpoint", "")
    if init_checkpoint and Path(init_checkpoint).exists():
        theta = np.array(load_checkpoint(Path(init_checkpoint))["theta"], dtype=np.float64)
    else:
        theta = DEFAULT_THETA.copy()
    sigma = float(args.sigma)
    archive: list[np.ndarray] = [DEFAULT_THETA.copy(), theta.copy()]
    curve = []
    best_theta = theta.copy()
    best_fitness = -1e9
    start = __import__("time").time()

    for gen in range(args.generations):
        candidates = []
        fitnesses = []
        for _ in range(args.population):
            candidate = theta + rng.normal(0.0, sigma, size=theta.shape)
            opponent = archive[int(rng.integers(0, len(archive)))]
            scores = []
            for ep in range(args.episodes_per_candidate):
                seed = args.seed + gen * 100000 + ep * 1000 + len(candidates)
                m1 = env.run_episode(candidate, opponent, seed)
                m2 = env.run_episode(opponent, candidate, seed + 17)
                scores.append(side_fitness(m1, "yellow"))
                scores.append(side_fitness(m2, "blue"))
            candidates.append(candidate)
            fitnesses.append(float(np.mean(scores)))
        order = np.argsort(fitnesses)[::-1]
        elite_n = max(2, int(args.population * args.elite_frac))
        elites = np.array([candidates[i] for i in order[:elite_n]])
        elite_scores = np.array([fitnesses[i] for i in order[:elite_n]], dtype=np.float64)
        weights = elite_scores - elite_scores.min() + 1e-6
        weights = weights / weights.sum()
        theta = (elites * weights[:, None]).sum(axis=0)
        gen_best = float(fitnesses[order[0]])
        gen_mean = float(np.mean(fitnesses))
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_theta = candidates[order[0]].copy()
        if gen % max(1, args.archive_interval) == 0:
            archive.append(best_theta.copy())
            archive = archive[-args.archive_size :]
        sigma = max(args.min_sigma, sigma * args.sigma_decay)

        eval_eps = []
        for k in range(args.probe_episodes):
            eval_eps.append(env.run_episode(best_theta, best_theta, args.seed + 900000 + gen * 100 + k))
        probe = summarize_episodes(eval_eps)
        row = {
            "generation": gen,
            "population": args.population,
            "episodes_seen": (gen + 1) * args.population * args.episodes_per_candidate * 2,
            "best_fitness": best_fitness,
            "generation_best_fitness": gen_best,
            "generation_mean_fitness": gen_mean,
            "sigma": sigma,
            "probe_yellow_win_rate": probe["yellow_win_rate"],
            "probe_blue_win_rate": probe["blue_win_rate"],
            "probe_draw_rate": probe["draw_rate"],
            "probe_mean_elapsed_s": probe["mean_elapsed_s"],
            "probe_mean_robot_contacts": probe["mean_robot_contacts"],
            "probe_mean_obstacle_contacts": probe["mean_obstacle_contacts"],
            "probe_mean_yellow_alive": probe["mean_yellow_alive"],
            "probe_mean_blue_alive": probe["mean_blue_alive"],
        }
        curve.append(row)
        with (DATA_DIR / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
            writer.writeheader()
            writer.writerows(curve)
        if args.verbose and (gen == 0 or (gen + 1) % args.log_interval == 0 or gen == args.generations - 1):
            print(
                f"gen {gen + 1:04d}/{args.generations} "
                f"best={best_fitness:.3f} mean={gen_mean:.3f} "
                f"probe Y/B/D={probe['yellow_win_rate']:.2f}/{probe['blue_win_rate']:.2f}/{probe['draw_rate']:.2f} "
                f"contacts={probe['mean_robot_contacts']:.1f}",
                flush=True,
            )

    selection_candidates = [DEFAULT_THETA.copy(), best_theta.copy(), theta.copy()] + [item.copy() for item in archive]
    selection_rows = []
    selected_theta = DEFAULT_THETA.copy()
    selected_score = -1e9
    for idx, candidate in enumerate(selection_candidates):
        eval_eps = [env.run_episode(candidate, candidate, args.seed + 7000000 + idx * 1000 + k) for k in range(args.selection_episodes)]
        summary = summarize_episodes(eval_eps)
        balance_penalty = abs(summary["yellow_win_rate"] - summary["blue_win_rate"])
        base_deficit = max(0.0, 8.0 - summary["mean_yellow_base_damage"]) + max(0.0, 8.0 - summary["mean_blue_base_damage"])
        contact_penalty = max(0.0, summary["mean_robot_contacts"] - 180.0) / 180.0
        score = 20.0 - 16.0 * balance_penalty - 1.4 * base_deficit - 2.0 * contact_penalty
        row = {"candidate": idx, "selection_score": score, **summary}
        selection_rows.append(row)
        if score > selected_score:
            selected_score = score
            selected_theta = candidate.copy()

    training_time = __import__("time").time() - start
    ckpt = {
        "algorithm": "population_based_swarm_flow_policy_search",
        "scenario": f"large_scale_{env.cfg.agents_per_team}v{env.cfg.agents_per_team}_control_zone_base_assault",
        "seed": args.seed,
        "theta": selected_theta.round(8).tolist(),
        "policy_params": policy_params(selected_theta),
        "config": asdict(env.cfg),
        "training": {
            "generations": args.generations,
            "population": args.population,
            "episodes_per_candidate": args.episodes_per_candidate,
            "probe_episodes": args.probe_episodes,
            "episodes_seen": args.generations * args.population * args.episodes_per_candidate * 2,
            "best_fitness": best_fitness,
            "selection_episodes_per_candidate": args.selection_episodes,
            "selected_validation_score": selected_score,
            "wall_time_s": training_time,
        },
    }
    (DATA_DIR / "policy_checkpoint.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    with (DATA_DIR / "training_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    summary = {"checkpoint": "docs/rl_data/large_scale_50v50/policy_checkpoint.json", **ckpt["training"]}
    (DATA_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (DATA_DIR / "policy_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selection_rows)
    return ckpt


def load_checkpoint(path: Path = DATA_DIR / "policy_checkpoint.json") -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = LargeScaleBattle50v50(config_from_args(args))
    ckpt = load_checkpoint(Path(args.checkpoint))
    theta = np.array(ckpt["theta"], dtype=np.float64)
    episodes = [env.run_episode(theta, theta, args.seed + i) for i in range(args.episodes)]
    summary = summarize_episodes(episodes)
    payload = {
        "scenario": f"large_scale_{env.cfg.agents_per_team}v{env.cfg.agents_per_team}_control_zone_base_assault",
        "policy_checkpoint": str(Path(args.checkpoint).as_posix()),
        "summary": summary,
        "episodes": episodes,
    }
    (DATA_DIR / "eval_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (DATA_DIR / "eval_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "episode",
            "winner",
            "elapsed_s",
            "yellow_score",
            "blue_score",
            "yellow_alive",
            "blue_alive",
            "yellow_kills",
            "blue_kills",
            "yellow_base_hp",
            "blue_base_hp",
            "robot_contacts",
            "obstacle_contacts",
            "yellow_base_damage",
            "blue_base_damage",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, ep in enumerate(episodes):
            writer.writerow({"episode": i, **{k: ep[k] for k in fieldnames if k != "episode"}})
    return payload


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def world_to_px(pos: np.ndarray, width: int, height: int, cfg: BattleConfig) -> np.ndarray:
    scale = min((width - 120) / cfg.width_m, (height - 120) / cfg.height_m)
    ox = (width - cfg.width_m * scale) / 2.0
    oy = (height - cfg.height_m * scale) / 2.0
    out = np.empty_like(pos, dtype=np.float64)
    out[:, 0] = ox + pos[:, 0] * scale
    out[:, 1] = oy + (cfg.height_m - pos[:, 1]) * scale
    return out


def render_frame(env: LargeScaleBattle50v50, item: dict[str, Any], width: int = 1920, height: int = 1080) -> Image.Image:
    cfg = env.cfg
    img = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(img)
    title_font = _font(34)
    label_font = _font(22)
    small_font = _font(18)
    scale = min((width - 120) / cfg.width_m, (height - 120) / cfg.height_m)
    ox = (width - cfg.width_m * scale) / 2.0
    oy = (height - cfg.height_m * scale) / 2.0

    def rect_world(rect, fill, outline="#334155"):
        xmin, ymin, xmax, ymax = rect
        x1 = ox + xmin * scale
        x2 = ox + xmax * scale
        y1 = oy + (cfg.height_m - ymax) * scale
        y2 = oy + (cfg.height_m - ymin) * scale
        draw.rounded_rectangle([x1, y1, x2, y2], radius=6, fill=fill, outline=outline, width=2)

    draw.rectangle([ox, oy, ox + cfg.width_m * scale, oy + cfg.height_m * scale], fill="#ffffff", outline="#0f172a", width=3)
    for i in range(1, 4):
        x = ox + cfg.width_m * scale * i / 4
        draw.line([x, oy, x, oy + cfg.height_m * scale], fill="#e2e8f0", width=1)
    for i in range(1, 4):
        y = oy + cfg.height_m * scale * i / 4
        draw.line([ox, y, ox + cfg.width_m * scale, y], fill="#e2e8f0", width=1)
    for rect in env.obstacles:
        rect_world(rect, "#cbd5e1")
    for idx, zone in enumerate(env.zones):
        state = item["zone_state"][idx]
        color = "#facc15" if state > 0.25 else "#3b82f6" if state < -0.25 else "#e2e8f0"
        center = world_to_px(zone[None, :], width, height, cfg)[0]
        r = cfg.capture_radius_m * scale
        draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], outline=color, width=5)
        draw.text((center[0] - 10, center[1] - 12), str(idx + 1), font=label_font, fill="#0f172a")

    for base, color, hp in [(env.yellow_base, "#eab308", item["yellow_base_hp"]), (env.blue_base, "#2563eb", item["blue_base_hp"])]:
        p = world_to_px(base[None, :], width, height, cfg)[0]
        draw.rounded_rectangle([p[0] - 24, p[1] - 34, p[0] + 24, p[1] + 34], radius=8, fill=color, outline="#0f172a", width=2)
        draw.rectangle([p[0] - 35, p[1] + 42, p[0] + 35, p[1] + 50], fill="#e5e7eb")
        draw.rectangle([p[0] - 35, p[1] + 42, p[0] - 35 + 70 * max(0.0, hp / cfg.base_hp), p[1] + 50], fill="#22c55e")

    for key, color, edge in [("yellow", "#facc15", "#854d0e"), ("blue", "#60a5fa", "#1e3a8a")]:
        pos = world_to_px(item[f"{key}_pos"], width, height, cfg)
        alive = item[f"{key}_alive"]
        for p, ok in zip(pos, alive):
            if ok:
                draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=color, outline=edge)
            else:
                draw.line([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill="#94a3b8", width=2)
                draw.line([p[0] - 4, p[1] + 4, p[0] + 4, p[1] - 4], fill="#94a3b8", width=2)

    y_alive = int(np.count_nonzero(item["yellow_alive"]))
    b_alive = int(np.count_nonzero(item["blue_alive"]))
    draw.text((70, 30), "Large-Scale 50v50 Multi-Agent Battle Replay", font=title_font, fill="#0f172a")
    draw.text(
        (70, 75),
        f"t={item['step'] * cfg.dt_s:.1f}s   yellow alive={y_alive}/{cfg.agents_per_team}   blue alive={b_alive}/{cfg.agents_per_team}",
        font=label_font,
        fill="#334155",
    )
    draw.text((width - 520, 35), f"base hp: Y {item['yellow_base_hp']:.1f} | B {item['blue_base_hp']:.1f}", font=label_font, fill="#334155")
    draw.text((width - 520, 75), "zones: yellow if gold, blue if blue, neutral if gray", font=small_font, fill="#64748b")
    return img


def render_video(args: argparse.Namespace) -> dict[str, str]:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    env = LargeScaleBattle50v50(config_from_args(args))
    ckpt = load_checkpoint(Path(args.checkpoint))
    theta = np.array(ckpt["theta"], dtype=np.float64)
    ep = env.run_episode(theta, theta, args.seed, collect_trace=True, trace_stride=args.trace_stride)
    trace = ep["trace"]
    np.savez_compressed(
        DATA_DIR / "isaaclab_replay_trace.npz",
        yellow_pos=np.stack([item["yellow_pos"] for item in trace], axis=0),
        blue_pos=np.stack([item["blue_pos"] for item in trace], axis=0),
        yellow_alive=np.stack([item["yellow_alive"] for item in trace], axis=0),
        blue_alive=np.stack([item["blue_alive"] for item in trace], axis=0),
        zone_state=np.stack([item["zone_state"] for item in trace], axis=0),
        yellow_base_hp=np.array([item["yellow_base_hp"] for item in trace], dtype=np.float32),
        blue_base_hp=np.array([item["blue_base_hp"] for item in trace], dtype=np.float32),
        yellow_base_open=np.array([item["yellow_base_open"] for item in trace], dtype=np.bool_),
        blue_base_open=np.array([item["blue_base_open"] for item in trace], dtype=np.bool_),
        dt=np.array([env.cfg.dt_s * args.trace_stride], dtype=np.float32),
        width_m=np.array([env.cfg.width_m], dtype=np.float32),
        height_m=np.array([env.cfg.height_m], dtype=np.float32),
    )
    mp4_path = MEDIA_DIR / "large_scale_50v50_replay.mp4"
    gif_path = MEDIA_DIR / "large_scale_50v50_replay.gif"
    fps = args.fps
    max_frames = max(1, int(args.seconds * fps))
    frame_indices = np.linspace(0, len(trace) - 1, max_frames).round().astype(int)
    gif_target = max(1, int(args.gif_seconds * args.gif_fps))
    gif_pick = set(np.linspace(0, max_frames - 1, gif_target).round().astype(int).tolist())
    gif_frames = []
    with imageio.get_writer(mp4_path, fps=fps, quality=8, macro_block_size=1) as writer:
        for out_idx, trace_idx in enumerate(frame_indices):
            frame = render_frame(env, trace[int(trace_idx)], args.width, args.height)
            writer.append_data(np.asarray(frame))
            if out_idx in gif_pick:
                gif_frames.append(frame.resize((960, 540), Image.Resampling.LANCZOS))
    imageio.mimsave(gif_path, gif_frames, fps=args.gif_fps, loop=0)
    return {"mp4": str(mp4_path), "gif": str(gif_path)}


def make_figures(args: argparse.Namespace) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    with (DATA_DIR / "training_curve.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            curve_rows.append({k: float(v) for k, v in row.items()})
    eval_payload = json.loads((DATA_DIR / "eval_summary.json").read_text(encoding="utf-8"))
    summary = eval_payload["summary"]

    x = [r["generation"] for r in curve_rows]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=160)
    fig.suptitle("50v50 Swarm Policy Training", fontsize=18, fontweight="bold")
    axes[0, 0].plot(x, [r["best_fitness"] for r in curve_rows], color="#2563eb", lw=2.4, label="best")
    axes[0, 0].plot(x, [r["generation_mean_fitness"] for r in curve_rows], color="#94a3b8", lw=1.8, label="population mean")
    axes[0, 0].set_title("Population fitness")
    axes[0, 0].legend()
    axes[0, 1].plot(x, [r["probe_yellow_win_rate"] for r in curve_rows], color="#eab308", lw=2.2, label="yellow")
    axes[0, 1].plot(x, [r["probe_blue_win_rate"] for r in curve_rows], color="#2563eb", lw=2.2, label="blue")
    axes[0, 1].plot(x, [r["probe_draw_rate"] for r in curve_rows], color="#64748b", lw=1.6, label="draw")
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_title("Probe self-play outcome")
    axes[0, 1].legend()
    axes[1, 0].plot(x, [r["probe_mean_robot_contacts"] for r in curve_rows], color="#dc2626", lw=2.2)
    axes[1, 0].set_title("Robot contacts per probe game")
    axes[1, 1].plot(x, [r["probe_mean_yellow_alive"] for r in curve_rows], color="#eab308", lw=2.2, label="yellow")
    axes[1, 1].plot(x, [r["probe_mean_blue_alive"] for r in curve_rows], color="#2563eb", lw=2.2, label="blue")
    axes[1, 1].set_title("Survivors per team")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(True, color="#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_scale_50v50_training.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    labels = ["Yellow win", "Blue win", "Draw", "Y alive", "B alive", "Contacts p95"]
    values = [
        summary["yellow_win_rate"] * 100,
        summary["blue_win_rate"] * 100,
        summary["draw_rate"] * 100,
        summary["mean_yellow_alive"],
        summary["mean_blue_alive"],
        summary["p95_robot_contacts"],
    ]
    colors = ["#eab308", "#2563eb", "#64748b", "#facc15", "#60a5fa", "#dc2626"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("50v50 Evaluation Summary", fontsize=18, fontweight="bold")
    ax.set_ylabel("Percent, count, or p95 event count")
    ax.grid(axis="y", color="#e2e8f0")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02, f"{value:.1f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_scale_50v50_eval.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    fig.suptitle("50v50 Rule-Scoring Closure and Replay Evidence", fontsize=18, fontweight="bold")
    boxes = [
        (4, 22, 18, 10, "Swarm-flow policy\nshared team actor\n50 vehicles/side", "#fef3c7", "#eab308"),
        (28, 22, 18, 10, "Rule simulation\nzones, LOS fire,\nshielded bases", "#dbeafe", "#2563eb"),
        (52, 22, 18, 10, "Scoring closure\nzone control -> shield\nbase damage -> win", "#dcfce7", "#16a34a"),
        (76, 22, 18, 10, "Selection gate\nwin balance,\ncontacts, damage", "#fee2e2", "#dc2626"),
        (16, 5, 24, 9, "256-game evaluation\nY win {0:.1f}% | B win {1:.1f}%\nbase damage {2:.1f}/{3:.1f}".format(
            summary["yellow_win_rate"] * 100,
            summary["blue_win_rate"] * 100,
            summary["mean_yellow_base_damage"],
            summary["mean_blue_base_damage"],
        ), "#f8fafc", "#475569"),
        (58, 5, 28, 9, "IsaacLab replay QA\n100 vehicle-shaped actors\n30 s MP4 + GIF + figures", "#f8fafc", "#475569"),
    ]
    for x0, y0, w, h, text, face, edge in boxes:
        rect = plt.Rectangle((x0, y0), w, h, facecolor=face, edgecolor=edge, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0f172a")
    for start, end in [((22, 27), (28, 27)), ((46, 27), (52, 27)), ((70, 27), (76, 27)), ((85, 22), (72, 14)), ((28, 22), (28, 14)), ((40, 9.5), (58, 9.5))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#334155"})
    ax.text(50, 36, "Promotion requires both tactical behavior and evidence artifacts, not reward-only training.", ha="center", fontsize=11, color="#334155")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_scale_50v50_rule_closure.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 50)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8fafc")
    layout_env = LargeScaleBattle50v50(config_from_args(args))
    for rect in layout_env.obstacles:
        xmin, ymin, xmax, ymax = rect
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color="#cbd5e1", ec="#334155"))
    for i, z in enumerate(layout_env.zones):
        ax.add_patch(plt.Circle(z, 6, fill=False, lw=3, color="#7c3aed"))
        ax.text(z[0], z[1], f"Zone {i+1}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.scatter([4.5], [25], s=800, marker="s", color="#eab308", edgecolor="#0f172a", label="Yellow base")
    ax.scatter([75.5], [25], s=800, marker="s", color="#2563eb", edgecolor="#0f172a", label="Blue base")
    ax.arrow(10, 25, 22, 0, width=0.3, head_width=2.0, color="#eab308", alpha=0.7)
    ax.arrow(70, 25, -22, 0, width=0.3, head_width=2.0, color="#2563eb", alpha=0.7)
    ax.set_title("50v50 Rule Layout: Three Control Zones + Shielded Base Assault", fontsize=18, fontweight="bold")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(loc="upper center", ncol=2)
    ax.grid(color="#e2e8f0")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "large_scale_50v50_rule_layout.png", bbox_inches="tight")
    plt.close(fig)


def write_report() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    eval_payload = json.loads((DATA_DIR / "eval_summary.json").read_text(encoding="utf-8"))
    train_summary = json.loads((DATA_DIR / "training_summary.json").read_text(encoding="utf-8"))
    s = eval_payload["summary"]
    report = f"""# Large-Scale 50v50 Multi-Agent Battle Report

This report documents the first formal large-scale rule-level extension for the repository. It is not a replacement for the two-robot IsaacLab result; it is a new 100-agent benchmark contract used to study scalable multi-agent decision making before expensive full-physics replay.

## Scenario

- Two teams: yellow and blue.
- Agents per team: 50.
- Arena: 80 m x 50 m.
- Objectives: capture three middle control zones, open the enemy base shield, eliminate opponents, then damage the enemy base.
- Obstacles: three static cover/barrier regions.
- Safety metrics: robot contacts, obstacle contacts, shielded base shots, survivors and base health.

## Training

- Algorithm: population-based swarm flow policy search.
- Generations: {train_summary['generations']}.
- Population: {train_summary['population']}.
- Candidate episodes: {train_summary['episodes_per_candidate']}.
- Total training episodes sampled: {train_summary['episodes_seen']}.
- Best fitness: {train_summary['best_fitness']:.4f}.
- Wall time: {train_summary['wall_time_s']:.2f} s.

## Evaluation

- Episodes: {s['episodes']}.
- Yellow win rate: {s['yellow_win_rate'] * 100:.2f}%.
- Blue win rate: {s['blue_win_rate'] * 100:.2f}%.
- Draw rate: {s['draw_rate'] * 100:.2f}%.
- Mean yellow score: {s['mean_yellow_score']:.2f}.
- Mean blue score: {s['mean_blue_score']:.2f}.
- Mean yellow survivors: {s['mean_yellow_alive']:.2f} / 50.
- Mean blue survivors: {s['mean_blue_alive']:.2f} / 50.
- Mean yellow base damage: {s['mean_yellow_base_damage']:.2f}.
- Mean blue base damage: {s['mean_blue_base_damage']:.2f}.
- Mean yellow base open rate: {s['mean_yellow_base_open_rate'] * 100:.2f}%.
- Mean blue base open rate: {s['mean_blue_base_open_rate'] * 100:.2f}%.
- Mean robot contacts: {s['mean_robot_contacts']:.2f}.
- P95 robot contacts: {s['p95_robot_contacts']:.2f}.
- Mean obstacle contacts: {s['mean_obstacle_contacts']:.2f}.
- Mean final zone state: {s['mean_final_zone_state']}.

## Artifacts

- Checkpoint: `docs/rl_data/large_scale_50v50/policy_checkpoint.json`
- Training curve: `docs/rl_data/large_scale_50v50/training_curve.csv`
- Evaluation JSON: `docs/rl_data/large_scale_50v50/eval_summary.json`
- Evaluation CSV: `docs/rl_data/large_scale_50v50/eval_episodes.csv`
- Rule-level preview MP4: `docs/media/large_scale_50v50_replay.mp4`
- Rule-level preview GIF: `docs/media/large_scale_50v50_replay.gif`
- IsaacLab tactical replay MP4: `docs/media/large_scale_50v50_isaaclab_replay.mp4`
- IsaacLab tactical replay GIF: `docs/media/large_scale_50v50_isaaclab_replay.gif`
- Figures: `docs/figures/large_scale_50v50/`

## Boundary

This benchmark validates scalable rule-level 50v50 mechanics and a trained swarm policy baseline. It does not claim IsaacLab rigid-body validation for all 100 robots and does not claim real-robot deployment. Those require a separate physics scaling and Sim2Real evidence package.
"""
    (ROOT / "docs" / "large_scale_50v50_report.md").write_text(report, encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    train_args = argparse.Namespace(**vars(args))
    ckpt = train(train_args)
    rule_kwargs = {
        "agents_per_team": args.agents_per_team,
        "base_hp": args.base_hp,
        "base_damage": args.base_damage,
        "blue_base_damage_multiplier": args.blue_base_damage_multiplier,
        "capture_rate": args.capture_rate,
        "shield_progress_to_open": args.shield_progress_to_open,
        "contact_radius": args.contact_radius,
        "separation_radius": args.separation_radius,
    }
    eval_args = argparse.Namespace(
        checkpoint=str(DATA_DIR / "policy_checkpoint.json"),
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        max_steps=args.max_steps,
        **rule_kwargs,
    )
    evaluate(eval_args)
    render_args = argparse.Namespace(
        checkpoint=str(DATA_DIR / "policy_checkpoint.json"),
        seed=args.render_seed,
        trace_stride=args.trace_stride,
        fps=args.fps,
        seconds=args.video_seconds,
        gif_seconds=args.gif_seconds,
        gif_fps=args.gif_fps,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        **rule_kwargs,
    )
    render_video(render_args)
    make_figures(args)
    write_report()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Large-scale 50v50 multi-agent battle training/evaluation/rendering")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_train_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=507050)
        p.add_argument("--agents-per-team", type=int, default=50)
        p.add_argument("--generations", type=int, default=80)
        p.add_argument("--population", type=int, default=16)
        p.add_argument("--episodes-per-candidate", type=int, default=2)
        p.add_argument("--probe-episodes", type=int, default=4)
        p.add_argument("--elite-frac", type=float, default=0.25)
        p.add_argument("--sigma", type=float, default=0.55)
        p.add_argument("--min-sigma", type=float, default=0.08)
        p.add_argument("--sigma-decay", type=float, default=0.985)
        p.add_argument("--archive-interval", type=int, default=4)
        p.add_argument("--archive-size", type=int, default=8)
        p.add_argument("--init-checkpoint", default="")
        p.add_argument("--log-interval", type=int, default=5)
        p.add_argument("--max-steps", type=int, default=420)
        p.add_argument("--base-hp", type=float, default=None)
        p.add_argument("--base-damage", type=float, default=None)
        p.add_argument("--blue-base-damage-multiplier", type=float, default=None)
        p.add_argument("--capture-rate", type=float, default=None)
        p.add_argument("--shield-progress-to-open", type=float, default=None)
        p.add_argument("--contact-radius", type=float, default=None)
        p.add_argument("--separation-radius", type=float, default=None)
        p.add_argument("--selection-episodes", type=int, default=24)
        p.add_argument("--verbose", action="store_true")

    p_train = sub.add_parser("train")
    add_train_flags(p_train)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", default=str(DATA_DIR / "policy_checkpoint.json"))
    p_eval.add_argument("--episodes", type=int, default=128)
    p_eval.add_argument("--seed", type=int, default=508000)
    p_eval.add_argument("--agents-per-team", type=int, default=50)
    p_eval.add_argument("--max-steps", type=int, default=420)
    p_eval.add_argument("--base-hp", type=float, default=None)
    p_eval.add_argument("--base-damage", type=float, default=None)
    p_eval.add_argument("--blue-base-damage-multiplier", type=float, default=None)
    p_eval.add_argument("--capture-rate", type=float, default=None)
    p_eval.add_argument("--shield-progress-to-open", type=float, default=None)
    p_eval.add_argument("--contact-radius", type=float, default=None)
    p_eval.add_argument("--separation-radius", type=float, default=None)
    p_render = sub.add_parser("render")
    p_render.add_argument("--checkpoint", default=str(DATA_DIR / "policy_checkpoint.json"))
    p_render.add_argument("--seed", type=int, default=509000)
    p_render.add_argument("--trace-stride", type=int, default=1)
    p_render.add_argument("--fps", type=int, default=30)
    p_render.add_argument("--seconds", type=float, default=30.0)
    p_render.add_argument("--gif-seconds", type=float, default=12.0)
    p_render.add_argument("--gif-fps", type=int, default=8)
    p_render.add_argument("--width", type=int, default=1920)
    p_render.add_argument("--height", type=int, default=1080)
    p_render.add_argument("--agents-per-team", type=int, default=50)
    p_render.add_argument("--max-steps", type=int, default=420)
    p_render.add_argument("--base-hp", type=float, default=None)
    p_render.add_argument("--base-damage", type=float, default=None)
    p_render.add_argument("--blue-base-damage-multiplier", type=float, default=None)
    p_render.add_argument("--capture-rate", type=float, default=None)
    p_render.add_argument("--shield-progress-to-open", type=float, default=None)
    p_render.add_argument("--contact-radius", type=float, default=None)
    p_render.add_argument("--separation-radius", type=float, default=None)
    p_fig = sub.add_parser("figures")
    p_fig.add_argument("--agents-per-team", type=int, default=50)
    p_fig.add_argument("--max-steps", type=int, default=420)
    p_fig.add_argument("--base-hp", type=float, default=None)
    p_fig.add_argument("--base-damage", type=float, default=None)
    p_fig.add_argument("--blue-base-damage-multiplier", type=float, default=None)
    p_fig.add_argument("--capture-rate", type=float, default=None)
    p_fig.add_argument("--shield-progress-to-open", type=float, default=None)
    p_fig.add_argument("--contact-radius", type=float, default=None)
    p_fig.add_argument("--separation-radius", type=float, default=None)
    sub.add_parser("report")
    p_all = sub.add_parser("all")
    add_train_flags(p_all)
    p_all.add_argument("--eval-episodes", type=int, default=128)
    p_all.add_argument("--eval-seed", type=int, default=508000)
    p_all.add_argument("--render-seed", type=int, default=509000)
    p_all.add_argument("--trace-stride", type=int, default=1)
    p_all.add_argument("--fps", type=int, default=30)
    p_all.add_argument("--video-seconds", type=float, default=30.0)
    p_all.add_argument("--gif-seconds", type=float, default=12.0)
    p_all.add_argument("--gif-fps", type=int, default=8)
    p_all.add_argument("--width", type=int, default=1920)
    p_all.add_argument("--height", type=int, default=1080)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "render":
        render_video(args)
    elif args.cmd == "figures":
        make_figures(args)
    elif args.cmd == "report":
        write_report()
    elif args.cmd == "all":
        run_all(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
