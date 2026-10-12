from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from analyze_phase31_falls import (
    actuator_names,
    foot_contacts,
    geom_id_by_name,
    joint_specs,
    normalize_obs,
    torso_metrics,
    vector_stats,
    write_csv,
    write_svg_plot,
)
from humanoid_ai.envs import load_vecnormalize_env, make_humanoid_env


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    model: Path
    vecnormalize: Path
    curriculum_config: Path
    target_velocity: float


DEFAULT_CONDITIONS = [
    Condition(
        key="A",
        label="current_stage1",
        model=ROOT / "checkpoints/phase32_A_current_100k/phase32_A_current_100k_seed4242_latest.zip",
        vecnormalize=ROOT / "checkpoints/phase32_A_current_100k/phase32_A_current_100k_seed4242_vecnormalize.pkl",
        curriculum_config=ROOT / "configs/phase31_stage1_A_current.json",
        target_velocity=0.25,
    ),
    Condition(
        key="B",
        label="fall_penalty",
        model=ROOT / "checkpoints/phase32_B_fall_penalty_100k/phase32_B_fall_penalty_100k_seed4242_latest.zip",
        vecnormalize=ROOT / "checkpoints/phase32_B_fall_penalty_100k/phase32_B_fall_penalty_100k_seed4242_vecnormalize.pkl",
        curriculum_config=ROOT / "configs/phase31_stage1_B_fall_penalty.json",
        target_velocity=0.25,
    ),
    Condition(
        key="C",
        label="slow_stability",
        model=ROOT / "checkpoints/phase32_C_slow_stability_100k/phase32_C_slow_stability_100k_seed4242_latest.zip",
        vecnormalize=ROOT / "checkpoints/phase32_C_slow_stability_100k/phase32_C_slow_stability_100k_seed4242_vecnormalize.pkl",
        curriculum_config=ROOT / "configs/phase31_stage1_C_slow_stability.json",
        target_velocity=0.18,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.3 gait and walking-quality analysis.")
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--curriculum-stage", default="stage1")
    parser.add_argument("--saturation-threshold", type=float, default=0.39)
    parser.add_argument("--slip-speed-threshold", type=float, default=0.35)
    parser.add_argument("--forward-distance-threshold", type=float, default=0.35)
    parser.add_argument("--lateral-distance-threshold", type=float, default=0.35)
    parser.add_argument("--upright-threshold", type=float, default=0.90)
    parser.add_argument("--out-dir", default="work/phase33_walk")
    parser.add_argument("--graph-dir", default="outputs/phase33_walk")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def support_phase(row: dict[str, Any]) -> str:
    left = bool(int(row.get("left_foot_contact", 0)))
    right = bool(int(row.get("right_foot_contact", 0)))
    if left and right:
        return "double"
    if left:
        return "left"
    if right:
        return "right"
    if bool(int(row.get("nonfoot_floor_contact", 0))):
        return "nonfoot"
    return "none"


def compressed_single_support(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    phases: list[tuple[str, int]] = []
    previous = ""
    for row in rows:
        phase = support_phase(row)
        if phase not in {"left", "right"}:
            continue
        if phase != previous:
            phases.append((phase, int(row["step"])))
            previous = phase
    return phases


def contact_switch_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    phases = compressed_single_support(rows)
    switch_count = sum(1 for index in range(1, len(phases)) if phases[index][0] != phases[index - 1][0])
    half_cycle_steps = [
        float(phases[index][1] - phases[index - 1][1])
        for index in range(1, len(phases))
        if phases[index][0] != phases[index - 1][0]
    ]
    full_cycle_steps = [
        float(phases[index][1] - phases[index - 2][1])
        for index in range(2, len(phases))
        if phases[index][0] == phases[index - 2][0] and phases[index][0] != phases[index - 1][0]
    ]
    return {
        "single_support_segments": float(len(phases)),
        "contact_switch_count": float(switch_count),
        "half_cycle_steps_mean": mean(half_cycle_steps),
        "walk_cycle_steps_mean": mean(full_cycle_steps),
    }


def geom_position(base_env, geom_name: str) -> np.ndarray:
    geom_id = geom_id_by_name(base_env.model, geom_name)
    if geom_id is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(base_env.data.geom_xpos[geom_id], dtype=np.float64).copy()


def saturation_fraction(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values) >= threshold))


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return 0.0
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def build_phase33_row(
    *,
    condition: Condition,
    episode: int,
    step: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    base_env,
    joints: list[dict[str, Any]],
    actuators: list[str],
    action: np.ndarray,
    previous: dict[str, np.ndarray] | None,
    saturation_threshold: float,
) -> dict[str, Any]:
    qpos = np.asarray(base_env.data.qpos, dtype=np.float64)
    qvel = np.asarray(base_env.data.qvel, dtype=np.float64)
    ctrl = np.asarray(base_env.data.ctrl, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    dt = float(getattr(base_env, "dt", 0.0) or 0.0)
    left_pos = geom_position(base_env, "left_foot")
    right_pos = geom_position(base_env, "right_foot")
    torso = torso_metrics(base_env)
    contacts = foot_contacts(base_env)

    if previous and dt > 0.0:
        left_velocity = (left_pos - previous["left_pos"]) / dt
        right_velocity = (right_pos - previous["right_pos"]) / dt
        torso_height_velocity = (torso["torso_height"] - float(previous["torso_height"])) / dt
        action_delta = action - previous["action"]
        qvel_delta = qvel - previous["qvel"]
        qvel_delta_rate = qvel_delta / dt
    else:
        left_velocity = np.zeros(3, dtype=np.float64)
        right_velocity = np.zeros(3, dtype=np.float64)
        torso_height_velocity = 0.0
        action_delta = action
        qvel_delta = np.zeros_like(qvel)
        qvel_delta_rate = np.zeros_like(qvel)

    left_slip_speed = float(np.linalg.norm(left_velocity[:2])) if contacts["left_foot_contact"] else 0.0
    right_slip_speed = float(np.linalg.norm(right_velocity[:2])) if contacts["right_foot_contact"] else 0.0
    row: dict[str, Any] = {
        "condition": condition.key,
        "condition_label": condition.label,
        "episode": episode,
        "step": step,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "x_velocity": float(info.get("x_velocity", 0.0)),
        "y_velocity": float(info.get("y_velocity", 0.0)),
        "target_velocity": condition.target_velocity,
        "target_velocity_error": float(info.get("x_velocity", 0.0)) - condition.target_velocity,
        "reward_survive": float(info.get("reward_survive", 0.0)),
        "reward_ctrl": float(info.get("reward_ctrl", 0.0)),
        "reward_contact": float(info.get("reward_contact", 0.0)),
        "phase3_reward_total": float(info.get("phase3_reward_total", 0.0)),
        "phase3_reward_velocity_tracking": float(info.get("phase3_reward_velocity_tracking", 0.0)),
        "phase3_penalty_lateral_velocity": float(info.get("phase3_penalty_lateral_velocity", 0.0)),
        "phase3_penalty_torso_tilt": float(info.get("phase3_penalty_torso_tilt", 0.0)),
        "phase3_penalty_torso_height": float(info.get("phase3_penalty_torso_height", 0.0)),
        "phase3_penalty_action_smoothness": float(info.get("phase3_penalty_action_smoothness", 0.0)),
        "phase3_action_delta_sq_mean": float(info.get("phase3_action_delta_sq_mean", rms(action_delta) ** 2)),
        "phase3_action_delta_rms": float(info.get("phase3_action_delta_rms", rms(action_delta))),
        "phase3_penalty_fall": float(info.get("phase3_penalty_fall", 0.0)),
        **torso,
        "torso_height_velocity": torso_height_velocity,
        **contacts,
        "support_phase": support_phase(contacts),
        "left_foot_x": float(left_pos[0]),
        "left_foot_y": float(left_pos[1]),
        "left_foot_z": float(left_pos[2]),
        "right_foot_x": float(right_pos[0]),
        "right_foot_y": float(right_pos[1]),
        "right_foot_z": float(right_pos[2]),
        "left_foot_speed_xy": float(np.linalg.norm(left_velocity[:2])),
        "right_foot_speed_xy": float(np.linalg.norm(right_velocity[:2])),
        "left_foot_slip_speed": left_slip_speed,
        "right_foot_slip_speed": right_slip_speed,
        "foot_slip_speed_max": max(left_slip_speed, right_slip_speed),
        "action_delta_rms": rms(action_delta),
        "action_delta_abs_max": float(np.max(np.abs(action_delta))) if action_delta.size else 0.0,
        "qvel_delta_rms": rms(qvel_delta),
        "qvel_delta_abs_max": float(np.max(np.abs(qvel_delta))) if qvel_delta.size else 0.0,
        "qvel_delta_rate_rms": rms(qvel_delta_rate),
        "qvel_delta_rate_abs_max": float(np.max(np.abs(qvel_delta_rate))) if qvel_delta_rate.size else 0.0,
        "action_saturation_fraction": saturation_fraction(action, saturation_threshold),
        "ctrl_saturation_fraction": saturation_fraction(ctrl, saturation_threshold),
        **vector_stats("qvel", qvel),
        **vector_stats("action", action),
        **vector_stats("ctrl", ctrl),
    }
    for spec in joints:
        row[f"joint_angle_{spec['name']}"] = float(qpos[spec["qposadr"]])
        row[f"joint_velocity_{spec['name']}"] = float(qvel[spec["dofadr"]])
    for index, name in enumerate(actuators):
        row[f"ctrl_{name}"] = float(ctrl[index]) if index < len(ctrl) else 0.0
        row[f"action_{name}"] = float(action[index]) if index < len(action) else 0.0
    return row


def summarize_episode(
    condition: Condition,
    rows: list[dict[str, Any]],
    *,
    slip_speed_threshold: float,
    forward_distance_threshold: float,
    lateral_distance_threshold: float,
    upright_threshold: float,
) -> dict[str, Any]:
    first = rows[0]
    last = rows[-1]
    switches = contact_switch_metrics(rows)
    left_slips = [safe_float(row["left_foot_slip_speed"]) for row in rows if int(row["left_foot_contact"])]
    right_slips = [safe_float(row["right_foot_slip_speed"]) for row in rows if int(row["right_foot_contact"])]
    all_slips = left_slips + right_slips
    forward_distance = safe_float(last["x_position"]) - safe_float(first["x_position"])
    lateral_displacement = safe_float(last["y_position"]) - safe_float(first["y_position"])
    lateral_abs = abs(lateral_displacement)
    walk_candidate = (
        forward_distance >= forward_distance_threshold
        and switches["contact_switch_count"] >= 2.0
        and mean(all_slips) <= slip_speed_threshold
        and lateral_abs <= lateral_distance_threshold
        and mean([safe_float(row["torso_upright"]) for row in rows]) >= upright_threshold
    )
    stable_walk_success = walk_candidate and not bool(last["terminated"]) and len(rows) >= 150
    return {
        "condition": condition.key,
        "condition_label": condition.label,
        "episode": int(first["episode"]),
        "length": len(rows),
        "terminated": bool(last["terminated"]),
        "truncated": bool(last["truncated"]),
        "reward_sum": sum(safe_float(row["reward"]) for row in rows),
        "forward_distance": forward_distance,
        "forward_velocity_mean": mean([safe_float(row["x_velocity"]) for row in rows]),
        "lateral_abs_displacement": lateral_abs,
        "torso_height_mean": mean([safe_float(row["torso_height"]) for row in rows]),
        "torso_height_std": float(np.std([safe_float(row["torso_height"]) for row in rows])),
        "torso_height_drop_last20": safe_float(rows[-min(20, len(rows))]["torso_height"]) - safe_float(last["torso_height"]),
        "torso_upright_mean": mean([safe_float(row["torso_upright"]) for row in rows]),
        "torso_upright_min": min(safe_float(row["torso_upright"]) for row in rows),
        "left_contact_rate": mean([safe_float(row["left_foot_contact"]) for row in rows]),
        "right_contact_rate": mean([safe_float(row["right_foot_contact"]) for row in rows]),
        "double_contact_rate": mean([safe_float(row["double_foot_contact"]) for row in rows]),
        "nonfoot_floor_contact_rate": mean([safe_float(row["nonfoot_floor_contact"]) for row in rows]),
        "left_slip_speed_mean": mean(left_slips),
        "right_slip_speed_mean": mean(right_slips),
        "foot_slip_speed_mean": mean(all_slips),
        "foot_slip_speed_max": max(all_slips) if all_slips else 0.0,
        "action_delta_rms_mean": mean([safe_float(row.get("action_delta_rms")) for row in rows]),
        "action_delta_abs_max": max(safe_float(row.get("action_delta_abs_max")) for row in rows),
        "qvel_delta_rms_mean": mean([safe_float(row.get("qvel_delta_rms")) for row in rows]),
        "qvel_delta_abs_max": max(safe_float(row.get("qvel_delta_abs_max")) for row in rows),
        "qvel_delta_rate_rms_mean": mean([safe_float(row.get("qvel_delta_rate_rms")) for row in rows]),
        "qvel_delta_rate_abs_max": max(safe_float(row.get("qvel_delta_rate_abs_max")) for row in rows),
        "slip_action_delta_correlation": pearson_correlation(
            [safe_float(row["foot_slip_speed_max"]) for row in rows],
            [safe_float(row.get("action_delta_rms")) for row in rows],
        ),
        "action_saturation_fraction_mean": mean([safe_float(row["action_saturation_fraction"]) for row in rows]),
        "ctrl_saturation_fraction_mean": mean([safe_float(row["ctrl_saturation_fraction"]) for row in rows]),
        "qvel_abs_max": max(safe_float(row["qvel_abs_max"]) for row in rows),
        "walk_candidate": walk_candidate,
        "stable_walk_success": stable_walk_success,
        **switches,
    }


def aggregate_condition(rows: list[dict[str, Any]], episode_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "condition": episode_summaries[0]["condition"],
        "condition_label": episode_summaries[0]["condition_label"],
        "episodes": len(episode_summaries),
        "survival_steps_mean": mean([safe_float(row["length"]) for row in episode_summaries]),
        "fall_rate": mean([1.0 if row["terminated"] else 0.0 for row in episode_summaries]),
        "truncated_rate": mean([1.0 if row["truncated"] else 0.0 for row in episode_summaries]),
        "walk_candidate_rate": mean([1.0 if row["walk_candidate"] else 0.0 for row in episode_summaries]),
        "stable_walk_success_rate": mean([1.0 if row["stable_walk_success"] else 0.0 for row in episode_summaries]),
        "forward_distance_mean": mean([safe_float(row["forward_distance"]) for row in episode_summaries]),
        "forward_velocity_mean": mean([safe_float(row["forward_velocity_mean"]) for row in episode_summaries]),
        "lateral_abs_displacement_mean": mean([safe_float(row["lateral_abs_displacement"]) for row in episode_summaries]),
        "torso_height_mean": mean([safe_float(row["torso_height_mean"]) for row in episode_summaries]),
        "torso_height_std_mean": mean([safe_float(row["torso_height_std"]) for row in episode_summaries]),
        "torso_height_drop_last20_mean": mean([safe_float(row["torso_height_drop_last20"]) for row in episode_summaries]),
        "torso_upright_mean": mean([safe_float(row["torso_upright_mean"]) for row in episode_summaries]),
        "torso_upright_min_mean": mean([safe_float(row["torso_upright_min"]) for row in episode_summaries]),
        "contact_switch_count_mean": mean([safe_float(row["contact_switch_count"]) for row in episode_summaries]),
        "walk_cycle_steps_mean": mean([safe_float(row["walk_cycle_steps_mean"]) for row in episode_summaries]),
        "left_contact_rate": mean([safe_float(row["left_foot_contact"]) for row in rows]),
        "right_contact_rate": mean([safe_float(row["right_foot_contact"]) for row in rows]),
        "double_contact_rate": mean([safe_float(row["double_foot_contact"]) for row in rows]),
        "nonfoot_floor_contact_rate": mean([safe_float(row["nonfoot_floor_contact"]) for row in rows]),
        "foot_slip_speed_mean": mean([safe_float(row["foot_slip_speed_mean"]) for row in episode_summaries]),
        "foot_slip_speed_max_mean": mean([safe_float(row["foot_slip_speed_max"]) for row in episode_summaries]),
        "action_delta_rms_mean": mean([safe_float(row["action_delta_rms_mean"]) for row in episode_summaries]),
        "action_delta_abs_max_mean": mean([safe_float(row["action_delta_abs_max"]) for row in episode_summaries]),
        "qvel_delta_rms_mean": mean([safe_float(row["qvel_delta_rms_mean"]) for row in episode_summaries]),
        "qvel_delta_abs_max_mean": mean([safe_float(row["qvel_delta_abs_max"]) for row in episode_summaries]),
        "qvel_delta_rate_rms_mean": mean([safe_float(row["qvel_delta_rate_rms_mean"]) for row in episode_summaries]),
        "qvel_delta_rate_abs_max_mean": mean([safe_float(row["qvel_delta_rate_abs_max"]) for row in episode_summaries]),
        "slip_action_delta_correlation_mean": mean(
            [safe_float(row["slip_action_delta_correlation"]) for row in episode_summaries]
        ),
        "action_saturation_fraction_mean": mean([safe_float(row["action_saturation_fraction_mean"]) for row in episode_summaries]),
        "ctrl_saturation_fraction_mean": mean([safe_float(row["ctrl_saturation_fraction_mean"]) for row in episode_summaries]),
        "qvel_abs_max_mean": mean([safe_float(row["qvel_abs_max"]) for row in episode_summaries]),
    }


def evaluate_condition(
    condition: Condition,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = PPO.load(condition.model, device=args.device)
    vecnorm_env = load_vecnormalize_env(
        condition.vecnormalize,
        args.env_id,
        seed=args.seed,
        training=False,
        norm_reward=False,
        curriculum_config_path=condition.curriculum_config,
        curriculum_stage=args.curriculum_stage,
    )
    env = make_humanoid_env(
        args.env_id,
        seed=args.seed,
        render_mode=None,
        curriculum_config_path=condition.curriculum_config,
        curriculum_stage=args.curriculum_stage,
    )

    rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    try:
        base_env = env.unwrapped
        joints = joint_specs(base_env.model)
        actuators = actuator_names(base_env.model)
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed + episode - 1)
            env.action_space.seed(args.seed + episode - 1)
            episode_rows: list[dict[str, Any]] = []
            previous: dict[str, np.ndarray] | None = None
            for step in range(1, args.max_steps + 1):
                normalized_obs = normalize_obs(vecnorm_env, obs)
                action, _ = model.predict(normalized_obs, deterministic=True)
                action_array = np.asarray(action, dtype=np.float64).reshape(-1)
                obs, reward, terminated, truncated, info = env.step(action_array)
                row = build_phase33_row(
                    condition=condition,
                    episode=episode,
                    step=step,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                    base_env=base_env,
                    joints=joints,
                    actuators=actuators,
                    action=action_array,
                    previous=previous,
                    saturation_threshold=args.saturation_threshold,
                )
                previous = {
                    "left_pos": np.asarray([row["left_foot_x"], row["left_foot_y"], row["left_foot_z"]], dtype=np.float64),
                    "right_pos": np.asarray([row["right_foot_x"], row["right_foot_y"], row["right_foot_z"]], dtype=np.float64),
                    "torso_height": np.asarray(row["torso_height"], dtype=np.float64),
                    "action": action_array.copy(),
                    "qvel": np.asarray(base_env.data.qvel, dtype=np.float64).copy(),
                }
                rows.append(row)
                episode_rows.append(row)
                if terminated or truncated:
                    break
            episode_summaries.append(
                summarize_episode(
                    condition,
                    episode_rows,
                    slip_speed_threshold=args.slip_speed_threshold,
                    forward_distance_threshold=args.forward_distance_threshold,
                    lateral_distance_threshold=args.lateral_distance_threshold,
                    upright_threshold=args.upright_threshold,
                )
            )
    finally:
        env.close()
        vecnorm_env.close()
    return rows, episode_summaries, aggregate_condition(rows, episode_summaries)


def write_contact_svg(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first_episode = int(rows[0]["episode"])
    rows = [row for row in rows if int(row["episode"]) == first_episode]
    width = 980
    height = 520
    left = 70
    right = 35
    top = 45
    bottom = 60
    max_step = max(int(row["step"]) for row in rows)
    max_x = max(safe_float(row["x_position"]) for row in rows)
    min_x = min(safe_float(row["x_position"]) for row in rows)
    min_height = min(safe_float(row["torso_height"]) for row in rows)
    max_height = max(safe_float(row["torso_height"]) for row in rows)
    if math.isclose(min_height, max_height):
        max_height += 1.0
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{title} episode {first_episode}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" />',
        f'<text x="{width / 2:.0f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">step</text>',
        f'<text x="20" y="135" transform="rotate(-90 20 135)" text-anchor="middle" font-family="Arial" font-size="13">contact</text>',
        f'<text x="20" y="330" transform="rotate(-90 20 330)" text-anchor="middle" font-family="Arial" font-size="13">height / x</text>',
    ]

    def sx(step: float) -> float:
        return left + (step - 1.0) / max(1.0, max_step - 1.0) * (width - left - right)

    def sy_contact(value: float, lane: float) -> float:
        return top + lane + (1.0 - value) * 34.0

    def sy_value(value: float, min_value: float, max_value: float, base: float, span: float) -> float:
        if math.isclose(min_value, max_value):
            return base + span / 2.0
        return base + (1.0 - (value - min_value) / (max_value - min_value)) * span

    for key, color, lane, label in [
        ("left_foot_contact", "#2563eb", 20.0, "left contact"),
        ("right_foot_contact", "#dc2626", 75.0, "right contact"),
        ("double_foot_contact", "#16a34a", 130.0, "double contact"),
    ]:
        points = " ".join(f'{sx(int(row["step"])):.2f},{sy_contact(safe_float(row[key]), lane):.2f}' for row in rows)
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" />')
        body.append(f'<text x="{width-right-150}" y="{top+lane+8}" font-family="Arial" font-size="12" fill="{color}">{label}</text>')

    height_points = " ".join(
        f'{sx(int(row["step"])):.2f},{sy_value(safe_float(row["torso_height"]), min_height, max_height, 255.0, 120.0):.2f}'
        for row in rows
    )
    x_points = " ".join(
        f'{sx(int(row["step"])):.2f},{sy_value(safe_float(row["x_position"]), min_x, max_x, 385.0, 80.0):.2f}'
        for row in rows
    )
    body.append(f'<polyline points="{height_points}" fill="none" stroke="#7c3aed" stroke-width="2" />')
    body.append(f'<text x="{width-right-150}" y="270" font-family="Arial" font-size="12" fill="#7c3aed">torso height</text>')
    body.append(f'<polyline points="{x_points}" fill="none" stroke="#ea580c" stroke-width="2" />')
    body.append(f'<text x="{width-right-150}" y="405" font-family="Arial" font-size="12" fill="#ea580c">x position</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    graph_dir = ROOT / args.graph_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    all_episode_summaries: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    graph_paths: list[str] = []
    for condition in DEFAULT_CONDITIONS:
        rows, episode_summaries, aggregate = evaluate_condition(condition, args)
        write_csv(out_dir / f"phase33_{condition.key}_step_log.csv", rows)
        write_csv(out_dir / f"phase33_{condition.key}_episode_summary.csv", episode_summaries)
        contact_svg = graph_dir / f"phase33_{condition.key}_contact_timeline.svg"
        height_svg = graph_dir / f"phase33_{condition.key}_height.svg"
        trajectory_svg = graph_dir / f"phase33_{condition.key}_trajectory.svg"
        write_contact_svg(contact_svg, rows, f"Phase 3.3 {condition.key} contact and motion")
        write_svg_plot(
            height_svg,
            {int(summary["episode"]): [row for row in rows if int(row["episode"]) == int(summary["episode"])] for summary in episode_summaries},
            title=f"Phase 3.3 {condition.key} torso height",
            x_key="step",
            y_key="torso_height",
            x_label="step",
            y_label="torso height",
        )
        write_svg_plot(
            trajectory_svg,
            {int(summary["episode"]): [row for row in rows if int(row["episode"]) == int(summary["episode"])] for summary in episode_summaries},
            title=f"Phase 3.3 {condition.key} trajectories",
            x_key="x_position",
            y_key="y_position",
            x_label="x position",
            y_label="y position",
        )
        graph_paths.extend([str(contact_svg), str(height_svg), str(trajectory_svg)])
        all_episode_summaries.extend(episode_summaries)
        aggregate_rows.append(aggregate)

    write_csv(out_dir / "phase33_episode_summary_all.csv", all_episode_summaries)
    write_csv(out_dir / "phase33_abc_walk_summary.csv", aggregate_rows)
    result = {
        "conditions": [condition.key for condition in DEFAULT_CONDITIONS],
        "episodes": args.episodes,
        "seed_start": args.seed,
        "max_steps": args.max_steps,
        "thresholds": {
            "saturation_threshold": args.saturation_threshold,
            "slip_speed_threshold": args.slip_speed_threshold,
            "forward_distance_threshold": args.forward_distance_threshold,
            "lateral_distance_threshold": args.lateral_distance_threshold,
            "upright_threshold": args.upright_threshold,
        },
        "summary_csv": str(out_dir / "phase33_abc_walk_summary.csv"),
        "episode_summary_csv": str(out_dir / "phase33_episode_summary_all.csv"),
        "graphs": graph_paths,
        "summary": aggregate_rows,
        "interpretation_notes": [
            "Contact switching alone is not treated as walking.",
            "walk_candidate requires forward distance, contact switches, low foot slip, limited lateral drift, and upright torso simultaneously.",
            "A high contact switch count with high slip or low forward distance is treated as unstable stepping or shuffling, not successful walking.",
        ],
    }
    json_path = out_dir / "phase33_walk_analysis.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("phase33_walk_analysis_done=true")


if __name__ == "__main__":
    main()
