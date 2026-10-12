from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from humanoid_ai.envs import load_vecnormalize_env, make_humanoid_env


ROOT = Path(__file__).resolve().parent
JOINT_START_INDEX = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect detailed Phase 3.1 fall diagnostics.")
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vec-normalize-stats", required=True)
    parser.add_argument("--curriculum-config", required=True)
    parser.add_argument("--curriculum-stage", default="stage1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--window-steps", type=int, default=20)
    parser.add_argument("--step-csv", default="work/phase31_stage1_100k_step_log.csv")
    parser.add_argument("--fall-window-csv", default="work/phase31_stage1_100k_fall_windows.csv")
    parser.add_argument("--analysis-json", default="work/phase31_stage1_100k_fall_analysis.json")
    parser.add_argument("--compare-csv", default="work/phase31_standard_vs_stage1_physics.csv")
    parser.add_argument("--standard-eval-csv", default="work/phase28_standard_100k_eval.csv")
    parser.add_argument("--stage-eval-csv", default="work/phase3_stage1_100k_eval.csv")
    parser.add_argument("--trajectory-svg", default="outputs/phase31_stage1_trajectories.svg")
    parser.add_argument("--height-svg", default="outputs/phase31_stage1_torso_height.svg")
    return parser.parse_args()


def safe_name(name: str | None, fallback: str) -> str:
    if name:
        return name.replace(" ", "_")
    return fallback


def joint_specs(model) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for joint_id in range(JOINT_START_INDEX, model.njnt):
        joint = model.joint(joint_id)
        specs.append(
            {
                "name": safe_name(joint.name, f"joint_{joint_id}"),
                "qposadr": int(model.jnt_qposadr[joint_id]),
                "dofadr": int(model.jnt_dofadr[joint_id]),
            }
        )
    return specs


def actuator_names(model) -> list[str]:
    return [safe_name(model.actuator(i).name, f"actuator_{i}") for i in range(model.nu)]


def geom_id_by_name(model, name: str) -> int | None:
    for geom_id in range(model.ngeom):
        if model.geom(geom_id).name == name:
            return geom_id
    return None


def foot_contacts(base_env) -> dict[str, int | str]:
    model = base_env.model
    data = base_env.data
    floor_id = geom_id_by_name(model, "floor")
    right_foot_id = geom_id_by_name(model, "right_foot")
    left_foot_id = geom_id_by_name(model, "left_foot")
    right = 0
    left = 0
    nonfoot_floor = 0

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id is None or floor_id not in geom_pair:
            continue
        if right_foot_id is not None and right_foot_id in geom_pair:
            right = 1
        elif left_foot_id is not None and left_foot_id in geom_pair:
            left = 1
        else:
            nonfoot_floor = 1

    if left and right:
        phase = "double"
    elif left:
        phase = "left"
    elif right:
        phase = "right"
    elif nonfoot_floor:
        phase = "nonfoot"
    else:
        phase = "none"
    return {
        "left_foot_contact": left,
        "right_foot_contact": right,
        "double_foot_contact": int(left and right),
        "nonfoot_floor_contact": nonfoot_floor,
        "foot_contact_phase": phase,
    }


def torso_metrics(base_env) -> dict[str, float]:
    torso_id = base_env.model.body("torso").id
    xmat = base_env.data.xmat[torso_id]
    upright = float(xmat[8])
    clipped = max(-1.0, min(1.0, upright))
    return {
        "x_position": float(base_env.data.qpos[0]),
        "y_position": float(base_env.data.qpos[1]),
        "torso_height": float(base_env.data.qpos[2]),
        "torso_upright": upright,
        "torso_tilt_deg": math.degrees(math.acos(clipped)),
    }


def vector_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    abs_values = np.abs(values)
    return {
        f"{prefix}_abs_mean": float(np.mean(abs_values)) if abs_values.size else 0.0,
        f"{prefix}_abs_max": float(np.max(abs_values)) if abs_values.size else 0.0,
        f"{prefix}_l2": float(np.linalg.norm(values)) if values.size else 0.0,
    }


def build_step_row(
    *,
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
) -> dict[str, Any]:
    qpos = np.asarray(base_env.data.qpos, dtype=np.float64)
    qvel = np.asarray(base_env.data.qvel, dtype=np.float64)
    ctrl = np.asarray(base_env.data.ctrl, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    row: dict[str, Any] = {
        "episode": episode,
        "step": step,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "x_velocity": float(info.get("x_velocity", 0.0)),
        "y_velocity": float(info.get("y_velocity", 0.0)),
        "reward_survive": float(info.get("reward_survive", 0.0)),
        "reward_ctrl": float(info.get("reward_ctrl", 0.0)),
        "reward_contact": float(info.get("reward_contact", 0.0)),
        "phase3_reward_total": float(info.get("phase3_reward_total", 0.0)),
        "phase3_reward_velocity_tracking": float(info.get("phase3_reward_velocity_tracking", 0.0)),
        "phase3_penalty_lateral_velocity": float(info.get("phase3_penalty_lateral_velocity", 0.0)),
        "phase3_penalty_torso_tilt": float(info.get("phase3_penalty_torso_tilt", 0.0)),
        "phase3_penalty_torso_height": float(info.get("phase3_penalty_torso_height", 0.0)),
        "phase3_penalty_fall": float(info.get("phase3_penalty_fall", 0.0)),
        **torso_metrics(base_env),
        **foot_contacts(base_env),
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


def normalize_obs(vecnorm_env, obs: np.ndarray) -> np.ndarray:
    normalized = vecnorm_env.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :])
    return normalized


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_falls(rows_by_episode: dict[int, list[dict[str, Any]]], window_steps: int) -> dict[str, Any]:
    episode_summaries: list[dict[str, Any]] = []
    for episode, rows in rows_by_episode.items():
        window = rows[-window_steps:]
        first = window[0]
        last = window[-1]
        episode_summaries.append(
            {
                "episode": episode,
                "length": len(rows),
                "final_torso_height": float(last["torso_height"]),
                "height_drop_last_window": float(first["torso_height"]) - float(last["torso_height"]),
                "min_torso_height_last_window": min(float(row["torso_height"]) for row in window),
                "final_torso_upright": float(last["torso_upright"]),
                "upright_drop_last_window": float(first["torso_upright"]) - float(last["torso_upright"]),
                "min_torso_upright_last_window": min(float(row["torso_upright"]) for row in window),
                "max_tilt_deg_last_window": max(float(row["torso_tilt_deg"]) for row in window),
                "mean_x_velocity_last_window": mean([float(row["x_velocity"]) for row in window]),
                "mean_y_velocity_last_window": mean([float(row["y_velocity"]) for row in window]),
                "max_abs_y_velocity_last_window": max(abs(float(row["y_velocity"])) for row in window),
                "max_qvel_abs_last_window": max(float(row["qvel_abs_max"]) for row in window),
                "max_action_abs_last_window": max(float(row["action_abs_max"]) for row in window),
                "left_foot_contact_fraction_last_window": mean([float(row["left_foot_contact"]) for row in window]),
                "right_foot_contact_fraction_last_window": mean([float(row["right_foot_contact"]) for row in window]),
                "double_foot_contact_fraction_last_window": mean([float(row["double_foot_contact"]) for row in window]),
                "nonfoot_floor_contact_fraction_last_window": mean([float(row["nonfoot_floor_contact"]) for row in window]),
                "final_contact_phase": str(last["foot_contact_phase"]),
            }
        )

    aggregate_keys = [
        key for key in episode_summaries[0].keys() if key not in ("episode", "final_contact_phase")
    ]
    aggregate = {
        key: mean([float(row[key]) for row in episode_summaries])
        for key in aggregate_keys
    }
    return {
        "episode_summaries": episode_summaries,
        "aggregate": aggregate,
        "facts": [
            "All values are computed from measured step logs, not inferred from video.",
            "The fall window is the last N recorded steps of each episode.",
        ],
        "hypotheses": [
            "If final torso height and upright stay high while termination occurs, the environment may be ending on brief height threshold crossings or fast contact events rather than slow posture collapse.",
            "Large y velocity or lateral penalties in the fall window suggest sideways drift contributes to instability.",
            "Large qvel/action spikes suggest the policy may be over-correcting near the end of the episode.",
        ],
    }


def read_eval_summary(path: Path, label: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics = {
        "label": label,
        "episodes": len(rows),
        "survival_steps_mean": mean([float(row["survival_steps"]) for row in rows]),
        "fall_rate": mean([1.0 if row["fell"].lower() == "true" else 0.0 for row in rows]),
        "truncated_rate": mean([1.0 if row["truncated"].lower() == "true" else 0.0 for row in rows]),
        "forward_distance_mean": mean([float(row["forward_distance"]) for row in rows]),
        "forward_velocity_mean": mean([float(row["x_velocity_per_step"]) for row in rows]),
        "lateral_abs_displacement_mean": mean([float(row["lateral_abs_displacement"]) for row in rows]),
        "torso_upright_mean": mean([float(row["torso_upright_mean"]) for row in rows]),
        "torso_height_mean": mean([float(row["torso_height_mean"]) for row in rows]),
        "target_velocity_fraction_mean": mean([float(row["target_velocity_fraction"]) for row in rows]),
    }
    return metrics


def write_comparison(path: Path, standard_csv: Path, stage_csv: Path) -> list[dict[str, Any]]:
    rows = [
        read_eval_summary(standard_csv, "standard_reward_100k"),
        read_eval_summary(stage_csv, "phase3_stage1_100k"),
    ]
    write_csv(path, rows)
    return rows


def svg_polyline(points: list[tuple[float, float]], color: str, width: float = 1.2) -> str:
    joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="{width}" />'


def scale(value: float, min_value: float, max_value: float, out_min: float, out_max: float) -> float:
    if math.isclose(max_value, min_value):
        return (out_min + out_max) / 2.0
    ratio = (value - min_value) / (max_value - min_value)
    return out_min + ratio * (out_max - out_min)


def write_svg_plot(
    path: Path,
    rows_by_episode: dict[int, list[dict[str, Any]]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 960
    height = 560
    left = 70
    right = 30
    top = 45
    bottom = 65
    colors = [
        "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
        "#0891b2", "#4f46e5", "#be123c", "#15803d", "#a16207",
    ]
    all_x = [float(row[x_key]) for rows in rows_by_episode.values() for row in rows]
    all_y = [float(row[y_key]) for rows in rows_by_episode.values() for row in rows]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" />',
        f'<text x="{width / 2:.0f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>',
        f'<text x="18" y="{height / 2:.0f}" transform="rotate(-90 18 {height / 2:.0f})" text-anchor="middle" font-family="Arial" font-size="14">{y_label}</text>',
        f'<text x="{left}" y="{height-bottom+20}" text-anchor="middle" font-family="Arial" font-size="11">{min_x:.2f}</text>',
        f'<text x="{width-right}" y="{height-bottom+20}" text-anchor="middle" font-family="Arial" font-size="11">{max_x:.2f}</text>',
        f'<text x="{left-10}" y="{height-bottom}" text-anchor="end" font-family="Arial" font-size="11">{min_y:.2f}</text>',
        f'<text x="{left-10}" y="{top+4}" text-anchor="end" font-family="Arial" font-size="11">{max_y:.2f}</text>',
    ]
    for idx, (episode, rows) in enumerate(rows_by_episode.items()):
        points = [
            (
                scale(float(row[x_key]), min_x, max_x, left, width - right),
                scale(float(row[y_key]), min_y, max_y, height - bottom, top),
            )
            for row in rows
        ]
        body.append(svg_polyline(points, colors[idx % len(colors)]))
        if idx < 10:
            body.append(
                f'<text x="{width-right-110}" y="{top + 16 + idx * 16}" font-family="Arial" font-size="11" fill="{colors[idx % len(colors)]}">ep {episode}</text>'
            )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model, device=args.device)
    vecnorm_env = load_vecnormalize_env(
        args.vec_normalize_stats,
        args.env_id,
        seed=args.seed,
        training=False,
        norm_reward=False,
        curriculum_config_path=args.curriculum_config,
        curriculum_stage=args.curriculum_stage,
    )
    env = make_humanoid_env(
        args.env_id,
        seed=args.seed,
        render_mode=None,
        curriculum_config_path=args.curriculum_config,
        curriculum_stage=args.curriculum_stage,
    )

    rows: list[dict[str, Any]] = []
    rows_by_episode: dict[int, list[dict[str, Any]]] = {}
    fall_window_rows: list[dict[str, Any]] = []

    try:
        base_env = env.unwrapped
        joints = joint_specs(base_env.model)
        actuators = actuator_names(base_env.model)

        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed + episode - 1)
            env.action_space.seed(args.seed + episode - 1)
            episode_rows: list[dict[str, Any]] = []
            recent: deque[dict[str, Any]] = deque(maxlen=args.window_steps)

            for step in range(1, args.max_steps + 1):
                normalized_obs = normalize_obs(vecnorm_env, obs)
                action, _ = model.predict(normalized_obs, deterministic=True)
                action_array = np.asarray(action, dtype=np.float64).reshape(-1)
                obs, reward, terminated, truncated, info = env.step(action_array)
                row = build_step_row(
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
                )
                rows.append(row)
                episode_rows.append(row)
                recent.append(row)
                if terminated or truncated:
                    break
            rows_by_episode[episode] = episode_rows
            fall_window_rows.extend(recent)
    finally:
        env.close()
        vecnorm_env.close()

    step_csv = ROOT / args.step_csv
    fall_window_csv = ROOT / args.fall_window_csv
    analysis_json = ROOT / args.analysis_json
    compare_csv = ROOT / args.compare_csv
    trajectory_svg = ROOT / args.trajectory_svg
    height_svg = ROOT / args.height_svg

    write_csv(step_csv, rows)
    write_csv(fall_window_csv, fall_window_rows)
    fall_summary = summarize_falls(rows_by_episode, args.window_steps)
    comparison_rows = write_comparison(
        compare_csv,
        ROOT / args.standard_eval_csv,
        ROOT / args.stage_eval_csv,
    )
    write_svg_plot(
        trajectory_svg,
        rows_by_episode,
        title="Phase 3 Stage 1 100k trajectories",
        x_key="x_position",
        y_key="y_position",
        x_label="x position",
        y_label="y position",
    )
    write_svg_plot(
        height_svg,
        rows_by_episode,
        title="Phase 3 Stage 1 100k torso height",
        x_key="step",
        y_key="torso_height",
        x_label="step",
        y_label="torso height",
    )

    result = {
        "step_csv": str(step_csv),
        "fall_window_csv": str(fall_window_csv),
        "analysis_json": str(analysis_json),
        "comparison_csv": str(compare_csv),
        "trajectory_svg": str(trajectory_svg),
        "height_svg": str(height_svg),
        "fall_summary": fall_summary,
        "comparison": comparison_rows,
    }
    analysis_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("phase31_fall_analysis_done=true")


if __name__ == "__main__":
    main()
