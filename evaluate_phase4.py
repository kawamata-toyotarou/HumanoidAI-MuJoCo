from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from analyze_phase31_falls import joint_specs, normalize_obs, write_csv
from analyze_phase33_walk import (
    Condition,
    aggregate_condition,
    build_phase33_row,
    summarize_episode,
    write_contact_svg,
)
from humanoid_ai.envs import load_vecnormalize_env, make_humanoid_env


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 4 standard Humanoid-v5 PPO model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vec-normalize-stats", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out-dir", default="work/phase4_eval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--saturation-threshold", type=float, default=0.39)
    parser.add_argument("--slip-speed-threshold", type=float, default=0.35)
    parser.add_argument("--forward-distance-threshold", type=float, default=0.35)
    parser.add_argument("--lateral-distance-threshold", type=float, default=0.35)
    parser.add_argument("--upright-threshold", type=float, default=0.90)
    return parser.parse_args()


def actuator_names(model) -> list[str]:
    names: list[str] = []
    for index in range(model.nu):
        name = model.actuator(index).name
        names.append(name or f"actuator_{index}")
    return names


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model_path = Path(args.model)
    vec_path = Path(args.vec_normalize_stats)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found: {vec_path}")

    model = PPO.load(model_path, device=args.device)
    vecnorm_env = load_vecnormalize_env(
        vec_path,
        args.env_id,
        seed=args.seed,
        training=False,
        norm_reward=False,
    )
    env = make_humanoid_env(
        args.env_id,
        seed=args.seed,
        render_mode=None,
    )
    condition = Condition(
        key=args.run_name or model_path.stem,
        label=args.run_name or model_path.stem,
        model=model_path,
        vecnormalize=vec_path,
        curriculum_config=Path(),
        target_velocity=1.0,
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
        aggregate = aggregate_condition(rows, episode_summaries)
    finally:
        env.close()
        vecnorm_env.close()
    return rows, episode_summaries, aggregate


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than 0")
    run_name = args.run_name or Path(args.model).stem
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, episode_summaries, aggregate = evaluate(args)
    step_csv = out_dir / "phase4_step_log.csv"
    episode_csv = out_dir / "phase4_episode_summary.csv"
    summary_json = out_dir / "phase4_summary.json"
    contact_svg = out_dir / "phase4_contact_timeline.svg"
    write_csv(step_csv, rows)
    write_csv(episode_csv, episode_summaries)
    write_contact_svg(contact_svg, rows, f"Phase 4 {run_name} contact and motion")
    result = {
        "experiment": "phase4_standard_humanoid_eval",
        "model": str(Path(args.model).resolve()),
        "vec_normalize_stats": str(Path(args.vec_normalize_stats).resolve()),
        "episodes": args.episodes,
        "evaluation_seeds": list(range(args.seed, args.seed + args.episodes)),
        "max_steps": args.max_steps,
        "deterministic": True,
        "vecnormalize_training": False,
        "vecnormalize_norm_reward": False,
        "step_csv": str(step_csv),
        "episode_csv": str(episode_csv),
        "contact_svg": str(contact_svg),
        "summary": aggregate,
    }
    summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("phase4_evaluation_done=true")


if __name__ == "__main__":
    main()
