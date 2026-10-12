from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO

from humanoid_ai.envs import load_vecnormalize_env, make_humanoid_env
from humanoid_ai.phase27_reward import PHASE27_INFO_KEYS
from humanoid_ai.phase3_curriculum_reward import PHASE3_INFO_KEYS


REWARD_COMPONENT_KEYS = (
    "reward_survive",
    "reward_forward",
    "reward_ctrl",
    "reward_contact",
    "x_velocity",
    "y_velocity",
    *PHASE27_INFO_KEYS,
    *PHASE3_INFO_KEYS,
)

STATE_METRIC_KEYS = (
    "torso_height",
    "torso_upright",
    "x_position",
    "y_position",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reward, survival length, forward distance, and fall rate."
    )
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--agent", choices=("model", "random"), default="model")
    parser.add_argument("--model", default="checkpoints/humanoid_ppo_latest.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--vec-normalize-stats",
        default=None,
        help="Path to VecNormalize .pkl stats for normalized models.",
    )
    parser.add_argument(
        "--reward-config",
        default=None,
        help="Optional Phase 2.7 reward config JSON for custom-reward models.",
    )
    parser.add_argument(
        "--curriculum-config",
        default=None,
        help="Optional Phase 3 curriculum config JSON. Mutually exclusive with --reward-config.",
    )
    parser.add_argument("--curriculum-stage", default=None)
    parser.add_argument("--target-forward-velocity", type=float, default=1.0)
    parser.add_argument("--target-velocity-tolerance", type=float, default=0.25)
    parser.add_argument("--csv", default=None, help="Optional path for per-episode CSV output.")
    return parser.parse_args()


def torso_x(env) -> float:
    return float(env.unwrapped.data.qpos[0])


def base_env_from_plain(env):
    return env.unwrapped


def torso_state(base_env) -> dict[str, float]:
    qpos = base_env.data.qpos
    torso_body_id = base_env.model.body("torso").id
    torso_matrix = base_env.data.xmat[torso_body_id]
    return {
        "x_position": float(qpos[0]),
        "y_position": float(qpos[1]),
        "torso_height": float(qpos[2]),
        "torso_upright": float(torso_matrix[8]),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def get_info_value(info: dict, key: str) -> float:
    value = info.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def init_episode_trackers() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    component_totals = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
    state_totals = {key: 0.0 for key in STATE_METRIC_KEYS}
    state_mins = {key: float("inf") for key in STATE_METRIC_KEYS}
    return component_totals, state_totals, state_mins


def update_state_trackers(
    state: dict[str, float],
    state_totals: dict[str, float],
    state_mins: dict[str, float],
) -> None:
    for key in STATE_METRIC_KEYS:
        value = float(state.get(key, 0.0))
        state_totals[key] += value
        state_mins[key] = min(state_mins[key], value)


def state_with_info_position(state: dict[str, float], info: dict) -> dict[str, float]:
    updated = state.copy()
    if "x_position" in info:
        updated["x_position"] = get_info_value(info, "x_position")
    if "y_position" in info:
        updated["y_position"] = get_info_value(info, "y_position")
    return updated


def build_row(
    *,
    episode: int,
    reward: float,
    length: int,
    start_state: dict[str, float],
    end_state: dict[str, float],
    terminated: bool,
    truncated: bool,
    component_totals: dict[str, float],
    state_totals: dict[str, float],
    state_mins: dict[str, float],
    target_velocity_hits: int,
) -> dict[str, float | int | bool]:
    row: dict[str, float | int | bool] = {
        "episode": episode,
        "reward": reward,
        "survival_steps": length,
        "forward_distance": end_state["x_position"] - start_state["x_position"],
        "lateral_displacement": end_state["y_position"] - start_state["y_position"],
        "lateral_abs_displacement": abs(end_state["y_position"] - start_state["y_position"]),
        "target_velocity_steps": target_velocity_hits,
        "target_velocity_fraction": target_velocity_hits / length if length > 0 else 0.0,
        "fell": bool(terminated),
        "truncated": bool(truncated),
    }
    row.update(component_totals)
    for key, total in component_totals.items():
        row[f"{key}_per_step"] = total / length if length > 0 else 0.0
    for key in STATE_METRIC_KEYS:
        row[f"{key}_mean"] = state_totals[key] / length if length > 0 else 0.0
        row[f"{key}_min"] = state_mins[key] if length > 0 else 0.0
        row[f"{key}_final"] = end_state[key]
    return row


def evaluate_plain(args: argparse.Namespace, model: PPO | None) -> list[dict[str, float | int | bool]]:
    env = make_humanoid_env(
        args.env_id,
        seed=args.seed,
        render_mode=None,
        reward_config_path=args.reward_config,
        curriculum_config_path=args.curriculum_config,
        curriculum_stage=args.curriculum_stage,
    )
    rows: list[dict[str, float | int | bool]] = []
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            base_env = base_env_from_plain(env)
            start_state = torso_state(base_env)
            end_state = start_state.copy()
            episode_reward = 0.0
            component_totals, state_totals, state_mins = init_episode_trackers()
            target_velocity_hits = 0
            terminated = False
            truncated = False
            length = 0

            for step in range(1, args.max_steps + 1):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                length = step
                end_state = state_with_info_position(torso_state(base_env), info)
                for key in REWARD_COMPONENT_KEYS:
                    component_totals[key] += get_info_value(info, key)
                if abs(get_info_value(info, "x_velocity") - args.target_forward_velocity) <= args.target_velocity_tolerance:
                    target_velocity_hits += 1
                update_state_trackers(end_state, state_totals, state_mins)
                if terminated or truncated:
                    break

            rows.append(
                build_row(
                    episode=episode + 1,
                    reward=episode_reward,
                    length=length,
                    start_state=start_state,
                    end_state=end_state,
                    terminated=terminated,
                    truncated=truncated,
                    component_totals=component_totals,
                    state_totals=state_totals,
                    state_mins=state_mins,
                    target_velocity_hits=target_velocity_hits,
                )
            )
    finally:
        env.close()

    return rows


def evaluate_vecnormalize(args: argparse.Namespace, model: PPO | None) -> list[dict[str, float | int | bool]]:
    if model is None:
        raise ValueError("--vec-normalize-stats is only supported with --agent model")
    env = load_vecnormalize_env(
        args.vec_normalize_stats,
        args.env_id,
        seed=args.seed,
        render_mode=None,
        training=False,
        norm_reward=False,
        reward_config_path=args.reward_config,
        curriculum_config_path=args.curriculum_config,
        curriculum_stage=args.curriculum_stage,
    )
    rows: list[dict[str, float | int | bool]] = []
    try:
        for episode in range(args.episodes):
            env.seed(args.seed + episode)
            obs = env.reset()
            base_env = env.venv.envs[0].unwrapped
            start_state = torso_state(base_env)
            end_state = start_state.copy()
            episode_reward = 0.0
            component_totals, state_totals, state_mins = init_episode_trackers()
            target_velocity_hits = 0
            terminated = False
            truncated = False
            length = 0

            for step in range(1, args.max_steps + 1):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                pre_step_state = torso_state(base_env)
                obs, rewards, dones, infos = env.step(action)
                info = infos[0]
                episode_reward += float(rewards[0])
                length = step
                terminated = bool(info.get("terminal_observation") is not None and not info.get("TimeLimit.truncated", False))
                truncated = bool(info.get("TimeLimit.truncated", False))
                if bool(dones[0]):
                    end_state = state_with_info_position(pre_step_state, info)
                else:
                    end_state = state_with_info_position(torso_state(base_env), info)
                for key in REWARD_COMPONENT_KEYS:
                    component_totals[key] += get_info_value(info, key)
                if abs(get_info_value(info, "x_velocity") - args.target_forward_velocity) <= args.target_velocity_tolerance:
                    target_velocity_hits += 1
                update_state_trackers(end_state, state_totals, state_mins)
                if bool(dones[0]):
                    break

            rows.append(
                build_row(
                    episode=episode + 1,
                    reward=episode_reward,
                    length=length,
                    start_state=start_state,
                    end_state=end_state,
                    terminated=terminated,
                    truncated=truncated,
                    component_totals=component_totals,
                    state_totals=state_totals,
                    state_mins=state_mins,
                    target_velocity_hits=target_velocity_hits,
                )
            )
    finally:
        env.close()

    return rows


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be greater than 0")
    if args.reward_config and args.curriculum_config:
        raise ValueError("--reward-config and --curriculum-config are mutually exclusive")

    model = None
    if args.agent == "model":
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = PPO.load(model_path, device=args.device)

    if args.vec_normalize_stats:
        rows = evaluate_vecnormalize(args, model)
    else:
        rows = evaluate_plain(args, model)

    rewards = [float(row["reward"]) for row in rows]
    survival_steps = [float(row["survival_steps"]) for row in rows]
    forward_distances = [float(row["forward_distance"]) for row in rows]
    avg_forward_velocities = [float(row["x_velocity_per_step"]) for row in rows]
    fall_rate = sum(1 for row in rows if row["fell"]) / len(rows)
    truncated_rate = sum(1 for row in rows if row["truncated"]) / len(rows)
    component_means = {
        f"{key}_mean": mean([float(row.get(key, 0.0)) for row in rows])
        for key in REWARD_COMPONENT_KEYS
    }
    component_per_step_means = {
        f"{key}_per_step_mean": mean([float(row.get(f"{key}_per_step", 0.0)) for row in rows])
        for key in REWARD_COMPONENT_KEYS
    }
    state_means = {
        f"{key}_mean": mean([float(row.get(f"{key}_mean", 0.0)) for row in rows])
        for key in STATE_METRIC_KEYS
    }
    state_mins = {
        f"{key}_min_mean": mean([float(row.get(f"{key}_min", 0.0)) for row in rows])
        for key in STATE_METRIC_KEYS
    }

    summary = {
        "env_id": args.env_id,
        "agent": args.agent,
        "model": str(Path(args.model).resolve()) if args.agent == "model" else None,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "reward_mean": mean(rewards),
        "survival_steps_mean": mean(survival_steps),
        "forward_distance_mean": mean(forward_distances),
        "forward_velocity_mean": mean(avg_forward_velocities),
        "lateral_abs_displacement_mean": mean([float(row["lateral_abs_displacement"]) for row in rows]),
        "target_forward_velocity": args.target_forward_velocity,
        "target_velocity_tolerance": args.target_velocity_tolerance,
        "target_velocity_fraction_mean": mean([float(row["target_velocity_fraction"]) for row in rows]),
        "fall_rate": fall_rate,
        "truncated_rate": truncated_rate,
        **component_means,
        **component_per_step_means,
        **state_means,
        **state_mins,
        "episodes_detail": rows,
    }

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode",
                    "reward",
                    "survival_steps",
                    "forward_distance",
                    "lateral_displacement",
                    "lateral_abs_displacement",
                    "target_velocity_steps",
                    "target_velocity_fraction",
                    "fell",
                    "truncated",
                    *REWARD_COMPONENT_KEYS,
                    *(f"{key}_per_step" for key in REWARD_COMPONENT_KEYS),
                    *(f"{key}_mean" for key in STATE_METRIC_KEYS),
                    *(f"{key}_min" for key in STATE_METRIC_KEYS),
                    *(f"{key}_final" for key in STATE_METRIC_KEYS),
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        summary["csv"] = str(csv_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("evaluation_done=true")


if __name__ == "__main__":
    main()
