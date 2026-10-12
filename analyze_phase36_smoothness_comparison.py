from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_phase31_falls import write_csv, write_svg_plot
from analyze_phase33_walk import Condition, evaluate_condition, safe_float, summarize_episode, write_contact_svg


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ComparedModel:
    reward_condition: str
    train_seed: int
    condition: Condition


COMPARED_MODELS = [
    ComparedModel(
        reward_condition="A",
        train_seed=4242,
        condition=Condition(
            key="A_seed4242",
            label="A_seed4242",
            model=ROOT / "checkpoints/phase32_A_current_100k/phase32_A_current_100k_seed4242_latest.zip",
            vecnormalize=ROOT / "checkpoints/phase32_A_current_100k/phase32_A_current_100k_seed4242_vecnormalize.pkl",
            curriculum_config=ROOT / "configs/phase31_stage1_A_current.json",
            target_velocity=0.25,
        ),
    ),
    ComparedModel(
        reward_condition="A",
        train_seed=5252,
        condition=Condition(
            key="A_seed5252",
            label="A_seed5252",
            model=ROOT / "checkpoints/phase34_A_seed5252_100k/phase34_A_seed5252_100k_latest.zip",
            vecnormalize=ROOT / "checkpoints/phase34_A_seed5252_100k/phase34_A_seed5252_100k_vecnormalize.pkl",
            curriculum_config=ROOT / "configs/phase31_stage1_A_current.json",
            target_velocity=0.25,
        ),
    ),
    ComparedModel(
        reward_condition="A",
        train_seed=6262,
        condition=Condition(
            key="A_seed6262",
            label="A_seed6262",
            model=ROOT / "checkpoints/phase34_A_seed6262_100k/phase34_A_seed6262_100k_latest.zip",
            vecnormalize=ROOT / "checkpoints/phase34_A_seed6262_100k/phase34_A_seed6262_100k_vecnormalize.pkl",
            curriculum_config=ROOT / "configs/phase31_stage1_A_current.json",
            target_velocity=0.25,
        ),
    ),
    ComparedModel(
        reward_condition="Smooth",
        train_seed=4242,
        condition=Condition(
            key="Smooth_seed4242",
            label="Smooth_seed4242",
            model=ROOT / "checkpoints/phase36_smooth_seed4242_100k/phase36_smooth_seed4242_100k_latest.zip",
            vecnormalize=ROOT / "checkpoints/phase36_smooth_seed4242_100k/phase36_smooth_seed4242_100k_vecnormalize.pkl",
            curriculum_config=ROOT / "configs/phase35_stage1_action_smoothness.json",
            target_velocity=0.25,
        ),
    ),
    ComparedModel(
        reward_condition="Smooth",
        train_seed=5252,
        condition=Condition(
            key="Smooth_seed5252",
            label="Smooth_seed5252",
            model=ROOT / "checkpoints/phase36_smooth_seed5252_100k/phase36_smooth_seed5252_100k_latest.zip",
            vecnormalize=ROOT / "checkpoints/phase36_smooth_seed5252_100k/phase36_smooth_seed5252_100k_vecnormalize.pkl",
            curriculum_config=ROOT / "configs/phase35_stage1_action_smoothness.json",
            target_velocity=0.25,
        ),
    ),
    ComparedModel(
        reward_condition="Smooth",
        train_seed=6262,
        condition=Condition(
            key="Smooth_seed6262",
            label="Smooth_seed6262",
            model=ROOT / "checkpoints/phase36_smooth_seed6262_100k/phase36_smooth_seed6262_100k_latest.zip",
            vecnormalize=ROOT / "checkpoints/phase36_smooth_seed6262_100k/phase36_smooth_seed6262_100k_vecnormalize.pkl",
            curriculum_config=ROOT / "configs/phase35_stage1_action_smoothness.json",
            target_velocity=0.25,
        ),
    ),
]


COMPARISON_METRICS = [
    "survival_steps_mean",
    "fall_rate",
    "stable_walk_success_rate",
    "walk_candidate_rate",
    "forward_distance_mean",
    "forward_velocity_mean",
    "lateral_abs_displacement_mean",
    "torso_height_mean",
    "torso_height_std_mean",
    "torso_upright_mean",
    "contact_switch_count_mean",
    "walk_cycle_steps_mean",
    "foot_slip_speed_mean",
    "foot_slip_speed_max_mean",
    "action_delta_rms_mean",
    "action_saturation_fraction_mean",
    "qvel_delta_rms_mean",
    "qvel_delta_rate_rms_mean",
    "slip_action_delta_correlation_mean",
]


HIGHER_IS_BETTER = {
    "survival_steps_mean",
    "stable_walk_success_rate",
    "walk_candidate_rate",
    "forward_distance_mean",
    "forward_velocity_mean",
    "torso_height_mean",
    "torso_upright_mean",
    "contact_switch_count_mean",
}


LOWER_IS_BETTER = {
    "fall_rate",
    "lateral_abs_displacement_mean",
    "torso_height_std_mean",
    "foot_slip_speed_mean",
    "foot_slip_speed_max_mean",
    "action_delta_rms_mean",
    "action_saturation_fraction_mean",
    "qvel_delta_rms_mean",
    "qvel_delta_rate_rms_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.6 A vs Smooth comparison.")
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
    parser.add_argument("--out-dir", default="work/phase36_smoothness_comparison")
    parser.add_argument("--graph-dir", default="outputs/phase36_smoothness_comparison")
    return parser.parse_args()


def metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher"
    if metric in LOWER_IS_BETTER:
        return "lower"
    return "neutral"


def is_improved(metric: str, delta: float) -> bool:
    direction = metric_direction(metric)
    if direction == "higher":
        return delta > 0.0
    if direction == "lower":
        return delta < 0.0
    return False


def same_seed_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["reward_condition"], int(row["train_seed"])): row for row in summary_rows}
    delta_rows: list[dict[str, Any]] = []
    seeds = sorted({int(row["train_seed"]) for row in summary_rows})
    for seed in seeds:
        base = by_key.get(("A", seed))
        smooth = by_key.get(("Smooth", seed))
        if base is None or smooth is None:
            continue
        row: dict[str, Any] = {"train_seed": seed}
        for metric in COMPARISON_METRICS:
            base_value = safe_float(base.get(metric))
            smooth_value = safe_float(smooth.get(metric))
            delta = smooth_value - base_value
            row[f"A_{metric}"] = base_value
            row[f"Smooth_{metric}"] = smooth_value
            row[f"delta_{metric}"] = delta
            row[f"improved_{metric}"] = is_improved(metric, delta)
        delta_rows.append(row)
    return delta_rows


def aggregate_by_reward(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for reward_condition in ["A", "Smooth"]:
        rows = [row for row in summary_rows if row["reward_condition"] == reward_condition]
        aggregate: dict[str, Any] = {"reward_condition": reward_condition, "model_count": len(rows)}
        for metric in COMPARISON_METRICS:
            values = [safe_float(row.get(metric)) for row in rows]
            aggregate[f"{metric}_mean"] = sum(values) / len(values) if values else 0.0
            aggregate[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            aggregate[f"{metric}_min"] = min(values) if values else 0.0
            aggregate[f"{metric}_max"] = max(values) if values else 0.0
        output.append(aggregate)
    return output


def improvement_counts(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in COMPARISON_METRICS:
        values = [safe_float(row.get(f"delta_{metric}")) for row in delta_rows]
        improved = sum(1 for row in delta_rows if bool(row.get(f"improved_{metric}")))
        output.append(
            {
                "metric": metric,
                "direction": metric_direction(metric),
                "seed_count": len(delta_rows),
                "improved_seed_count": improved,
                "worsened_or_unchanged_seed_count": len(delta_rows) - improved,
                "delta_mean": sum(values) / len(values) if values else 0.0,
                "delta_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "delta_min": min(values) if values else 0.0,
                "delta_max": max(values) if values else 0.0,
            }
        )
    return output


def rows_by_episode(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), []).append(row)
    return grouped


def threshold_sensitivity(model: ComparedModel, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    scenarios = [
        ("base", args.slip_speed_threshold, args.forward_distance_threshold, args.lateral_distance_threshold, args.upright_threshold),
        ("strict_slip", 0.25, args.forward_distance_threshold, args.lateral_distance_threshold, args.upright_threshold),
        ("loose_slip", 0.45, args.forward_distance_threshold, args.lateral_distance_threshold, args.upright_threshold),
        ("strict_lateral", args.slip_speed_threshold, args.forward_distance_threshold, 0.25, args.upright_threshold),
        ("loose_lateral", args.slip_speed_threshold, args.forward_distance_threshold, 0.50, args.upright_threshold),
        ("strict_upright", args.slip_speed_threshold, args.forward_distance_threshold, args.lateral_distance_threshold, 0.95),
        ("loose_upright", args.slip_speed_threshold, args.forward_distance_threshold, args.lateral_distance_threshold, 0.85),
    ]
    output: list[dict[str, Any]] = []
    grouped = rows_by_episode(rows)
    for scenario, slip, forward, lateral, upright in scenarios:
        summaries = [
            summarize_episode(
                model.condition,
                episode_rows,
                slip_speed_threshold=slip,
                forward_distance_threshold=forward,
                lateral_distance_threshold=lateral,
                upright_threshold=upright,
            )
            for episode_rows in grouped.values()
        ]
        output.append(
            {
                "reward_condition": model.reward_condition,
                "train_seed": model.train_seed,
                "scenario": scenario,
                "walk_candidate_rate": sum(1.0 for item in summaries if item["walk_candidate"]) / len(summaries),
                "stable_walk_success_rate": sum(1.0 for item in summaries if item["stable_walk_success"]) / len(summaries),
            }
        )
    return output


def write_comparison_svg(path: Path, delta_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("delta_survival_steps_mean", "survival step delta"),
        ("delta_forward_distance_mean", "forward distance delta"),
        ("delta_lateral_abs_displacement_mean", "lateral drift delta"),
        ("delta_action_delta_rms_mean", "action delta RMS delta"),
        ("delta_foot_slip_speed_mean", "foot slip delta"),
        ("delta_walk_candidate_rate", "walk candidate delta"),
    ]
    width = 1020
    height = 520
    left = 190
    right = 35
    top = 45
    bottom = 45
    row_height = (height - top - bottom) / len(metrics)
    colors = {4242: "#2563eb", 5252: "#16a34a", 6262: "#f97316"}
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="510" y="28" text-anchor="middle" font-family="Arial" font-size="20">Phase 3.6 Smooth - A deltas by training seed</text>',
    ]
    center = left + (width - left - right) / 2.0
    body.append(f'<line x1="{center:.1f}" y1="{top}" x2="{center:.1f}" y2="{height-bottom}" stroke="#111827" />')
    for metric_index, (metric, label) in enumerate(metrics):
        y0 = top + metric_index * row_height
        values = [safe_float(row.get(metric)) for row in delta_rows]
        max_abs = max([abs(value) for value in values] + [1e-9])
        body.append(f'<text x="15" y="{y0 + 30:.1f}" font-family="Arial" font-size="13">{label}</text>')
        body.append(f'<line x1="{left}" y1="{y0 + row_height - 8:.1f}" x2="{width-right}" y2="{y0 + row_height - 8:.1f}" stroke="#e5e7eb" />')
        for row_index, row in enumerate(delta_rows):
            value = safe_float(row.get(metric))
            seed = int(row["train_seed"])
            bar_width = abs(value) / max_abs * ((width - left - right) / 2.0 - 35.0)
            x = center if value >= 0 else center - bar_width
            y = y0 + 8 + row_index * 17
            color = colors.get(seed, "#111827")
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="12" fill="{color}" />')
            text_x = x + bar_width + 5 if value >= 0 else x - 5
            anchor = "start" if value >= 0 else "end"
            body.append(
                f'<text x="{text_x:.1f}" y="{y + 10:.1f}" text-anchor="{anchor}" font-family="Arial" font-size="11">'
                f'{seed}: {value:.3f}</text>'
            )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def ensure_inputs_exist(models: list[ComparedModel]) -> None:
    missing = []
    for model in models:
        for path in [model.condition.model, model.condition.vecnormalize, model.condition.curriculum_config]:
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing phase 3.6 input files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    ensure_inputs_exist(COMPARED_MODELS)
    out_dir = ROOT / args.out_dir
    graph_dir = ROOT / args.graph_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    graph_paths: list[str] = []
    for model in COMPARED_MODELS:
        rows, episode_summaries, aggregate = evaluate_condition(model.condition, args)
        aggregate["reward_condition"] = model.reward_condition
        aggregate["train_seed"] = model.train_seed
        summary_rows.append(aggregate)
        all_episode_rows.extend(episode_summaries)
        threshold_rows.extend(threshold_sensitivity(model, rows, args))
        write_csv(out_dir / f"phase36_{model.condition.key}_step_log.csv", rows)
        write_csv(out_dir / f"phase36_{model.condition.key}_episode_summary.csv", episode_summaries)
        contact_svg = graph_dir / f"phase36_{model.condition.key}_contact_timeline.svg"
        trajectory_svg = graph_dir / f"phase36_{model.condition.key}_trajectory.svg"
        write_contact_svg(contact_svg, rows, f"Phase 3.6 {model.condition.key} contact and motion")
        write_svg_plot(
            trajectory_svg,
            {1: [row for row in rows if int(row["episode"]) == 1]},
            title=f"Phase 3.6 {model.condition.key} first-episode trajectory",
            x_key="x_position",
            y_key="y_position",
            x_label="x position",
            y_label="y position",
        )
        graph_paths.extend([str(contact_svg), str(trajectory_svg)])

    delta_rows = same_seed_deltas(summary_rows)
    aggregate_rows = aggregate_by_reward(summary_rows)
    improvement_rows = improvement_counts(delta_rows)
    comparison_svg = graph_dir / "phase36_smooth_minus_A_deltas.svg"
    write_comparison_svg(comparison_svg, delta_rows)
    graph_paths.append(str(comparison_svg))

    write_csv(out_dir / "phase36_model_summary.csv", summary_rows)
    write_csv(out_dir / "phase36_episode_summary_all.csv", all_episode_rows)
    write_csv(out_dir / "phase36_same_seed_deltas.csv", delta_rows)
    write_csv(out_dir / "phase36_reward_condition_stats.csv", aggregate_rows)
    write_csv(out_dir / "phase36_improvement_counts.csv", improvement_rows)
    write_csv(out_dir / "phase36_threshold_sensitivity.csv", threshold_rows)

    result = {
        "experiment": "phase36_smoothness_comparison",
        "episodes_per_model": args.episodes,
        "evaluation_seeds": list(range(args.seed, args.seed + args.episodes)),
        "max_steps": args.max_steps,
        "summary_csv": str(out_dir / "phase36_model_summary.csv"),
        "delta_csv": str(out_dir / "phase36_same_seed_deltas.csv"),
        "condition_stats_csv": str(out_dir / "phase36_reward_condition_stats.csv"),
        "improvement_counts_csv": str(out_dir / "phase36_improvement_counts.csv"),
        "threshold_sensitivity_csv": str(out_dir / "phase36_threshold_sensitivity.csv"),
        "graphs": graph_paths,
        "summary": summary_rows,
        "same_seed_deltas": delta_rows,
        "condition_stats": aggregate_rows,
        "improvement_counts": improvement_rows,
        "notes": [
            "A and Smooth have different reward functions, so physical metrics are the primary comparison.",
            "The 20 evaluation episodes are repeated measurements per trained model, not independent training runs.",
            "Smooth is considered useful only if action-delta reductions do not erase forward distance or survival.",
        ],
    }
    json_path = out_dir / "phase36_smoothness_comparison.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("phase36_smoothness_comparison_done=true")


if __name__ == "__main__":
    main()
