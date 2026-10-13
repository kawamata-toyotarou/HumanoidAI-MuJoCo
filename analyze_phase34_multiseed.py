from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_phase31_falls import write_csv, write_svg_plot
from analyze_phase33_walk import Condition, evaluate_condition, safe_float, write_contact_svg


ROOT = Path(__file__).resolve().parent


PHASE34_CONDITIONS = [
    Condition(
        key="seed4242",
        label="stage1_A_seed4242_reused_phase32",
        model=ROOT / "checkpoints/phase32_A_current_100k/phase32_A_current_100k_seed4242_latest.zip",
        vecnormalize=ROOT / "checkpoints/phase32_A_current_100k/phase32_A_current_100k_seed4242_vecnormalize.pkl",
        curriculum_config=ROOT / "configs/phase31_stage1_A_current.json",
        target_velocity=0.25,
    ),
    Condition(
        key="seed5252",
        label="stage1_A_seed5252_phase34",
        model=ROOT / "checkpoints/phase34_A_seed5252_100k/phase34_A_seed5252_100k_latest.zip",
        vecnormalize=ROOT / "checkpoints/phase34_A_seed5252_100k/phase34_A_seed5252_100k_vecnormalize.pkl",
        curriculum_config=ROOT / "configs/phase31_stage1_A_current.json",
        target_velocity=0.25,
    ),
    Condition(
        key="seed6262",
        label="stage1_A_seed6262_phase34",
        model=ROOT / "checkpoints/phase34_A_seed6262_100k/phase34_A_seed6262_100k_latest.zip",
        vecnormalize=ROOT / "checkpoints/phase34_A_seed6262_100k/phase34_A_seed6262_100k_vecnormalize.pkl",
        curriculum_config=ROOT / "configs/phase31_stage1_A_current.json",
        target_velocity=0.25,
    ),
]


MODEL_STAT_METRICS = [
    "survival_steps_mean",
    "fall_rate",
    "truncated_rate",
    "walk_candidate_rate",
    "stable_walk_success_rate",
    "forward_distance_mean",
    "forward_velocity_mean",
    "lateral_abs_displacement_mean",
    "torso_height_mean",
    "torso_height_std_mean",
    "torso_height_drop_last20_mean",
    "torso_upright_mean",
    "torso_upright_min_mean",
    "contact_switch_count_mean",
    "walk_cycle_steps_mean",
    "left_contact_rate",
    "right_contact_rate",
    "double_contact_rate",
    "nonfoot_floor_contact_rate",
    "foot_slip_speed_mean",
    "foot_slip_speed_max_mean",
    "action_delta_rms_mean",
    "action_delta_abs_max_mean",
    "qvel_delta_rms_mean",
    "qvel_delta_rate_rms_mean",
    "slip_action_delta_correlation_mean",
    "action_saturation_fraction_mean",
    "ctrl_saturation_fraction_mean",
    "qvel_abs_max_mean",
]


@dataclass(frozen=True)
class TrainingRun:
    seed: int
    run_name: str
    trained_steps: int
    elapsed_seconds: float | None
    steps_per_second: float | None
    reused_existing_model: bool


TRAINING_RUNS = [
    TrainingRun(
        seed=4242,
        run_name="phase32_A_current_100k",
        trained_steps=100352,
        elapsed_seconds=None,
        steps_per_second=None,
        reused_existing_model=True,
    ),
    TrainingRun(
        seed=5252,
        run_name="phase34_A_seed5252_100k",
        trained_steps=100352,
        elapsed_seconds=264.871,
        steps_per_second=378.871,
        reused_existing_model=False,
    ),
    TrainingRun(
        seed=6262,
        run_name="phase34_A_seed6262_100k",
        trained_steps=100352,
        elapsed_seconds=196.373,
        steps_per_second=511.028,
        reused_existing_model=False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.4 Stage 1 A multiseed reproducibility analysis.")
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
    parser.add_argument("--out-dir", default="work/phase34_multiseed")
    parser.add_argument("--graph-dir", default="outputs/phase34_multiseed")
    return parser.parse_args()


def model_metric_stats(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for metric in MODEL_STAT_METRICS:
        values = [safe_float(row.get(metric)) for row in aggregate_rows]
        stats.append(
            {
                "metric": metric,
                "model_count": len(values),
                "mean": sum(values) / len(values) if values else 0.0,
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
        )
    return stats


def attach_training_metadata(aggregate_rows: list[dict[str, Any]]) -> None:
    run_by_seed = {f"seed{run.seed}": run for run in TRAINING_RUNS}
    for row in aggregate_rows:
        run = run_by_seed.get(str(row["condition"]))
        if run is None:
            continue
        row["training_seed"] = run.seed
        row["training_run_name"] = run.run_name
        row["trained_steps"] = run.trained_steps
        row["training_elapsed_seconds"] = run.elapsed_seconds
        row["training_steps_per_second"] = run.steps_per_second
        row["reused_existing_model"] = run.reused_existing_model


def write_multiseed_bar_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("survival_steps_mean", "survival steps"),
        ("forward_distance_mean", "forward distance"),
        ("forward_velocity_mean", "forward velocity"),
        ("lateral_abs_displacement_mean", "lateral drift"),
        ("walk_candidate_rate", "walk candidate"),
        ("stable_walk_success_rate", "stable success"),
    ]
    width = 1020
    height = 520
    left = 170
    right = 35
    top = 50
    bottom = 45
    group_height = (height - top - bottom) / len(metrics)
    colors = ["#2563eb", "#16a34a", "#f97316"]
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="510" y="28" text-anchor="middle" font-family="Arial" font-size="20">Phase 3.4 Stage 1 A multiseed comparison</text>',
    ]
    for metric_index, (metric, label) in enumerate(metrics):
        y0 = top + metric_index * group_height
        max_value = max([safe_float(row.get(metric)) for row in rows] + [1e-9])
        if metric.endswith("_rate"):
            max_value = 1.0
        body.append(f'<text x="15" y="{y0 + 32:.1f}" font-family="Arial" font-size="13">{label}</text>')
        body.append(f'<line x1="{left}" y1="{y0 + group_height - 8:.1f}" x2="{width-right}" y2="{y0 + group_height - 8:.1f}" stroke="#e5e7eb" />')
        for row_index, row in enumerate(rows):
            value = safe_float(row.get(metric))
            bar_width = (value / max_value) * (width - left - right - 95)
            y = y0 + 8 + row_index * 17
            color = colors[row_index % len(colors)]
            body.append(f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" height="12" fill="{color}" />')
            body.append(
                f'<text x="{left + bar_width + 6:.1f}" y="{y + 10:.1f}" font-family="Arial" font-size="11" fill="#111827">'
                f'{row["condition"]}: {value:.3f}</text>'
            )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def ensure_inputs_exist(conditions: list[Condition]) -> None:
    missing = []
    for condition in conditions:
        for path in [condition.model, condition.vecnormalize, condition.curriculum_config]:
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing phase 3.4 input files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    ensure_inputs_exist(PHASE34_CONDITIONS)
    out_dir = ROOT / args.out_dir
    graph_dir = ROOT / args.graph_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    all_episode_summaries: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    graph_paths: list[str] = []
    for condition in PHASE34_CONDITIONS:
        rows, episode_summaries, aggregate = evaluate_condition(condition, args)
        attach_training_metadata([aggregate])
        write_csv(out_dir / f"phase34_{condition.key}_step_log.csv", rows)
        write_csv(out_dir / f"phase34_{condition.key}_episode_summary.csv", episode_summaries)

        first_episode_rows = {1: [row for row in rows if int(row["episode"]) == 1]}
        contact_svg = graph_dir / f"phase34_{condition.key}_contact_timeline.svg"
        height_svg = graph_dir / f"phase34_{condition.key}_height.svg"
        trajectory_svg = graph_dir / f"phase34_{condition.key}_trajectory.svg"
        write_contact_svg(contact_svg, rows, f"Phase 3.4 {condition.key} contact and motion")
        write_svg_plot(
            height_svg,
            first_episode_rows,
            title=f"Phase 3.4 {condition.key} torso height",
            x_key="step",
            y_key="torso_height",
            x_label="step",
            y_label="torso height",
        )
        write_svg_plot(
            trajectory_svg,
            first_episode_rows,
            title=f"Phase 3.4 {condition.key} first-episode trajectory",
            x_key="x_position",
            y_key="y_position",
            x_label="x position",
            y_label="y position",
        )
        graph_paths.extend([str(contact_svg), str(height_svg), str(trajectory_svg)])
        all_episode_summaries.extend(episode_summaries)
        aggregate_rows.append(aggregate)

    stats_rows = model_metric_stats(aggregate_rows)
    stats_by_metric = {row["metric"]: row for row in stats_rows}
    summary_svg = graph_dir / "phase34_multiseed_metric_bars.svg"
    write_multiseed_bar_svg(summary_svg, aggregate_rows)
    graph_paths.append(str(summary_svg))

    write_csv(out_dir / "phase34_episode_summary_all.csv", all_episode_summaries)
    write_csv(out_dir / "phase34_seed_summary.csv", aggregate_rows)
    write_csv(out_dir / "phase34_model_metric_stats.csv", stats_rows)

    result = {
        "experiment": "phase34_multiseed_stage1_A",
        "conditions": [condition.key for condition in PHASE34_CONDITIONS],
        "episodes_per_model": args.episodes,
        "evaluation_seed_start": args.seed,
        "evaluation_seeds": list(range(args.seed, args.seed + args.episodes)),
        "max_steps": args.max_steps,
        "curriculum_config": "configs/phase31_stage1_A_current.json",
        "curriculum_stage": args.curriculum_stage,
        "thresholds": {
            "saturation_threshold": args.saturation_threshold,
            "slip_speed_threshold": args.slip_speed_threshold,
            "forward_distance_threshold": args.forward_distance_threshold,
            "lateral_distance_threshold": args.lateral_distance_threshold,
            "upright_threshold": args.upright_threshold,
        },
        "training_runs": [run.__dict__ for run in TRAINING_RUNS],
        "summary_csv": str(out_dir / "phase34_seed_summary.csv"),
        "episode_summary_csv": str(out_dir / "phase34_episode_summary_all.csv"),
        "metric_stats_csv": str(out_dir / "phase34_model_metric_stats.csv"),
        "graphs": graph_paths,
        "summary": aggregate_rows,
        "model_metric_stats": stats_by_metric,
        "interpretation_notes": [
            "Each training seed is treated as one trained model.",
            "The 20 evaluation episodes are repeated measurements of each model, not 20 independent training runs.",
            "Contact switching is interpreted with forward distance, lateral drift, foot slip, and torso posture to avoid mistaking shuffling for walking.",
            "Reward totals are not used as the primary comparison metric because all models share the same reward here, but physical metrics are still prioritized.",
        ],
    }
    json_path = out_dir / "phase34_multiseed_analysis.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("phase34_multiseed_analysis_done=true")


if __name__ == "__main__":
    main()
