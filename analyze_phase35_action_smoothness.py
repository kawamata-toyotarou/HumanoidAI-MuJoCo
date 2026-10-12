from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_phase31_falls import write_csv
from analyze_phase33_walk import (
    Condition,
    evaluate_condition,
    pearson_correlation,
    safe_float,
    summarize_episode,
)


ROOT = Path(__file__).resolve().parent


BASELINE_CONDITIONS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.5 action smoothness baseline analysis.")
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
    parser.add_argument("--out-dir", default="work/phase35_action_smoothness")
    parser.add_argument("--graph-dir", default="outputs/phase35_action_smoothness")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def action_distribution(condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_delta = [safe_float(row.get("action_delta_rms")) for row in rows]
    qvel_delta = [safe_float(row.get("qvel_delta_rms")) for row in rows]
    qvel_delta_rate = [safe_float(row.get("qvel_delta_rate_rms")) for row in rows]
    slip = [safe_float(row.get("foot_slip_speed_max")) for row in rows]
    return {
        "condition": condition,
        "steps": len(rows),
        "action_delta_rms_mean": float(np.mean(action_delta)) if action_delta else 0.0,
        "action_delta_rms_p50": percentile(action_delta, 50),
        "action_delta_rms_p75": percentile(action_delta, 75),
        "action_delta_rms_p90": percentile(action_delta, 90),
        "action_delta_rms_p95": percentile(action_delta, 95),
        "qvel_delta_rms_mean": float(np.mean(qvel_delta)) if qvel_delta else 0.0,
        "qvel_delta_rate_rms_mean": float(np.mean(qvel_delta_rate)) if qvel_delta_rate else 0.0,
        "foot_slip_speed_max_mean": float(np.mean(slip)) if slip else 0.0,
        "slip_action_delta_correlation": pearson_correlation(slip, action_delta),
        "suggested_action_smoothness_weight": 0.2,
        "expected_penalty_at_p75": -0.2 * percentile(action_delta, 75) ** 2,
        "expected_penalty_at_p90": -0.2 * percentile(action_delta, 90) ** 2,
    }


def rows_by_episode(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), []).append(row)
    return grouped


def threshold_sensitivity(
    condition: Condition,
    rows: list[dict[str, Any]],
    *,
    base_slip: float,
    base_forward: float,
    base_lateral: float,
    base_upright: float,
) -> list[dict[str, Any]]:
    scenarios = [
        ("base", base_slip, base_forward, base_lateral, base_upright),
        ("strict_slip", 0.25, base_forward, base_lateral, base_upright),
        ("loose_slip", 0.45, base_forward, base_lateral, base_upright),
        ("strict_forward", base_slip, 0.50, base_lateral, base_upright),
        ("loose_forward", base_slip, 0.25, base_lateral, base_upright),
        ("strict_lateral", base_slip, base_forward, 0.25, base_upright),
        ("loose_lateral", base_slip, base_forward, 0.50, base_upright),
        ("strict_upright", base_slip, base_forward, base_lateral, 0.95),
        ("loose_upright", base_slip, base_forward, base_lateral, 0.85),
    ]
    grouped = rows_by_episode(rows)
    output = []
    for scenario, slip, forward, lateral, upright in scenarios:
        summaries = [
            summarize_episode(
                condition,
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
                "condition": condition.key,
                "scenario": scenario,
                "slip_speed_threshold": slip,
                "forward_distance_threshold": forward,
                "lateral_distance_threshold": lateral,
                "upright_threshold": upright,
                "walk_candidate_rate": sum(1.0 for item in summaries if item["walk_candidate"]) / len(summaries),
                "stable_walk_success_rate": sum(1.0 for item in summaries if item["stable_walk_success"]) / len(summaries),
            }
        )
    return output


def write_slip_action_svg(path: Path, rows_by_condition: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 920
    height = 560
    left = 70
    right = 35
    top = 45
    bottom = 60
    colors = {
        "seed4242": "#2563eb",
        "seed5252": "#16a34a",
        "seed6262": "#f97316",
    }
    all_x = [safe_float(row.get("action_delta_rms")) for rows in rows_by_condition.values() for row in rows]
    all_y = [safe_float(row.get("foot_slip_speed_max")) for rows in rows_by_condition.values() for row in rows]
    max_x = max(all_x + [1e-9])
    max_y = max(all_y + [1e-9])

    def sx(value: float) -> float:
        return left + value / max_x * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - value / max_y * (height - top - bottom)

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="460" y="28" text-anchor="middle" font-family="Arial" font-size="20">Phase 3.5 foot slip vs action delta RMS</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" />',
        f'<text x="{width / 2:.0f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">action delta RMS</text>',
        '<text x="20" y="280" transform="rotate(-90 20 280)" text-anchor="middle" font-family="Arial" font-size="14">contact foot slip speed</text>',
    ]
    legend_y = 58
    for condition, rows in rows_by_condition.items():
        color = colors.get(condition, "#111827")
        body.append(f'<circle cx="{width - 170}" cy="{legend_y}" r="5" fill="{color}" />')
        body.append(f'<text x="{width - 158}" y="{legend_y + 4}" font-family="Arial" font-size="12">{condition}</text>')
        legend_y += 18
        for row in rows:
            body.append(
                f'<circle cx="{sx(safe_float(row.get("action_delta_rms"))):.2f}" '
                f'cy="{sy(safe_float(row.get("foot_slip_speed_max"))):.2f}" r="2.1" fill="{color}" opacity="0.35" />'
            )
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def ensure_inputs_exist() -> None:
    missing = []
    for condition in BASELINE_CONDITIONS:
        for path in [condition.model, condition.vecnormalize, condition.curriculum_config]:
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing phase 3.5 input files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    ensure_inputs_exist()
    out_dir = ROOT / args.out_dir
    graph_dir = ROOT / args.graph_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = []
    action_distribution_rows = []
    threshold_rows = []
    rows_for_graph: dict[str, list[dict[str, Any]]] = {}
    all_episode_summaries = []

    for condition in BASELINE_CONDITIONS:
        rows, episode_summaries, aggregate = evaluate_condition(condition, args)
        write_csv(out_dir / f"phase35_{condition.key}_step_log.csv", rows)
        write_csv(out_dir / f"phase35_{condition.key}_episode_summary.csv", episode_summaries)
        baseline_summary.append(aggregate)
        all_episode_summaries.extend(episode_summaries)
        action_distribution_rows.append(action_distribution(condition.key, rows))
        threshold_rows.extend(
            threshold_sensitivity(
                condition,
                rows,
                base_slip=args.slip_speed_threshold,
                base_forward=args.forward_distance_threshold,
                base_lateral=args.lateral_distance_threshold,
                base_upright=args.upright_threshold,
            )
        )
        rows_for_graph[condition.key] = rows

    graph_path = graph_dir / "phase35_slip_vs_action_delta.svg"
    write_slip_action_svg(graph_path, rows_for_graph)
    write_csv(out_dir / "phase35_baseline_summary.csv", baseline_summary)
    write_csv(out_dir / "phase35_episode_summary_all.csv", all_episode_summaries)
    write_csv(out_dir / "phase35_action_delta_distribution.csv", action_distribution_rows)
    write_csv(out_dir / "phase35_threshold_sensitivity.csv", threshold_rows)

    result = {
        "experiment": "phase35_action_smoothness",
        "episodes_per_model": args.episodes,
        "evaluation_seeds": list(range(args.seed, args.seed + args.episodes)),
        "max_steps": args.max_steps,
        "baseline_summary_csv": str(out_dir / "phase35_baseline_summary.csv"),
        "episode_summary_csv": str(out_dir / "phase35_episode_summary_all.csv"),
        "action_distribution_csv": str(out_dir / "phase35_action_delta_distribution.csv"),
        "threshold_sensitivity_csv": str(out_dir / "phase35_threshold_sensitivity.csv"),
        "slip_action_graph": str(graph_path),
        "smoothness_config": "configs/phase35_stage1_action_smoothness.json",
        "baseline_summary": baseline_summary,
        "action_delta_distribution": action_distribution_rows,
        "threshold_sensitivity": threshold_rows,
        "notes": [
            "The smoothness reward uses the raw action sent to the MuJoCo environment, independent of VecNormalize observation scaling.",
            "A conservative weight of 0.2 is used so the p75 action-delta penalty stays smaller than the main velocity/survival terms.",
            "Threshold sensitivity is reported because walk_candidate can change when lateral, slip, forward, or upright thresholds move.",
        ],
    }
    json_path = out_dir / "phase35_action_smoothness_analysis.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("phase35_action_smoothness_analysis_done=true")


if __name__ == "__main__":
    main()
