from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_phase31_falls import write_csv, write_svg_plot
from analyze_phase33_walk import Condition, evaluate_condition, safe_float, summarize_episode, write_contact_svg
from analyze_phase36_smoothness_comparison import (
    COMPARISON_METRICS,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    metric_direction,
    threshold_sensitivity as phase36_threshold_sensitivity,
)


ROOT = Path(__file__).resolve().parent
STAGE1_CONFIG = ROOT / "configs/phase31_stage1_A_current.json"


@dataclass(frozen=True)
class BaselineCheckpoint:
    key: str
    label: str
    train_seed: int
    total_steps: int
    model: Path
    vecnormalize: Path

    @property
    def condition(self) -> Condition:
        return Condition(
            key=self.key,
            label=self.label,
            model=self.model,
            vecnormalize=self.vecnormalize,
            curriculum_config=STAGE1_CONFIG,
            target_velocity=0.25,
        )


CHECKPOINTS = [
    BaselineCheckpoint(
        key="A100k",
        label="A seed4242 100352 steps",
        train_seed=4242,
        total_steps=100352,
        model=ROOT / "checkpoints/phase32_A_current_100k/phase32_A_current_100k_seed4242_latest.zip",
        vecnormalize=ROOT / "checkpoints/phase32_A_current_100k/phase32_A_current_100k_seed4242_vecnormalize.pkl",
    ),
    BaselineCheckpoint(
        key="A125k",
        label="A seed4242 125352 steps",
        train_seed=4242,
        total_steps=125352,
        model=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_ppo_125352_steps.zip",
        vecnormalize=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_ppo_vecnormalize_125352_steps.pkl",
    ),
    BaselineCheckpoint(
        key="A150k",
        label="A seed4242 150352 steps",
        train_seed=4242,
        total_steps=150352,
        model=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_ppo_150352_steps.zip",
        vecnormalize=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_ppo_vecnormalize_150352_steps.pkl",
    ),
    BaselineCheckpoint(
        key="A175k",
        label="A seed4242 175352 steps",
        train_seed=4242,
        total_steps=175352,
        model=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_ppo_175352_steps.zip",
        vecnormalize=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_ppo_vecnormalize_175352_steps.pkl",
    ),
    BaselineCheckpoint(
        key="A200k_mid",
        label="A seed4242 200352 steps",
        train_seed=4242,
        total_steps=200352,
        model=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_ppo_200352_steps.zip",
        vecnormalize=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_ppo_vecnormalize_200352_steps.pkl",
    ),
    BaselineCheckpoint(
        key="A200k",
        label="A seed4242 200704 steps",
        train_seed=4242,
        total_steps=200704,
        model=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_latest.zip",
        vecnormalize=ROOT / "checkpoints/phase37_A4242_resume_200k/phase37_A4242_resume_200k_vecnormalize.pkl",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.7 longer Stage 1 A baseline analysis.")
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
    parser.add_argument("--out-dir", default="work/phase37_longer_baseline")
    parser.add_argument("--graph-dir", default="outputs/phase37_longer_baseline")
    return parser.parse_args()


def rows_by_episode(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), []).append(row)
    return grouped


def threshold_sensitivity(
    checkpoint: BaselineCheckpoint,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    pseudo_model = type(
        "Phase37ThresholdModel",
        (),
        {
            "reward_condition": "A",
            "train_seed": checkpoint.train_seed,
            "condition": checkpoint.condition,
        },
    )()
    output = phase36_threshold_sensitivity(pseudo_model, rows, args)
    for row in output:
        row["checkpoint"] = checkpoint.key
        row["total_steps"] = checkpoint.total_steps
    return output


def add_checkpoint_metadata(row: dict[str, Any], checkpoint: BaselineCheckpoint) -> dict[str, Any]:
    enriched = dict(row)
    enriched["checkpoint"] = checkpoint.key
    enriched["total_steps"] = checkpoint.total_steps
    enriched["train_seed"] = checkpoint.train_seed
    return enriched


def delta_100k_to_200k(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_checkpoint = {row["checkpoint"]: row for row in summary_rows}
    base = by_checkpoint["A100k"]
    final = by_checkpoint["A200k"]
    rows: list[dict[str, Any]] = []
    for metric in COMPARISON_METRICS:
        before = safe_float(base.get(metric))
        after = safe_float(final.get(metric))
        delta = after - before
        direction = metric_direction(metric)
        if direction == "higher":
            improved = delta > 0.0
        elif direction == "lower":
            improved = delta < 0.0
        else:
            improved = False
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "steps_100k": before,
                "steps_200k": after,
                "delta": delta,
                "improved": improved,
            }
        )
    return rows


def write_progress_svg(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("survival_steps_mean", "survival steps", "#2563eb"),
        ("forward_distance_mean", "forward distance", "#16a34a"),
        ("forward_velocity_mean", "forward velocity", "#f97316"),
        ("lateral_abs_displacement_mean", "lateral drift", "#dc2626"),
        ("walk_candidate_rate", "walk candidate rate", "#7c3aed"),
        ("action_delta_rms_mean", "action delta RMS", "#0891b2"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1120
    height = 760
    left = 170
    right = 45
    top = 42
    bottom = 40
    panel_height = (height - top - bottom) / len(metrics)
    min_step = min(int(row["total_steps"]) for row in summary_rows)
    max_step = max(int(row["total_steps"]) for row in summary_rows)

    def x_for_step(step: int) -> float:
        if max_step == min_step:
            return left
        return left + (step - min_step) / (max_step - min_step) * (width - left - right)

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="560" y="27" text-anchor="middle" font-family="Arial" font-size="20">Phase 3.7 A seed4242 checkpoint metrics</text>',
    ]
    sorted_rows = sorted(summary_rows, key=lambda row: int(row["total_steps"]))
    for index, (metric, label, color) in enumerate(metrics):
        y0 = top + index * panel_height
        values = [safe_float(row.get(metric)) for row in sorted_rows]
        value_min = min(values)
        value_max = max(values)
        if abs(value_max - value_min) < 1e-9:
            value_max = value_min + 1.0

        def y_for_value(value: float) -> float:
            return y0 + 12 + (value_max - value) / (value_max - value_min) * (panel_height - 32)

        body.append(f'<text x="16" y="{y0 + 30:.1f}" font-family="Arial" font-size="13">{label}</text>')
        body.append(f'<line x1="{left}" y1="{y0 + panel_height - 12:.1f}" x2="{width-right}" y2="{y0 + panel_height - 12:.1f}" stroke="#e5e7eb" />')
        points = []
        for row in sorted_rows:
            x = x_for_step(int(row["total_steps"]))
            y = y_for_value(safe_float(row.get(metric)))
            points.append(f"{x:.1f},{y:.1f}")
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />')
            body.append(
                f'<text x="{x + 5:.1f}" y="{y - 5:.1f}" font-family="Arial" font-size="10">'
                f'{safe_float(row.get(metric)):.3f}</text>'
            )
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" />')
        body.append(f'<text x="{left}" y="{y0 + panel_height - 1:.1f}" font-family="Arial" font-size="10">{min_step}</text>')
        body.append(f'<text x="{width-right-45}" y="{y0 + panel_height - 1:.1f}" font-family="Arial" font-size="10">{max_step}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def ensure_inputs_exist(checkpoints: list[BaselineCheckpoint]) -> None:
    missing = []
    for checkpoint in checkpoints:
        for path in [checkpoint.model, checkpoint.vecnormalize, STAGE1_CONFIG]:
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing phase 3.7 input files:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    ensure_inputs_exist(CHECKPOINTS)
    out_dir = ROOT / args.out_dir
    graph_dir = ROOT / args.graph_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    graph_paths: list[str] = []
    for checkpoint in CHECKPOINTS:
        rows, episode_summaries, aggregate = evaluate_condition(checkpoint.condition, args)
        aggregate = add_checkpoint_metadata(aggregate, checkpoint)
        summary_rows.append(aggregate)
        all_episode_rows.extend(add_checkpoint_metadata(row, checkpoint) for row in episode_summaries)
        threshold_rows.extend(threshold_sensitivity(checkpoint, rows, args))
        write_csv(out_dir / f"phase37_{checkpoint.key}_step_log.csv", rows)
        write_csv(out_dir / f"phase37_{checkpoint.key}_episode_summary.csv", episode_summaries)
        contact_svg = graph_dir / f"phase37_{checkpoint.key}_contact_timeline.svg"
        trajectory_svg = graph_dir / f"phase37_{checkpoint.key}_trajectory.svg"
        write_contact_svg(contact_svg, rows, f"Phase 3.7 {checkpoint.key} contact and motion")
        write_svg_plot(
            trajectory_svg,
            {1: [row for row in rows if int(row["episode"]) == 1]},
            title=f"Phase 3.7 {checkpoint.key} first-episode trajectory",
            x_key="x_position",
            y_key="y_position",
            x_label="x position",
            y_label="y position",
        )
        graph_paths.extend([str(contact_svg), str(trajectory_svg)])

    delta_rows = delta_100k_to_200k(summary_rows)
    progress_svg = graph_dir / "phase37_checkpoint_progress.svg"
    write_progress_svg(progress_svg, summary_rows)
    graph_paths.append(str(progress_svg))

    write_csv(out_dir / "phase37_checkpoint_summary.csv", summary_rows)
    write_csv(out_dir / "phase37_episode_summary_all.csv", all_episode_rows)
    write_csv(out_dir / "phase37_100k_to_200k_delta.csv", delta_rows)
    write_csv(out_dir / "phase37_threshold_sensitivity.csv", threshold_rows)

    result = {
        "experiment": "phase37_longer_baseline",
        "episodes_per_checkpoint": args.episodes,
        "evaluation_seeds": list(range(args.seed, args.seed + args.episodes)),
        "max_steps": args.max_steps,
        "checkpoint_summary_csv": str(out_dir / "phase37_checkpoint_summary.csv"),
        "episode_summary_csv": str(out_dir / "phase37_episode_summary_all.csv"),
        "delta_csv": str(out_dir / "phase37_100k_to_200k_delta.csv"),
        "threshold_sensitivity_csv": str(out_dir / "phase37_threshold_sensitivity.csv"),
        "graphs": graph_paths,
        "summary": summary_rows,
        "delta_100k_to_200k": delta_rows,
        "notes": [
            "All checkpoints use the unchanged Stage 1 A reward configuration.",
            "The 20 evaluation episodes are repeated measurements of one trained seed, not independent training runs.",
            "Survival improvements are not considered sufficient if forward distance or lateral stability worsens.",
        ],
    }
    json_path = out_dir / "phase37_longer_baseline.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("phase37_longer_baseline_done=true")


if __name__ == "__main__":
    main()
