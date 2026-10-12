from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent


CONDITIONS = [
    {
        "condition": "A",
        "label": "現行Stage 1",
        "config": "configs/phase31_stage1_A_current.json",
        "eval_csv": "work/phase32_A_current_100k_eval.csv",
        "step_csv": "work/phase32_A_current_100k_step_log.csv",
        "analysis_json": "work/phase32_A_current_100k_fall_analysis.json",
        "train_seconds": 191.922,
        "train_steps_per_second": 522.879,
    },
    {
        "condition": "B",
        "label": "転倒ペナルティー強化",
        "config": "configs/phase31_stage1_B_fall_penalty.json",
        "eval_csv": "work/phase32_B_fall_penalty_100k_eval.csv",
        "step_csv": "work/phase32_B_fall_penalty_100k_step_log.csv",
        "analysis_json": "work/phase32_B_fall_penalty_100k_fall_analysis.json",
        "train_seconds": 183.729,
        "train_steps_per_second": 546.194,
    },
    {
        "condition": "C",
        "label": "低速・姿勢重視",
        "config": "configs/phase31_stage1_C_slow_stability.json",
        "eval_csv": "work/phase32_C_slow_stability_100k_eval.csv",
        "step_csv": "work/phase32_C_slow_stability_100k_step_log.csv",
        "analysis_json": "work/phase32_C_slow_stability_100k_fall_analysis.json",
        "train_seconds": 180.967,
        "train_steps_per_second": 554.533,
    },
]


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def truth(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"true", "1", "yes"}


def summarize_condition(config: dict[str, object]) -> dict[str, object]:
    eval_rows = read_csv(str(config["eval_csv"]))
    step_rows = read_csv(str(config["step_csv"]))
    analysis = json.loads((ROOT / str(config["analysis_json"])).read_text(encoding="utf-8"))
    last_window = analysis["fall_summary"]["aggregate"]

    result: dict[str, object] = {
        "condition": config["condition"],
        "label": config["label"],
        "config": config["config"],
        "train_steps": 100352,
        "train_seconds": config["train_seconds"],
        "train_steps_per_second": config["train_steps_per_second"],
        "episodes": len(eval_rows),
        "reward_mean": mean(f(row, "reward") for row in eval_rows),
        "survival_steps_mean": mean(f(row, "survival_steps") for row in eval_rows),
        "terminated_count": sum(1 for row in eval_rows if truth(row, "fell")),
        "truncated_count": sum(1 for row in eval_rows if truth(row, "truncated")),
        "fall_rate": mean(1.0 if truth(row, "fell") else 0.0 for row in eval_rows),
        "forward_distance_mean": mean(f(row, "forward_distance") for row in eval_rows),
        "forward_velocity_mean": mean(f(row, "x_velocity_per_step") for row in eval_rows),
        "lateral_abs_displacement_mean": mean(f(row, "lateral_abs_displacement") for row in eval_rows),
        "torso_height_mean": mean(f(row, "torso_height_mean") for row in eval_rows),
        "torso_upright_mean": mean(f(row, "torso_upright_mean") for row in eval_rows),
        "target_velocity_fraction_mean": mean(f(row, "target_velocity_fraction") for row in eval_rows),
        "phase3_reward_total_per_step_mean": mean(f(row, "phase3_reward_total_per_step") for row in eval_rows),
        "phase3_survive_per_step_mean": mean(f(row, "phase3_reward_survive_per_step") for row in eval_rows),
        "phase3_velocity_tracking_per_step_mean": mean(f(row, "phase3_reward_velocity_tracking_per_step") for row in eval_rows),
        "phase3_lateral_penalty_per_step_mean": mean(f(row, "phase3_penalty_lateral_velocity_per_step") for row in eval_rows),
        "phase3_tilt_penalty_per_step_mean": mean(f(row, "phase3_penalty_torso_tilt_per_step") for row in eval_rows),
        "phase3_height_penalty_per_step_mean": mean(f(row, "phase3_penalty_torso_height_per_step") for row in eval_rows),
        "phase3_fall_penalty_per_step_mean": mean(f(row, "phase3_penalty_fall_per_step") for row in eval_rows),
        "left_foot_contact_rate": mean(f(row, "left_foot_contact") for row in step_rows),
        "right_foot_contact_rate": mean(f(row, "right_foot_contact") for row in step_rows),
        "double_foot_contact_rate": mean(f(row, "double_foot_contact") for row in step_rows),
        "nonfoot_floor_contact_rate": mean(f(row, "nonfoot_floor_contact") for row in step_rows),
        "left_foot_contact_rate_last20": last_window["left_foot_contact_fraction_last_window"],
        "right_foot_contact_rate_last20": last_window["right_foot_contact_fraction_last_window"],
        "double_foot_contact_rate_last20": last_window["double_foot_contact_fraction_last_window"],
        "nonfoot_floor_contact_rate_last20": last_window["nonfoot_floor_contact_fraction_last_window"],
    }
    return result


def main() -> None:
    summaries = [summarize_condition(config) for config in CONDITIONS]
    out_csv = ROOT / "work/phase32_abc_summary.csv"
    out_json = ROOT / "work/phase32_abc_summary.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    out_json.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_csv={out_csv}")
    print(f"summary_json={out_json}")
    print("phase32_summary_done=true")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
