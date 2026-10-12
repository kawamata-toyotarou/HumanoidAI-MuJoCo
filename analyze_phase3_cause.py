from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
RUNS = {
    "phase28_reward_100k": {
        "eval_csv": ROOT / "work" / "phase28_reward_100k_eval.csv",
        "vecnormalize": ROOT
        / "checkpoints"
        / "phase28_reward_100k"
        / "phase28_reward_100k_seed4242_vecnormalize.pkl",
    },
    "phase29_reward_200k": {
        "eval_csv": ROOT / "work" / "phase29_reward_200k_eval.csv",
        "vecnormalize": ROOT
        / "checkpoints"
        / "phase29_reward_200k"
        / "phase29_reward_200k_seed4242_vecnormalize.pkl",
    },
}

METRIC_COLUMNS = [
    "reward",
    "survival_steps",
    "forward_distance",
    "x_velocity_per_step",
    "lateral_abs_displacement",
    "torso_upright_mean",
    "torso_upright_min",
    "target_velocity_fraction",
    "phase27_reward_total_per_step",
    "phase27_reward_survive_per_step",
    "phase27_reward_velocity_tracking_per_step",
    "phase27_reward_ctrl_per_step",
    "phase27_reward_contact_per_step",
    "phase27_penalty_lateral_velocity_per_step",
    "phase27_penalty_torso_tilt_per_step",
    "phase27_penalty_fall_per_step",
    "phase27_forward_velocity_error_per_step",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_value(rows: list[dict[str, str]], column: str) -> float:
    values = []
    for row in rows:
        raw_value = row.get(column)
        if raw_value in (None, ""):
            continue
        lowered = raw_value.lower()
        if lowered == "true":
            values.append(1.0)
        elif lowered == "false":
            values.append(0.0)
        else:
            values.append(float(raw_value))
    return float(np.mean(values)) if values else float("nan")


def summarize_array(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_vecnormalize(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "error": "file not found"}
    try:
        with path.open("rb") as handle:
            vecnorm = pickle.load(handle)
        obs_rms = getattr(vecnorm, "obs_rms", None)
        ret_rms = getattr(vecnorm, "ret_rms", None)
        return {
            "exists": True,
            "training": bool(getattr(vecnorm, "training", False)),
            "norm_obs": bool(getattr(vecnorm, "norm_obs", False)),
            "norm_reward": bool(getattr(vecnorm, "norm_reward", False)),
            "gamma": float(getattr(vecnorm, "gamma", float("nan"))),
            "obs_mean": summarize_array(obs_rms.mean) if obs_rms is not None else None,
            "obs_var": summarize_array(obs_rms.var) if obs_rms is not None else None,
            "ret_mean": summarize_array(ret_rms.mean) if ret_rms is not None else None,
            "ret_var": summarize_array(ret_rms.var) if ret_rms is not None else None,
        }
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"exists": True, "error": repr(exc)}


def main() -> None:
    summary: dict[str, Any] = {}
    comparison_rows: list[dict[str, str]] = []

    for name, paths in RUNS.items():
        rows = read_rows(paths["eval_csv"])
        metrics = {column: mean_value(rows, column) for column in METRIC_COLUMNS}
        metrics["fall_rate"] = mean_value(rows, "fell")
        summary[name] = {
            "metrics": metrics,
            "vecnormalize": summarize_vecnormalize(paths["vecnormalize"]),
        }
        comparison_rows.append(
            {
                "run": name,
                **{column: f"{metrics[column]:.9f}" for column in METRIC_COLUMNS},
                "fall_rate": f"{metrics['fall_rate']:.9f}",
            }
        )

    before = summary["phase28_reward_100k"]["metrics"]
    after = summary["phase29_reward_200k"]["metrics"]
    summary["delta_200k_minus_100k"] = {
        key: float(after[key] - before[key]) for key in [*METRIC_COLUMNS, "fall_rate"]
    }

    json_path = ROOT / "work" / "phase3_cause_analysis.json"
    csv_path = ROOT / "work" / "phase3_cause_analysis_summary.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    compact = {
        "100k": summary["phase28_reward_100k"]["metrics"],
        "200k": summary["phase29_reward_200k"]["metrics"],
        "delta": summary["delta_200k_minus_100k"],
        "json": str(json_path),
        "csv": str(csv_path),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print("phase3_cause_analysis_done=true")


if __name__ == "__main__":
    main()
