from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_phase36_smoothness_comparison import (  # noqa: E402
    aggregate_by_reward,
    improvement_counts,
    same_seed_deltas,
)


def main() -> None:
    rows = [
        {
            "reward_condition": "A",
            "train_seed": 1,
            "survival_steps_mean": 80.0,
            "fall_rate": 1.0,
            "forward_distance_mean": 0.8,
            "action_delta_rms_mean": 0.4,
        },
        {
            "reward_condition": "Smooth",
            "train_seed": 1,
            "survival_steps_mean": 90.0,
            "fall_rate": 0.8,
            "forward_distance_mean": 0.7,
            "action_delta_rms_mean": 0.2,
        },
    ]
    deltas = same_seed_deltas(rows)
    assert len(deltas) == 1
    assert deltas[0]["delta_survival_steps_mean"] == 10.0
    assert deltas[0]["improved_survival_steps_mean"] is True
    assert deltas[0]["delta_fall_rate"] == -0.19999999999999996
    assert deltas[0]["improved_fall_rate"] is True
    assert deltas[0]["delta_forward_distance_mean"] == -0.10000000000000009
    assert deltas[0]["improved_forward_distance_mean"] is False
    assert deltas[0]["delta_action_delta_rms_mean"] == -0.2
    assert deltas[0]["improved_action_delta_rms_mean"] is True

    counts = {row["metric"]: row for row in improvement_counts(deltas)}
    assert counts["survival_steps_mean"]["improved_seed_count"] == 1
    assert counts["forward_distance_mean"]["improved_seed_count"] == 0

    aggregate = {row["reward_condition"]: row for row in aggregate_by_reward(rows)}
    assert aggregate["A"]["survival_steps_mean_mean"] == 80.0
    assert aggregate["Smooth"]["action_delta_rms_mean_mean"] == 0.2
    print("phase36_smoothness_comparison_tests_done=true")


if __name__ == "__main__":
    main()
