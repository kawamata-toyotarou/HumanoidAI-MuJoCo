from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_phase37_longer_baseline import (  # noqa: E402
    CHECKPOINTS,
    delta_100k_to_200k,
)


def main() -> None:
    steps = [checkpoint.total_steps for checkpoint in CHECKPOINTS]
    assert steps == sorted(steps)
    assert steps[0] == 100352
    assert steps[-1] == 200704
    assert all(checkpoint.train_seed == 4242 for checkpoint in CHECKPOINTS)

    rows = [
        {
            "checkpoint": "A100k",
            "survival_steps_mean": 80.0,
            "fall_rate": 1.0,
            "forward_distance_mean": 0.8,
            "lateral_abs_displacement_mean": 0.4,
        },
        {
            "checkpoint": "A200k",
            "survival_steps_mean": 100.0,
            "fall_rate": 0.8,
            "forward_distance_mean": 0.6,
            "lateral_abs_displacement_mean": 0.2,
        },
    ]
    deltas = {row["metric"]: row for row in delta_100k_to_200k(rows)}
    assert deltas["survival_steps_mean"]["delta"] == 20.0
    assert deltas["survival_steps_mean"]["improved"] is True
    assert deltas["fall_rate"]["delta"] == -0.19999999999999996
    assert deltas["fall_rate"]["improved"] is True
    assert deltas["forward_distance_mean"]["delta"] == -0.20000000000000007
    assert deltas["forward_distance_mean"]["improved"] is False
    assert deltas["lateral_abs_displacement_mean"]["delta"] == -0.2
    assert deltas["lateral_abs_displacement_mean"]["improved"] is True
    print("phase37_longer_baseline_tests_done=true")


if __name__ == "__main__":
    main()
