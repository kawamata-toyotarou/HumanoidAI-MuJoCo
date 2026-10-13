from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_phase34_multiseed import model_metric_stats  # noqa: E402


def main() -> None:
    rows = [
        {"survival_steps_mean": 80.0, "fall_rate": 1.0, "forward_distance_mean": 0.5},
        {"survival_steps_mean": 100.0, "fall_rate": 0.5, "forward_distance_mean": 0.7},
        {"survival_steps_mean": 120.0, "fall_rate": 0.0, "forward_distance_mean": 0.9},
    ]
    stats = {row["metric"]: row for row in model_metric_stats(rows)}
    assert stats["survival_steps_mean"]["mean"] == 100.0
    assert round(stats["survival_steps_mean"]["std"], 6) == round((800.0 / 3.0) ** 0.5, 6)
    assert stats["survival_steps_mean"]["min"] == 80.0
    assert stats["survival_steps_mean"]["max"] == 120.0
    assert stats["fall_rate"]["mean"] == 0.5
    print("phase34_multiseed_stats_tests_done=true")


if __name__ == "__main__":
    main()
