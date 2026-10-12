from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_phase33_walk import (  # noqa: E402
    compressed_single_support,
    contact_switch_metrics,
    saturation_fraction,
    summarize_episode,
)


def row(step: int, phase: str, *, x: float = 0.0, y: float = 0.0, slip: float = 0.0) -> dict[str, object]:
    left = phase in {"left", "double"}
    right = phase in {"right", "double"}
    return {
        "condition": "T",
        "condition_label": "test",
        "episode": 1,
        "step": step,
        "terminated": step == 6,
        "truncated": False,
        "reward": 1.0,
        "x_position": x,
        "y_position": y,
        "x_velocity": 0.2,
        "torso_height": 1.2,
        "torso_upright": 0.95,
        "left_foot_contact": int(left),
        "right_foot_contact": int(right),
        "double_foot_contact": int(left and right),
        "nonfoot_floor_contact": 0,
        "left_foot_slip_speed": slip if left else 0.0,
        "right_foot_slip_speed": slip if right else 0.0,
        "action_saturation_fraction": 0.25,
        "ctrl_saturation_fraction": 0.5,
        "qvel_abs_max": 2.0,
    }


def main() -> None:
    rows = [
        row(1, "left", x=0.0),
        row(2, "left", x=0.1),
        row(3, "double", x=0.2),
        row(4, "right", x=0.3),
        row(5, "right", x=0.4),
        row(6, "left", x=0.5),
    ]
    assert compressed_single_support(rows) == [("left", 1), ("right", 4), ("left", 6)]
    switches = contact_switch_metrics(rows)
    assert switches["contact_switch_count"] == 2.0
    assert switches["walk_cycle_steps_mean"] == 5.0

    summary = summarize_episode(
        type("Condition", (), {"key": "T", "label": "test"})(),
        rows,
        slip_speed_threshold=0.35,
        forward_distance_threshold=0.35,
        lateral_distance_threshold=0.35,
        upright_threshold=0.90,
    )
    assert summary["walk_candidate"] is True

    slipping_rows = [dict(item, left_foot_slip_speed=1.0, right_foot_slip_speed=1.0) for item in rows]
    slipping_summary = summarize_episode(
        type("Condition", (), {"key": "T", "label": "test"})(),
        slipping_rows,
        slip_speed_threshold=0.35,
        forward_distance_threshold=0.35,
        lateral_distance_threshold=0.35,
        upright_threshold=0.90,
    )
    assert slipping_summary["walk_candidate"] is False

    assert saturation_fraction([0.0, 0.4, -0.41, 0.2], 0.39) == 0.5
    print("phase33_walk_metric_tests_done=true")


if __name__ == "__main__":
    main()
