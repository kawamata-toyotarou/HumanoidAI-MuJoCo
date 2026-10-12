from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humanoid_ai.phase3_curriculum_reward import (  # noqa: E402
    Phase3CurriculumConfig,
    compute_phase3_reward,
)

CONFIG_PATH = ROOT / "configs" / "phase3_curriculum_default.json"


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def check_stage(stage_name: str) -> None:
    stage = Phase3CurriculumConfig.from_path(CONFIG_PATH, stage_name=stage_name).stage
    original_reward = 123.0
    info = {
        "reward_survive": 5.0,
        "reward_ctrl": -0.1,
        "reward_contact": -0.05,
        "x_velocity": stage.target_forward_velocity,
        "y_velocity": 0.2,
    }
    torso_upright = 0.95
    torso_height = stage.min_torso_height + 0.1

    reward, components = compute_phase3_reward(
        stage,
        original_reward=original_reward,
        terminated=False,
        info=info,
        torso_upright=torso_upright,
        torso_height=torso_height,
    )

    expected_velocity_tracking = stage.forward_tracking_weight
    expected_lateral_penalty = -stage.lateral_velocity_weight * (info["y_velocity"] ** 2)
    expected_tilt_penalty = -stage.torso_tilt_weight * ((1.0 - torso_upright) ** 2)
    expected_reward = (
        stage.survive_weight * info["reward_survive"]
        + expected_velocity_tracking
        + stage.ctrl_weight * info["reward_ctrl"]
        + stage.contact_weight * info["reward_contact"]
        + expected_lateral_penalty
        + expected_tilt_penalty
    )

    assert_close(reward, expected_reward, f"{stage_name}.reward")
    assert_close(
        components["phase3_reward_velocity_tracking"],
        expected_velocity_tracking,
        f"{stage_name}.velocity_tracking",
    )
    assert_close(
        components["phase3_penalty_lateral_velocity"],
        expected_lateral_penalty,
        f"{stage_name}.lateral_penalty",
    )
    assert_close(
        components["phase3_penalty_torso_tilt"],
        expected_tilt_penalty,
        f"{stage_name}.tilt_penalty",
    )
    assert_close(
        components["phase3_penalty_torso_height"],
        0.0,
        f"{stage_name}.height_penalty",
    )
    assert_close(
        components["phase3_forward_velocity_error"],
        0.0,
        f"{stage_name}.velocity_error",
    )

    fallen_reward, fallen_components = compute_phase3_reward(
        stage,
        original_reward=original_reward,
        terminated=True,
        info=info,
        torso_upright=torso_upright,
        torso_height=torso_height,
    )
    assert_close(
        fallen_components["phase3_penalty_fall"],
        stage.fall_penalty,
        f"{stage_name}.fall_penalty",
    )
    assert_close(
        fallen_reward,
        expected_reward + stage.fall_penalty,
        f"{stage_name}.fallen_reward",
    )


def main() -> None:
    for stage_name in ("stage1", "stage2", "stage3"):
        check_stage(stage_name)
    print("phase3_curriculum_reward_tests_done=true")


if __name__ == "__main__":
    main()
