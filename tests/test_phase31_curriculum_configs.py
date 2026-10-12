from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humanoid_ai.phase3_curriculum_reward import (  # noqa: E402
    Phase3CurriculumConfig,
    compute_phase3_reward,
)


CONFIGS = {
    "A": ROOT / "configs" / "phase31_stage1_A_current.json",
    "B": ROOT / "configs" / "phase31_stage1_B_fall_penalty.json",
    "C": ROOT / "configs" / "phase31_stage1_C_slow_stability.json",
}


def main() -> None:
    stages = {
        key: Phase3CurriculumConfig.from_path(path, stage_name="stage1").stage
        for key, path in CONFIGS.items()
    }
    assert stages["A"].target_forward_velocity == 0.25
    assert stages["B"].fall_penalty < stages["A"].fall_penalty
    assert stages["C"].target_forward_velocity < stages["A"].target_forward_velocity
    assert stages["C"].torso_tilt_weight >= stages["A"].torso_tilt_weight

    base_info = {
        "reward_survive": 5.0,
        "reward_ctrl": -0.1,
        "reward_contact": -0.05,
        "x_velocity": 0.18,
        "y_velocity": 0.1,
    }
    for key, stage in stages.items():
        reward, components = compute_phase3_reward(
            stage,
            original_reward=0.0,
            terminated=True,
            info=base_info,
            torso_upright=0.9,
            torso_height=stage.min_torso_height - 0.02,
        )
        assert components["phase3_penalty_fall"] == stage.fall_penalty
        expected_without_total = sum(
            value
            for name, value in components.items()
            if (
                name not in ("phase3_reward_total", "phase3_original_reward")
                and (name.startswith("phase3_reward_") or name.startswith("phase3_penalty_"))
            )
        )
        assert abs(reward - expected_without_total) < 1e-12, key
    print("phase31_curriculum_config_tests_done=true")


if __name__ == "__main__":
    main()
