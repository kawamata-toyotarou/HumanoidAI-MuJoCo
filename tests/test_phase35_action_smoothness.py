from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humanoid_ai.phase3_curriculum_reward import (  # noqa: E402
    Phase3CurriculumConfig,
    Phase3CurriculumRewardWrapper,
)


class DummyEnv(gym.Env):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def main() -> None:
    config = Phase3CurriculumConfig.from_path(ROOT / "configs" / "phase35_stage1_action_smoothness.json")
    wrapper = Phase3CurriculumRewardWrapper(DummyEnv(), config)

    assert wrapper.action_delta_sq_mean(np.array([1.0, -1.0, 0.5])) == 0.75
    assert wrapper.action_delta_sq_mean(np.array([1.0, -0.5, 0.0])) == (0.0 + 0.25 + 0.25) / 3.0

    wrapper.reset()
    assert wrapper.action_delta_sq_mean(np.array([1.0, -1.0, 0.5])) == 0.75
    assert config.stage.action_smoothness_weight == 0.2
    print("phase35_action_smoothness_tests_done=true")


if __name__ == "__main__":
    main()
