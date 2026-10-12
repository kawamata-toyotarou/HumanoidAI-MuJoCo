from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym


PHASE27_INFO_KEYS = (
    "phase27_original_reward",
    "phase27_reward_total",
    "phase27_reward_survive",
    "phase27_reward_velocity_tracking",
    "phase27_reward_ctrl",
    "phase27_reward_contact",
    "phase27_penalty_lateral_velocity",
    "phase27_penalty_torso_tilt",
    "phase27_penalty_fall",
    "phase27_forward_velocity_error",
    "phase27_x_velocity",
    "phase27_y_velocity",
    "phase27_torso_upright",
    "phase27_torso_height",
)


@dataclass(frozen=True)
class Phase27RewardConfig:
    target_forward_velocity: float = 1.0
    velocity_sigma: float = 0.5
    forward_tracking_weight: float = 3.0
    survive_weight: float = 1.0
    ctrl_weight: float = 1.0
    contact_weight: float = 1.0
    lateral_velocity_weight: float = 1.0
    torso_tilt_weight: float = 2.0
    fall_penalty: float = -25.0

    @classmethod
    def from_path(cls, path: str | Path | None) -> "Phase27RewardConfig":
        if path is None:
            return cls()
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class Phase27HumanoidRewardWrapper(gym.Wrapper):
    """Replace Humanoid-v5 reward with forward-tracking and stability terms."""

    def __init__(self, env: gym.Env, config: Phase27RewardConfig | None = None) -> None:
        super().__init__(env)
        self.config = config or Phase27RewardConfig()

    @property
    def reward_config(self) -> dict[str, float]:
        return self.config.to_dict()

    def torso_state(self) -> tuple[float, float]:
        base_env = self.unwrapped
        torso_body_id = base_env.model.body("torso").id
        torso_upright = float(base_env.data.xmat[torso_body_id][8])
        torso_height = float(base_env.data.qpos[2])
        return torso_upright, torso_height

    def step(self, action: Any):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        cfg = self.config

        x_velocity = float(info.get("x_velocity", 0.0))
        y_velocity = float(info.get("y_velocity", 0.0))
        torso_upright, torso_height = self.torso_state()
        upright_clipped = max(-1.0, min(1.0, torso_upright))

        survive_reward = cfg.survive_weight * float(info.get("reward_survive", 0.0))
        forward_error = x_velocity - cfg.target_forward_velocity
        velocity_tracking = cfg.forward_tracking_weight * math.exp(
            -(forward_error * forward_error) / (2.0 * cfg.velocity_sigma * cfg.velocity_sigma)
        )
        ctrl_reward = cfg.ctrl_weight * float(info.get("reward_ctrl", 0.0))
        contact_reward = cfg.contact_weight * float(info.get("reward_contact", 0.0))
        lateral_penalty = -cfg.lateral_velocity_weight * y_velocity * y_velocity
        tilt_penalty = -cfg.torso_tilt_weight * (1.0 - upright_clipped) * (1.0 - upright_clipped)
        fall_penalty = cfg.fall_penalty if terminated else 0.0

        reward = (
            survive_reward
            + velocity_tracking
            + ctrl_reward
            + contact_reward
            + lateral_penalty
            + tilt_penalty
            + fall_penalty
        )

        info.update(
            {
                "phase27_original_reward": float(original_reward),
                "phase27_reward_total": float(reward),
                "phase27_reward_survive": float(survive_reward),
                "phase27_reward_velocity_tracking": float(velocity_tracking),
                "phase27_reward_ctrl": float(ctrl_reward),
                "phase27_reward_contact": float(contact_reward),
                "phase27_penalty_lateral_velocity": float(lateral_penalty),
                "phase27_penalty_torso_tilt": float(tilt_penalty),
                "phase27_penalty_fall": float(fall_penalty),
                "phase27_forward_velocity_error": float(forward_error),
                "phase27_x_velocity": float(x_velocity),
                "phase27_y_velocity": float(y_velocity),
                "phase27_torso_upright": float(torso_upright),
                "phase27_torso_height": float(torso_height),
            }
        )
        return obs, float(reward), terminated, truncated, info
