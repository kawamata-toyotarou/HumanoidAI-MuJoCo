from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


PHASE3_INFO_KEYS = (
    "phase3_original_reward",
    "phase3_reward_total",
    "phase3_reward_survive",
    "phase3_reward_velocity_tracking",
    "phase3_reward_ctrl",
    "phase3_reward_contact",
    "phase3_penalty_lateral_velocity",
    "phase3_penalty_torso_tilt",
    "phase3_penalty_torso_height",
    "phase3_penalty_action_smoothness",
    "phase3_action_delta_sq_mean",
    "phase3_action_delta_rms",
    "phase3_penalty_fall",
    "phase3_forward_velocity_error",
    "phase3_x_velocity",
    "phase3_y_velocity",
    "phase3_torso_upright",
    "phase3_torso_height",
    "phase3_stage_index",
)


@dataclass(frozen=True)
class Phase3StageRewardConfig:
    name: str
    stage_index: int
    target_forward_velocity: float
    velocity_sigma: float
    forward_tracking_weight: float
    survive_weight: float
    ctrl_weight: float
    contact_weight: float
    lateral_velocity_weight: float
    torso_tilt_weight: float
    torso_height_weight: float
    min_torso_height: float
    fall_penalty: float
    action_smoothness_weight: float = 0.0

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "Phase3StageRewardConfig":
        return cls(name=name, **data)

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class Phase3CurriculumConfig:
    active_stage: str
    stages: dict[str, Phase3StageRewardConfig]
    success_criteria: dict[str, dict[str, float]]

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        stage_name: str | None = None,
    ) -> "Phase3CurriculumConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        stages = {
            name: Phase3StageRewardConfig.from_dict(name, stage_data)
            for name, stage_data in data["stages"].items()
        }
        active_stage = stage_name or data.get("active_stage", "stage1")
        if active_stage not in stages:
            raise ValueError(f"Unknown curriculum stage: {active_stage}")
        return cls(
            active_stage=active_stage,
            stages=stages,
            success_criteria=data.get("success_criteria", {}),
        )

    @property
    def stage(self) -> Phase3StageRewardConfig:
        return self.stages[self.active_stage]

    @property
    def stage_success_criteria(self) -> dict[str, float]:
        return self.success_criteria.get(self.active_stage, {})


def compute_phase3_reward(
    cfg: Phase3StageRewardConfig,
    *,
    original_reward: float,
    terminated: bool,
    info: dict[str, Any],
    torso_upright: float,
    torso_height: float,
    action_delta_sq_mean: float = 0.0,
) -> tuple[float, dict[str, float]]:
    x_velocity = float(info.get("x_velocity", 0.0))
    y_velocity = float(info.get("y_velocity", 0.0))
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
    height_error = max(0.0, cfg.min_torso_height - torso_height)
    height_penalty = -cfg.torso_height_weight * height_error * height_error
    action_delta_sq_mean = max(0.0, float(action_delta_sq_mean))
    action_smoothness_penalty = -cfg.action_smoothness_weight * action_delta_sq_mean
    fall_penalty = cfg.fall_penalty if terminated else 0.0

    reward = (
        survive_reward
        + velocity_tracking
        + ctrl_reward
        + contact_reward
        + lateral_penalty
        + tilt_penalty
        + height_penalty
        + action_smoothness_penalty
        + fall_penalty
    )

    components = {
        "phase3_original_reward": float(original_reward),
        "phase3_reward_total": float(reward),
        "phase3_reward_survive": float(survive_reward),
        "phase3_reward_velocity_tracking": float(velocity_tracking),
        "phase3_reward_ctrl": float(ctrl_reward),
        "phase3_reward_contact": float(contact_reward),
        "phase3_penalty_lateral_velocity": float(lateral_penalty),
        "phase3_penalty_torso_tilt": float(tilt_penalty),
        "phase3_penalty_torso_height": float(height_penalty),
        "phase3_penalty_action_smoothness": float(action_smoothness_penalty),
        "phase3_action_delta_sq_mean": float(action_delta_sq_mean),
        "phase3_action_delta_rms": float(math.sqrt(action_delta_sq_mean)),
        "phase3_penalty_fall": float(fall_penalty),
        "phase3_forward_velocity_error": float(forward_error),
        "phase3_x_velocity": float(x_velocity),
        "phase3_y_velocity": float(y_velocity),
        "phase3_torso_upright": float(torso_upright),
        "phase3_torso_height": float(torso_height),
        "phase3_stage_index": float(cfg.stage_index),
    }
    return float(reward), components


class Phase3CurriculumRewardWrapper(gym.Wrapper):
    """Curriculum reward wrapper for Humanoid-v5 without modifying the base env."""

    def __init__(self, env: gym.Env, config: Phase3CurriculumConfig) -> None:
        super().__init__(env)
        self.config = config
        self._previous_action: np.ndarray | None = None

    @property
    def reward_config(self) -> dict[str, float | int | str]:
        return self.config.stage.to_dict()

    @property
    def success_criteria(self) -> dict[str, float]:
        return self.config.stage_success_criteria

    def torso_state(self) -> tuple[float, float]:
        base_env = self.unwrapped
        torso_body_id = base_env.model.body("torso").id
        torso_upright = float(base_env.data.xmat[torso_body_id][8])
        torso_height = float(base_env.data.qpos[2])
        return torso_upright, torso_height

    def reset(self, **kwargs: Any):
        self._previous_action = None
        return self.env.reset(**kwargs)

    def action_delta_sq_mean(self, action: Any) -> float:
        current = np.asarray(action, dtype=np.float64).reshape(-1)
        if self._previous_action is None:
            previous = np.zeros_like(current)
        else:
            previous = self._previous_action
        if previous.shape != current.shape:
            previous = np.zeros_like(current)
        delta = current - previous
        self._previous_action = current.copy()
        return float(np.mean(delta * delta)) if delta.size else 0.0

    def step(self, action: Any):
        action_delta_sq_mean = self.action_delta_sq_mean(action)
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        torso_upright, torso_height = self.torso_state()
        reward, components = compute_phase3_reward(
            self.config.stage,
            original_reward=float(original_reward),
            terminated=bool(terminated),
            info=info,
            torso_upright=torso_upright,
            torso_height=torso_height,
            action_delta_sq_mean=action_delta_sq_mean,
        )
        info.update(components)
        return obs, reward, terminated, truncated, info
