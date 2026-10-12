from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from humanoid_ai.phase27_reward import (
    PHASE27_INFO_KEYS,
    Phase27HumanoidRewardWrapper,
    Phase27RewardConfig,
)
from humanoid_ai.phase3_curriculum_reward import (
    PHASE3_INFO_KEYS,
    Phase3CurriculumConfig,
    Phase3CurriculumRewardWrapper,
)


def make_humanoid_env(
    env_id: str = "Humanoid-v5",
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    monitor_dir: str | Path | None = None,
    reward_config_path: str | Path | None = None,
    curriculum_config_path: str | Path | None = None,
    curriculum_stage: str | None = None,
) -> gym.Env:
    """Create a Humanoid environment with SB3's Monitor wrapper."""
    if reward_config_path is not None and curriculum_config_path is not None:
        raise ValueError("Use either reward_config_path or curriculum_config_path, not both")

    env = gym.make(env_id, render_mode=render_mode)
    monitor_info_keywords: tuple[str, ...] = ()
    if curriculum_config_path is not None:
        curriculum_config = Phase3CurriculumConfig.from_path(
            curriculum_config_path,
            stage_name=curriculum_stage,
        )
        env = Phase3CurriculumRewardWrapper(env, curriculum_config)
        monitor_info_keywords = PHASE3_INFO_KEYS
    elif reward_config_path is not None:
        reward_config = Phase27RewardConfig.from_path(reward_config_path)
        env = Phase27HumanoidRewardWrapper(env, reward_config)
        monitor_info_keywords = PHASE27_INFO_KEYS

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    monitor_file = None
    if monitor_dir is not None:
        monitor_path = Path(monitor_dir)
        monitor_path.mkdir(parents=True, exist_ok=True)
        monitor_file = str(monitor_path / "monitor.csv")

    return Monitor(env, filename=monitor_file, info_keywords=monitor_info_keywords)


def make_humanoid_vec_env(
    env_id: str = "Humanoid-v5",
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    monitor_dir: str | Path | None = None,
    reward_config_path: str | Path | None = None,
    curriculum_config_path: str | Path | None = None,
    curriculum_stage: str | None = None,
) -> DummyVecEnv:
    """Create a single-process vectorized Humanoid environment for SB3."""

    def _make_env() -> gym.Env:
        return make_humanoid_env(
            env_id,
            seed=seed,
            render_mode=render_mode,
            monitor_dir=monitor_dir,
            reward_config_path=reward_config_path,
            curriculum_config_path=curriculum_config_path,
            curriculum_stage=curriculum_stage,
        )

    return DummyVecEnv([_make_env])


def make_vecnormalize_env(
    env_id: str = "Humanoid-v5",
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    monitor_dir: str | Path | None = None,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    training: bool = True,
    reward_config_path: str | Path | None = None,
    curriculum_config_path: str | Path | None = None,
    curriculum_stage: str | None = None,
) -> VecNormalize:
    """Create a VecNormalize-wrapped Humanoid environment."""
    env = make_humanoid_vec_env(
        env_id,
        seed=seed,
        render_mode=render_mode,
        monitor_dir=monitor_dir,
        reward_config_path=reward_config_path,
        curriculum_config_path=curriculum_config_path,
        curriculum_stage=curriculum_stage,
    )
    return VecNormalize(
        env,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        training=training,
    )


def load_vecnormalize_env(
    stats_path: str | Path,
    env_id: str = "Humanoid-v5",
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    monitor_dir: str | Path | None = None,
    training: bool = False,
    norm_reward: bool = False,
    reward_config_path: str | Path | None = None,
    curriculum_config_path: str | Path | None = None,
    curriculum_stage: str | None = None,
) -> VecNormalize:
    """Load VecNormalize statistics around a fresh Humanoid environment."""
    env = make_humanoid_vec_env(
        env_id,
        seed=seed,
        render_mode=render_mode,
        monitor_dir=monitor_dir,
        reward_config_path=reward_config_path,
        curriculum_config_path=curriculum_config_path,
        curriculum_stage=curriculum_stage,
    )
    vec_env = VecNormalize.load(str(stats_path), env)
    vec_env.training = training
    vec_env.norm_reward = norm_reward
    return vec_env
