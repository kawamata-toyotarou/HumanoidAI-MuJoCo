from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from train import choose_device


ENV_ID = "Humanoid-v5"
LEARNING_RATE_SCHEDULES = ("constant",)


@dataclass(frozen=True)
class Phase4Preset:
    name: str
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    ent_coef: float
    net_arch: list[int]
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


PRESETS: dict[str, Phase4Preset] = {
    "baseline": Phase4Preset(
        name="baseline",
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        learning_rate=3e-4,
        ent_coef=0.0,
        net_arch=[64, 64],
    ),
    "candidate": Phase4Preset(
        name="candidate",
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        learning_rate=1e-4,
        ent_coef=0.01,
        net_arch=[256, 256],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 standard Humanoid-v5 PPO + VecNormalize trainer.")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--total-timesteps", type=int, required=True, help="New env steps to train in this run.")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", default=None, help="Existing PPO .zip to continue from.")
    parser.add_argument("--vec-normalize-stats", default=None, help="Matching VecNormalize .pkl for --resume.")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint frequency in env-step units.")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override preset learning rate. Useful for controlled resume experiments.",
    )
    parser.add_argument("--log-dir", default="runs/tensorboard")
    parser.add_argument("--monitor-dir", default="runs/monitor_phase4")
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument("--learning-rate-schedule", choices=LEARNING_RATE_SCHEDULES, default="constant")
    return parser.parse_args()


def make_standard_humanoid_env(env_id: str, seed: int, monitor_file: Path | None) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    if monitor_file is not None:
        monitor_file.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_file))
    return env


def make_env_factory(env_id: str, seed: int, monitor_dir: Path, rank: int):
    def _init() -> gym.Env:
        return make_standard_humanoid_env(
            env_id,
            seed + rank,
            monitor_dir / f"env_{rank:02d}" / "monitor.csv",
        )

    return _init


def make_phase4_vec_env(
    *,
    n_envs: int,
    seed: int,
    monitor_dir: Path,
    vecnormalize_stats: Path | None,
    clip_obs: float,
) -> VecNormalize:
    env_fns = [make_env_factory(ENV_ID, seed, monitor_dir, rank) for rank in range(n_envs)]
    if n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
        vec_env_kind = "DummyVecEnv"
    else:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        vec_env_kind = "SubprocVecEnv"

    if vecnormalize_stats is None:
        normalized = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=clip_obs,
            training=True,
        )
    else:
        normalized = VecNormalize.load(str(vecnormalize_stats), vec_env)
        normalized.training = True
        normalized.norm_reward = True
    setattr(normalized, "_phase4_vec_env_kind", vec_env_kind)
    return normalized


def metadata_path_for_model(model_path: Path) -> Path:
    if model_path.suffix == ".zip":
        return model_path.with_name(f"{model_path.stem}_metadata.json")
    return model_path.with_name(f"{model_path.name}_metadata.json")


def optimizer_state_entries(model: PPO) -> int:
    return len(model.policy.optimizer.state_dict().get("state", {}))


def resolve_learning_rate(args: argparse.Namespace, preset: Phase4Preset) -> float:
    return float(args.learning_rate) if args.learning_rate is not None else float(preset.learning_rate)


def optimizer_learning_rates(model: PPO) -> list[float]:
    return [float(group["lr"]) for group in model.policy.optimizer.param_groups]


def apply_model_learning_rate(model: PPO, learning_rate: float) -> None:
    """Update both SB3's schedule and the already-created optimizer param groups."""
    model.learning_rate = float(learning_rate)
    model.lr_schedule = FloatSchedule(float(learning_rate))
    for group in model.policy.optimizer.param_groups:
        group["lr"] = float(learning_rate)


def policy_net_arch(model: PPO) -> list[int] | dict[str, Any] | None:
    value = getattr(model.policy, "net_arch", None)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if isinstance(value, dict):
        return value
    return None


def space_summary(env: VecNormalize) -> dict[str, Any]:
    obs_space = env.observation_space
    action_space = env.action_space
    return {
        "observation_space": str(obs_space),
        "observation_shape": list(obs_space.shape or ()),
        "observation_dtype": str(getattr(obs_space, "dtype", "")),
        "action_space": str(action_space),
        "action_shape": list(action_space.shape or ()),
        "action_dtype": str(getattr(action_space, "dtype", "")),
        "action_low_min": float(action_space.low.min()),
        "action_high_max": float(action_space.high.max()),
    }


def load_existing_metadata(resume_path: Path) -> dict[str, Any] | None:
    metadata_path = metadata_path_for_model(resume_path)
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return None


def validate_resume_metadata(
    *,
    args: argparse.Namespace,
    preset: Phase4Preset,
    existing: dict[str, Any] | None,
) -> None:
    if existing is None:
        print("resume_metadata=None")
        return
    checks = {
        "preset": preset.name,
        "env_id": ENV_ID,
        "n_envs": args.n_envs,
        "seed": args.seed,
    }
    for key, expected in checks.items():
        actual = existing.get(key)
        if actual != expected:
            raise ValueError(f"Resume metadata mismatch for {key}: expected {expected!r}, got {actual!r}")
    existing_preset = existing.get("preset_config", {})
    if existing_preset.get("net_arch") != preset.net_arch:
        raise ValueError("Resume metadata net_arch does not match selected preset")
    print(f"resume_metadata={metadata_path_for_model(Path(args.resume)).resolve()}")


def validate_model_against_env(model: PPO, env: VecNormalize, preset: Phase4Preset) -> None:
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(f"Observation space mismatch: model={model.observation_space}, env={env.observation_space}")
    if model.action_space.shape != env.action_space.shape:
        raise ValueError(f"Action space mismatch: model={model.action_space}, env={env.action_space}")
    actual_arch = policy_net_arch(model)
    if actual_arch is not None and actual_arch != preset.net_arch:
        raise ValueError(f"Policy net_arch mismatch: model={actual_arch}, preset={preset.net_arch}")


def build_metadata(
    *,
    args: argparse.Namespace,
    preset: Phase4Preset,
    model: PPO,
    env: VecNormalize,
    run_kind: str,
    start_num_timesteps: int,
    elapsed_seconds: float | None = None,
    trained_this_run: int | None = None,
    saved_model: Path | None = None,
    saved_vecnormalize: Path | None = None,
) -> dict[str, Any]:
    metadata = {
        "phase": "phase4",
        "run_kind": run_kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env_id": ENV_ID,
        "preset": preset.name,
        "preset_config": asdict(preset),
        "learning_rate_preset": preset.learning_rate,
        "learning_rate_effective": resolve_learning_rate(args, preset),
        "learning_rate_override": args.learning_rate,
        "optimizer_learning_rates": optimizer_learning_rates(model),
        "learning_rate_schedule": args.learning_rate_schedule,
        "learning_rate_schedule_note": "Only constant learning rates are used in the first comparison; schedule support is intentionally reserved.",
        "run_name": args.run_name,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "vec_env_kind": getattr(env, "_phase4_vec_env_kind", type(env.venv).__name__),
        "device": str(model.device),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gym.__version__,
        "total_timesteps_argument": args.total_timesteps,
        "save_freq_env_steps": args.save_freq,
        "model_num_timesteps": int(model.num_timesteps),
        "model_start_num_timesteps": start_num_timesteps,
        "trained_this_run": trained_this_run,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": (trained_this_run / elapsed_seconds) if elapsed_seconds and trained_this_run else None,
        "optimizer_state_entries": optimizer_state_entries(model),
        "policy_net_arch": policy_net_arch(model),
        "policy_class": type(model.policy).__name__,
        "resume": str(Path(args.resume).resolve()) if args.resume else None,
        "vec_normalize_stats": str(Path(args.vec_normalize_stats).resolve()) if args.vec_normalize_stats else None,
        "vecnormalize_training": getattr(env, "training", None),
        "vecnormalize_norm_reward": getattr(env, "norm_reward", None),
        "saved_model": str(saved_model) if saved_model else None,
        "saved_vecnormalize": str(saved_vecnormalize) if saved_vecnormalize else None,
        **space_summary(env),
        "sb3_defaults_explicitly_recorded": {
            "gamma": preset.gamma,
            "gae_lambda": preset.gae_lambda,
            "clip_range": preset.clip_range,
            "vf_coef": preset.vf_coef,
            "max_grad_norm": preset.max_grad_norm,
            "normalize_advantage": True,
            "target_kl": None,
        },
    }
    return metadata


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def save_training_artifacts(
    *,
    model: PPO,
    env: VecNormalize,
    checkpoint_dir: Path,
    name: str,
    args: argparse.Namespace,
    preset: Phase4Preset,
    run_kind: str,
    start_num_timesteps: int,
    elapsed_seconds: float | None = None,
    trained_this_run: int | None = None,
) -> tuple[Path, Path, Path]:
    model_path = checkpoint_dir / f"{name}.zip"
    vec_path = checkpoint_dir / f"{name}_vecnormalize.pkl"
    model.save(model_path)
    env.save(str(vec_path))
    metadata = build_metadata(
        args=args,
        preset=preset,
        model=model,
        env=env,
        run_kind=run_kind,
        start_num_timesteps=start_num_timesteps,
        elapsed_seconds=elapsed_seconds,
        trained_this_run=trained_this_run,
        saved_model=model_path,
        saved_vecnormalize=vec_path,
    )
    metadata_path = metadata_path_for_model(model_path)
    write_metadata(metadata_path, metadata)
    return model_path, vec_path, metadata_path


class Phase4CheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        run_name: str,
        save_freq_env_steps: int,
        args: argparse.Namespace,
        preset: Phase4Preset,
        start_num_timesteps: int,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.save_freq_env_steps = max(save_freq_env_steps, 1)
        self.args = args
        self.preset = preset
        self.start_num_timesteps = start_num_timesteps
        self.last_saved = start_num_timesteps

    def _on_step(self) -> bool:
        if int(self.model.num_timesteps) - self.last_saved < self.save_freq_env_steps:
            return True
        step = int(self.model.num_timesteps)
        name = f"{self.run_name}_steps_{step}"
        model_path, vec_path, metadata_path = save_training_artifacts(
            model=self.model,
            env=self.training_env,
            checkpoint_dir=self.checkpoint_dir,
            name=name,
            args=self.args,
            preset=self.preset,
            run_kind="checkpoint",
            start_num_timesteps=self.start_num_timesteps,
        )
        print(f"saved_checkpoint_model={model_path}")
        print(f"saved_checkpoint_vecnormalize={vec_path}")
        print(f"saved_checkpoint_metadata={metadata_path}")
        self.last_saved = step
        return True


def validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be greater than 0")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be greater than 0")
    if args.save_freq <= 0:
        raise ValueError("--save-freq must be greater than 0")
    if args.torch_threads is not None and args.torch_threads <= 0:
        raise ValueError("--torch-threads must be greater than 0")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")
    if args.resume and not args.vec_normalize_stats:
        raise ValueError("--vec-normalize-stats is required when using --resume")
    if args.vec_normalize_stats and not args.resume:
        raise ValueError("--vec-normalize-stats is only valid together with --resume")


def main() -> None:
    args = parse_args()
    validate_args(args)
    preset = PRESETS[args.preset]
    effective_learning_rate = resolve_learning_rate(args, preset)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = Path(args.monitor_dir) / args.run_name
    tensorboard_dir = Path(args.log_dir)
    resume_path = Path(args.resume) if args.resume else None
    vec_stats_path = Path(args.vec_normalize_stats) if args.vec_normalize_stats else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume model not found: {resume_path}")
    if vec_stats_path is not None and not vec_stats_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found: {vec_stats_path}")

    set_random_seed(args.seed)
    env = make_phase4_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=monitor_dir,
        vecnormalize_stats=vec_stats_path,
        clip_obs=args.clip_obs,
    )
    device = choose_device(args.device)

    try:
        if resume_path is not None:
            validate_resume_metadata(args=args, preset=preset, existing=load_existing_metadata(resume_path))
            model = PPO.load(
                resume_path,
                env=env,
                device=device,
                tensorboard_log=str(tensorboard_dir),
            )
            apply_model_learning_rate(model, effective_learning_rate)
            reset_num_timesteps = False
            validate_model_against_env(model, env, preset)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=str(tensorboard_dir),
                device=device,
                seed=args.seed,
                n_steps=preset.n_steps,
                batch_size=preset.batch_size,
                n_epochs=preset.n_epochs,
                learning_rate=effective_learning_rate,
                ent_coef=preset.ent_coef,
                gamma=preset.gamma,
                gae_lambda=preset.gae_lambda,
                clip_range=preset.clip_range,
                vf_coef=preset.vf_coef,
                max_grad_norm=preset.max_grad_norm,
                policy_kwargs={"net_arch": preset.net_arch},
            )
            reset_num_timesteps = True

        start_num_timesteps = int(model.num_timesteps)
        print(f"env_id={ENV_ID}")
        print(f"preset={preset.name}")
        print(f"preset_config={asdict(preset)}")
        print(f"learning_rate_preset={preset.learning_rate}")
        print(f"learning_rate_effective={effective_learning_rate}")
        print(f"optimizer_learning_rates={optimizer_learning_rates(model)}")
        print(f"run_name={args.run_name}")
        print(f"total_timesteps_additional={args.total_timesteps}")
        print(f"seed={args.seed}")
        print(f"n_envs={args.n_envs}")
        print(f"vec_env_kind={getattr(env, '_phase4_vec_env_kind', type(env.venv).__name__)}")
        print(f"device={device}")
        print(f"torch_threads={torch.get_num_threads()}")
        print(f"torch_interop_threads={torch.get_num_interop_threads()}")
        print(f"model_start_num_timesteps={start_num_timesteps}")
        print(f"optimizer_state_entries={optimizer_state_entries(model)}")
        print(f"policy_net_arch={policy_net_arch(model)}")
        print(f"vecnormalize_training={getattr(env, 'training', None)}")
        print(f"vecnormalize_norm_reward={getattr(env, 'norm_reward', None)}")
        print(f"observation_space={env.observation_space}")
        print(f"action_space={env.action_space}")
        print("rendering=disabled")

        initial_metadata = build_metadata(
            args=args,
            preset=preset,
            model=model,
            env=env,
            run_kind="start",
            start_num_timesteps=start_num_timesteps,
        )
        write_metadata(checkpoint_dir / f"{args.run_name}_start_metadata.json", initial_metadata)

        callback = Phase4CheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            run_name=args.run_name,
            save_freq_env_steps=args.save_freq,
            args=args,
            preset=preset,
            start_num_timesteps=start_num_timesteps,
        )
        start_time = time.perf_counter()
        interrupted = False
        try:
            model.learn(
                total_timesteps=args.total_timesteps,
                callback=callback,
                tb_log_name=args.run_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("phase4_keyboard_interrupt=true")
        elapsed_seconds = time.perf_counter() - start_time
        trained_this_run = int(model.num_timesteps) - start_num_timesteps
        suffix = "interrupt" if interrupted else "latest"
        latest_name = f"{args.run_name}_{suffix}"
        model_path, vec_path, metadata_path = save_training_artifacts(
            model=model,
            env=env,
            checkpoint_dir=checkpoint_dir,
            name=latest_name,
            args=args,
            preset=preset,
            run_kind=suffix,
            start_num_timesteps=start_num_timesteps,
            elapsed_seconds=elapsed_seconds,
            trained_this_run=trained_this_run,
        )
        steps_name = f"{args.run_name}_steps_{int(model.num_timesteps)}"
        steps_model_path, steps_vec_path, steps_metadata_path = save_training_artifacts(
            model=model,
            env=env,
            checkpoint_dir=checkpoint_dir,
            name=steps_name,
            args=args,
            preset=preset,
            run_kind="steps",
            start_num_timesteps=start_num_timesteps,
            elapsed_seconds=elapsed_seconds,
            trained_this_run=trained_this_run,
        )
        steps_per_second = trained_this_run / elapsed_seconds if elapsed_seconds > 0 else 0.0
        print(f"saved_latest={model_path}")
        print(f"saved_latest_vecnormalize={vec_path}")
        print(f"saved_latest_metadata={metadata_path}")
        print(f"saved_steps={steps_model_path}")
        print(f"saved_steps_vecnormalize={steps_vec_path}")
        print(f"saved_steps_metadata={steps_metadata_path}")
        print(f"trained_timesteps={int(model.num_timesteps)}")
        print(f"trained_this_run={trained_this_run}")
        print(f"elapsed_seconds={elapsed_seconds:.3f}")
        print(f"steps_per_second={steps_per_second:.3f}")
        print(f"phase4_train_interrupted={str(interrupted).lower()}")
        print("phase4_train_done=true")
    finally:
        env.close()


if __name__ == "__main__":
    main()
