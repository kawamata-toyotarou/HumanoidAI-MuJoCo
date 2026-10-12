from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from humanoid_ai.envs import load_vecnormalize_env, make_humanoid_vec_env, make_vecnormalize_env
from humanoid_ai.phase27_reward import PHASE27_INFO_KEYS
from humanoid_ai.phase3_curriculum_reward import PHASE3_INFO_KEYS
from train import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2.5 PPO experiment with optional VecNormalize."
    )
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--checkpoint-dir", default="checkpoints/phase25")
    parser.add_argument("--log-dir", default="runs/tensorboard")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--vec-normalize", action="store_true")
    parser.add_argument(
        "--reward-config",
        default=None,
        help="Optional Phase 2.7 reward config JSON. When set, uses the custom reward wrapper.",
    )
    parser.add_argument(
        "--curriculum-config",
        default=None,
        help="Optional Phase 3 curriculum config JSON. Mutually exclusive with --reward-config.",
    )
    parser.add_argument(
        "--curriculum-stage",
        default=None,
        help="Stage name inside --curriculum-config, for example stage1, stage2, or stage3.",
    )
    parser.add_argument("--resume", default=None, help="Path to a saved PPO .zip model.")
    parser.add_argument(
        "--vec-normalize-stats",
        default=None,
        help="Path to VecNormalize .pkl stats when resuming a normalized model.",
    )
    parser.add_argument("--clip-obs", type=float, default=10.0)
    parser.add_argument(
        "--final-name-prefix",
        default=None,
        help="Prefix for final model/stat files. Defaults to run-name.",
    )
    return parser.parse_args()


def optimizer_state_entries(model: PPO) -> int:
    return len(model.policy.optimizer.state_dict().get("state", {}))


class RewardInfoTensorboardCallback(BaseCallback):
    def __init__(self, info_keys: tuple[str, ...]) -> None:
        super().__init__()
        self.info_keys = info_keys

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        for key in self.info_keys:
            values = [float(info[key]) for info in infos if key in info]
            if values:
                prefix = "phase3" if key.startswith("phase3_") else "phase27"
                self.logger.record(f"{prefix}/{key}", sum(values) / len(values))
        return True


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be greater than 0")
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError("--torch-threads must be greater than 0")
        torch.set_num_threads(args.torch_threads)

    run_name = args.run_name or datetime.now().strftime("phase25_%Y%m%d_%H%M%S")
    final_name_prefix = args.final_name_prefix or run_name
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = Path(args.log_dir)
    monitor_dir = Path("runs") / "monitor" / run_name
    device = choose_device(args.device)

    set_random_seed(args.seed)
    resume_path = Path(args.resume) if args.resume else None
    vecnormalize_stats_path = Path(args.vec_normalize_stats) if args.vec_normalize_stats else None
    reward_config_path = Path(args.reward_config) if args.reward_config else None
    curriculum_config_path = Path(args.curriculum_config) if args.curriculum_config else None
    if reward_config_path is not None and curriculum_config_path is not None:
        raise ValueError("--reward-config and --curriculum-config are mutually exclusive")
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume model not found: {resume_path}")
    if vecnormalize_stats_path is not None and not vecnormalize_stats_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found: {vecnormalize_stats_path}")
    if reward_config_path is not None and not reward_config_path.exists():
        raise FileNotFoundError(f"Reward config not found: {reward_config_path}")
    if curriculum_config_path is not None and not curriculum_config_path.exists():
        raise FileNotFoundError(f"Curriculum config not found: {curriculum_config_path}")
    if resume_path is not None and args.vec_normalize and vecnormalize_stats_path is None:
        raise ValueError("--vec-normalize-stats is required when resuming with --vec-normalize")

    if args.vec_normalize and vecnormalize_stats_path is not None:
        env = load_vecnormalize_env(
            vecnormalize_stats_path,
            args.env_id,
            seed=args.seed,
            monitor_dir=monitor_dir,
            training=True,
            norm_reward=True,
            reward_config_path=reward_config_path,
            curriculum_config_path=curriculum_config_path,
            curriculum_stage=args.curriculum_stage,
        )
    elif args.vec_normalize:
        env = make_vecnormalize_env(
            args.env_id,
            seed=args.seed,
            monitor_dir=monitor_dir,
            norm_obs=True,
            norm_reward=True,
            clip_obs=args.clip_obs,
            training=True,
            reward_config_path=reward_config_path,
            curriculum_config_path=curriculum_config_path,
            curriculum_stage=args.curriculum_stage,
        )
    else:
        env = make_humanoid_vec_env(
            args.env_id,
            seed=args.seed,
            monitor_dir=monitor_dir,
            reward_config_path=reward_config_path,
            curriculum_config_path=curriculum_config_path,
            curriculum_stage=args.curriculum_stage,
        )

    print(f"env_id={args.env_id}")
    print(f"run_name={run_name}")
    print(f"total_timesteps={args.total_timesteps}")
    print(f"device={device}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_threads={torch.get_num_threads()}")
    print(f"torch_interop_threads={torch.get_num_interop_threads()}")
    print(f"vec_normalize={args.vec_normalize}")
    print(f"resume={resume_path.resolve() if resume_path else None}")
    print(f"vec_normalize_stats={vecnormalize_stats_path.resolve() if vecnormalize_stats_path else None}")
    print(f"reward_config={reward_config_path.resolve() if reward_config_path else None}")
    print(f"curriculum_config={curriculum_config_path.resolve() if curriculum_config_path else None}")
    print(f"curriculum_stage={args.curriculum_stage}")
    print(f"checkpoint_dir={checkpoint_dir.resolve()}")
    print(f"tensorboard_log={tensorboard_dir.resolve()}")
    print("rendering=disabled")

    if resume_path is not None:
        model = PPO.load(
            resume_path,
            env=env,
            device=device,
            tensorboard_log=str(tensorboard_dir),
        )
        reset_num_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            device=device,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
        )
        reset_num_timesteps = True

    print(f"model_start_num_timesteps={model.num_timesteps}")
    print(f"optimizer_state_entries={optimizer_state_entries(model)}")
    print(f"vecnormalize_training={getattr(env, 'training', None)}")
    print(f"vecnormalize_norm_reward={getattr(env, 'norm_reward', None)}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{run_name}_ppo",
        save_replay_buffer=False,
        save_vecnormalize=args.vec_normalize,
    )
    callbacks = [checkpoint_callback]
    if curriculum_config_path is not None:
        callbacks.append(RewardInfoTensorboardCallback(PHASE3_INFO_KEYS))
    elif reward_config_path is not None:
        callbacks.append(RewardInfoTensorboardCallback(PHASE27_INFO_KEYS))

    try:
        start_num_timesteps = int(getattr(model, "num_timesteps", 0))
        start_time = time.perf_counter()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=run_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        elapsed_seconds = time.perf_counter() - start_time
        trained_this_run = int(model.num_timesteps) - start_num_timesteps
        steps_per_second = trained_this_run / elapsed_seconds if elapsed_seconds > 0 else 0.0

        latest_path = checkpoint_dir / f"{final_name_prefix}_latest"
        steps_path = checkpoint_dir / f"{final_name_prefix}_steps_{model.num_timesteps}"
        model.save(latest_path)
        model.save(steps_path)
        print(f"saved_latest={latest_path.with_suffix('.zip')}")
        print(f"saved_steps={steps_path.with_suffix('.zip')}")

        if args.vec_normalize:
            vecnormalize_path = checkpoint_dir / f"{final_name_prefix}_vecnormalize.pkl"
            env.save(str(vecnormalize_path))
            print(f"saved_vecnormalize={vecnormalize_path}")

        print(f"trained_timesteps={model.num_timesteps}")
        print(f"trained_this_run={trained_this_run}")
        print(f"elapsed_seconds={elapsed_seconds:.3f}")
        print(f"steps_per_second={steps_per_second:.3f}")
        print("phase25_train_done=true")
    finally:
        env.close()


if __name__ == "__main__":
    main()
