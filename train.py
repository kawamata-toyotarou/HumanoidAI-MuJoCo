from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from humanoid_ai.envs import make_humanoid_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Gymnasium Humanoid-v5.")
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--final-name-prefix",
        default="humanoid_ppo",
        help="Prefix for final latest/step model files saved at the end of training.",
    )
    parser.add_argument("--log-dir", default="runs/tensorboard")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="Path to a saved PPO .zip model.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Limit PyTorch CPU threads to reduce heat/noise. Defaults to PyTorch's setting.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but torch.cuda.is_available() is false. Falling back to CPU.")
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be greater than 0")
    if args.torch_threads is not None:
        if args.torch_threads <= 0:
            raise ValueError("--torch-threads must be greater than 0")
        torch.set_num_threads(args.torch_threads)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("humanoid_ppo_%Y%m%d_%H%M%S")
    tensorboard_dir = Path(args.log_dir)
    monitor_dir = Path("runs") / "monitor" / run_name
    device = choose_device(args.device)

    set_random_seed(args.seed)
    env = make_humanoid_env(args.env_id, seed=args.seed, monitor_dir=monitor_dir)

    print(f"env_id={args.env_id}")
    print(f"run_name={run_name}")
    print(f"total_timesteps={args.total_timesteps}")
    print(f"device={device}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_threads={torch.get_num_threads()}")
    print(f"torch_interop_threads={torch.get_num_interop_threads()}")
    print(f"checkpoint_dir={checkpoint_dir.resolve()}")
    print(f"tensorboard_log={tensorboard_dir.resolve()}")
    print("rendering=disabled")

    if args.resume:
        print(f"resume={args.resume}")
        model = PPO.load(args.resume, env=env, device=device, tensorboard_log=str(tensorboard_dir))
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

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq, 1),
        save_path=str(checkpoint_dir),
        name_prefix=f"{run_name}_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    try:
        start_num_timesteps = int(getattr(model, "num_timesteps", 0))
        start_time = time.perf_counter()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            tb_log_name=run_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        elapsed_seconds = time.perf_counter() - start_time
        trained_this_run = int(model.num_timesteps) - start_num_timesteps
        steps_per_second = trained_this_run / elapsed_seconds if elapsed_seconds > 0 else 0.0
        latest_path = checkpoint_dir / f"{args.final_name_prefix}_latest"
        steps_path = checkpoint_dir / f"{args.final_name_prefix}_steps_{model.num_timesteps}"
        model.save(latest_path)
        model.save(steps_path)
        print(f"saved_latest={latest_path.with_suffix('.zip')}")
        print(f"saved_steps={steps_path.with_suffix('.zip')}")
        print(f"trained_timesteps={model.num_timesteps}")
        print(f"trained_this_run={trained_this_run}")
        print(f"elapsed_seconds={elapsed_seconds:.3f}")
        print(f"steps_per_second={steps_per_second:.3f}")
        print("phase2_train_done=true")
    finally:
        env.close()


if __name__ == "__main__":
    main()
