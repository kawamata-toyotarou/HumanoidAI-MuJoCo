from __future__ import annotations

import argparse
import time

import gymnasium as gym


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gymnasium Humanoid-v5 with random joint actions.",
    )
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array", "none"),
        default="human",
        help="Use human for a 3D window, rgb_array for offscreen render, or none for simulation only.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Extra delay per step in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = gym.make(args.env_id, render_mode=render_mode)

    obs, info = env.reset(seed=args.seed)
    print(f"env_id={args.env_id}")
    print(f"render_mode={args.render_mode}")
    print(f"observation_shape={getattr(obs, 'shape', None)}")
    print(f"action_shape={env.action_space.shape}")

    total_reward = 0.0
    resets = 0
    last_frame_shape = None

    try:
        for step in range(1, args.steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if render_mode == "rgb_array":
                frame = env.render()
                last_frame_shape = getattr(frame, "shape", None)

            if terminated or truncated:
                resets += 1
                obs, info = env.reset()

            if args.sleep > 0:
                time.sleep(args.sleep)

            if step == 1 or step % 100 == 0 or step == args.steps:
                print(
                    f"step={step} total_reward={total_reward:.3f} "
                    f"terminated={terminated} truncated={truncated} resets={resets}"
                )
    finally:
        env.close()

    if last_frame_shape is not None:
        print(f"last_rgb_frame_shape={last_frame_shape}")

    print("phase1_random_humanoid_done=true")


if __name__ == "__main__":
    main()
