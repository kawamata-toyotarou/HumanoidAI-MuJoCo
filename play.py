from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
from stable_baselines3 import PPO

from humanoid_ai.envs import load_vecnormalize_env, make_humanoid_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained PPO model or random Humanoid-v5 actions.")
    parser.add_argument("--env-id", default="Humanoid-v5")
    parser.add_argument("--agent", choices=("model", "random"), default="model")
    parser.add_argument("--model", default="checkpoints/humanoid_ppo_latest.zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--vec-normalize-stats",
        default=None,
        help="Path to VecNormalize .pkl stats for normalized models.",
    )
    parser.add_argument(
        "--reward-config",
        default=None,
        help="Optional Phase 2.7 reward config JSON for custom-reward models.",
    )
    parser.add_argument(
        "--curriculum-config",
        default=None,
        help="Optional Phase 3 curriculum config JSON. Mutually exclusive with --reward-config.",
    )
    parser.add_argument("--curriculum-stage", default=None)
    parser.add_argument(
        "--render-mode",
        choices=("human", "rgb_array", "none"),
        default="human",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--video",
        default=None,
        help="Optional output video path. Requires --render-mode rgb_array. GIF is supported without extra ffmpeg.",
    )
    parser.add_argument("--video-fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video and args.render_mode != "rgb_array":
        raise ValueError("--video requires --render-mode rgb_array")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be greater than 0")
    if args.reward_config and args.curriculum_config:
        raise ValueError("--reward-config and --curriculum-config are mutually exclusive")

    render_mode = None if args.render_mode == "none" else args.render_mode
    if args.vec_normalize_stats:
        if args.agent != "model":
            raise ValueError("--vec-normalize-stats requires --agent model")
        env = load_vecnormalize_env(
            args.vec_normalize_stats,
            args.env_id,
            seed=args.seed,
            render_mode=render_mode,
            training=False,
            norm_reward=False,
            reward_config_path=args.reward_config,
            curriculum_config_path=args.curriculum_config,
            curriculum_stage=args.curriculum_stage,
        )
        vectorized = True
    else:
        env = make_humanoid_env(
            args.env_id,
            seed=args.seed,
            render_mode=render_mode,
            reward_config_path=args.reward_config,
            curriculum_config_path=args.curriculum_config,
            curriculum_stage=args.curriculum_stage,
        )
        vectorized = False

    model = None
    if args.agent == "model":
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = PPO.load(model_path, device=args.device)

    print(f"env_id={args.env_id}")
    print(f"agent={args.agent}")
    print(f"render_mode={args.render_mode}")
    if args.vec_normalize_stats:
        print(f"vec_normalize_stats={Path(args.vec_normalize_stats).resolve()}")
    if args.reward_config:
        print(f"reward_config={Path(args.reward_config).resolve()}")
    if args.curriculum_config:
        print(f"curriculum_config={Path(args.curriculum_config).resolve()}")
        print(f"curriculum_stage={args.curriculum_stage}")
    if model is not None:
        print(f"model={Path(args.model).resolve()}")
    if args.video:
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"video={video_path.resolve()}")
        print(f"video_fps={args.video_fps}")

    try:
        video_writer = None
        if args.video:
            video_writer = imageio.get_writer(args.video, mode="I", fps=args.video_fps)
        for episode in range(1, args.episodes + 1):
            if vectorized:
                env.seed(args.seed + episode - 1)
                obs = env.reset()
            else:
                obs, info = env.reset(seed=args.seed + episode - 1)
            episode_reward = 0.0
            episode_length = 0
            last_frame_shape = None
            terminated = False
            truncated = False

            for step in range(1, args.max_steps + 1):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=args.deterministic)

                if vectorized:
                    obs, rewards, dones, infos = env.step(action)
                    info = infos[0]
                    episode_reward += float(rewards[0])
                    terminated = bool(info.get("terminal_observation") is not None and not info.get("TimeLimit.truncated", False))
                    truncated = bool(info.get("TimeLimit.truncated", False))
                    done = bool(dones[0])
                else:
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += float(reward)
                    done = terminated or truncated
                episode_length = step

                if render_mode == "rgb_array":
                    frame = env.render()
                    last_frame_shape = getattr(frame, "shape", None)
                    if video_writer is not None:
                        video_writer.append_data(frame)

                if args.sleep > 0:
                    time.sleep(args.sleep)

                if done:
                    break

            print(
                f"episode={episode} reward={episode_reward:.3f} "
                f"length={episode_length} terminated={terminated} truncated={truncated}"
            )
            if last_frame_shape is not None:
                print(f"last_rgb_frame_shape={last_frame_shape}")
    finally:
        if "video_writer" in locals() and video_writer is not None:
            video_writer.close()
        env.close()

    print("phase2_play_done=true")


if __name__ == "__main__":
    main()
