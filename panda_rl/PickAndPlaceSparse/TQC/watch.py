"""Watch a trained TQC policy for Franka PickAndPlace."""
import os
import sys
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

import panda_mujoco_gym  # noqa: F401


class BoolDoneWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, bool(terminated), bool(truncated), info


def make_env():
    env = gym.make(
        "FrankaPickAndPlaceSparse-v0",
        reward_type="sparse",
        max_episode_steps=100,
        render_mode="human",
        disable_env_checker=True,
    )
    env = BoolDoneWrapper(env)
    return env


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/tqc_pnp_sparse_curriculum_v2/final_model.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = DummyVecEnv([make_env])
    model = TQC.load(args.model, env=env)

    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")

    success_count = 0
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            done = False
            episode_success = False
            step = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = env.step(action)
                done = dones[0]
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                step += 1

                env.render()
                time.sleep(args.sleep)

                if info.get("is_success", False):
                    episode_success = True

            if episode_success:
                success_count += 1

            print(f"Episode {ep + 1:2d}: {'✅' if episode_success else '❌'}  steps={step}")

    finally:
        env.close()

    print(f"\nSuccess rate: {success_count}/{args.episodes} = {success_count/args.episodes*100:.0f}%")


if __name__ == "__main__":
    main()
