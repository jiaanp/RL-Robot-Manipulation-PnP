import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.her import HerReplayBuffer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import panda_mujoco_gym  # noqa: E402,F401 - registers Franka environments


class SuccessMetricCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "is_success" in info:
                self.logger.record_mean("rollout/is_success", float(info["is_success"]))
        return True


class TaskCurriculumCallback(BaseCallback):
    PHASES = (
        (0, dict(obj_xy_range=0.08, goal_xy_range=0.12, goal_z_range=0.08, distance_threshold=0.05)),
        (80_000, dict(obj_xy_range=0.12, goal_xy_range=0.16, goal_z_range=0.10, distance_threshold=0.05)),
        (180_000, dict(obj_xy_range=0.18, goal_xy_range=0.20, goal_z_range=0.14, distance_threshold=0.05)),
        (350_000, dict(obj_xy_range=0.22, goal_xy_range=0.22, goal_z_range=0.16, distance_threshold=0.05)),
        (600_000, dict(obj_xy_range=0.30, goal_xy_range=0.30, goal_z_range=0.20, distance_threshold=0.05)),
    )

    def __init__(self):
        super().__init__()
        self.phase_index = -1

    def _on_training_start(self) -> None:
        self._maybe_update_phase(force=True)

    def _on_step(self) -> bool:
        self._maybe_update_phase()
        return True

    def _maybe_update_phase(self, force: bool = False) -> None:
        idx = 0
        for candidate_idx, (start_step, _) in enumerate(self.PHASES):
            if self.num_timesteps >= start_step:
                idx = candidate_idx
        if not force and idx == self.phase_index:
            return
        self.phase_index = idx
        _, kwargs = self.PHASES[idx]
        self.training_env.env_method("set_task_difficulty", **kwargs)
        print(f"[curriculum] step={self.num_timesteps} {kwargs}")


class BoolDoneWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, bool(terminated), bool(truncated), info


def make_env(rank: int, args):
    def _init():
        env = gym.make(
            "FrankaPickAndPlaceSparse-v0",
            reward_type="sparse",
            max_episode_steps=args.max_episode_steps,
            n_substeps=args.n_substeps,
            disable_env_checker=True,
        )
        env = BoolDoneWrapper(env)
        env.reset(seed=args.seed + rank)
        return Monitor(env)

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], default="subproc")
    parser.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], default="fork")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--n-substeps", type=int, default=15)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-dir", default="runs/tqc_pnp_sparse_opt")
    parser.add_argument("--load-model", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        import tensorboard  # noqa: F401

        tensorboard_log = str(save_dir / "tensorboard")
    except ImportError:
        tensorboard_log = None

    vec_cls = SubprocVecEnv if args.vec_env == "subproc" and args.n_envs > 1 else DummyVecEnv
    if vec_cls is SubprocVecEnv:
        env = vec_cls([make_env(i, args) for i in range(args.n_envs)], start_method=args.start_method)
    else:
        env = vec_cls([make_env(i, args) for i in range(args.n_envs)])
    eval_args = argparse.Namespace(**vars(args))
    eval_args.n_substeps = 25
    eval_env = DummyVecEnv([make_env(10_000 + i, eval_args) for i in range(1)])

    policy_kwargs = dict(net_arch=[256, 256, 256], n_critics=2)
    replay_buffer_kwargs = dict(
        n_sampled_goal=args.n_sampled_goal,
        goal_selection_strategy="future",
    )

    if args.load_model:
        model = TQC.load(args.load_model, env=env, device=args.device)
    else:
        model = TQC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            policy_kwargs=policy_kwargs,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            verbose=1,
            top_quantiles_to_drop_per_net=2,
            tensorboard_log=tensorboard_log,
            device=args.device,
            seed=args.seed,
        )

    eval_freq = max(args.eval_freq // max(args.n_envs, 1), 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    print("=== Optimized Franka PickAndPlaceSparse TQC+HER ===")
    print(f"panda_mujoco_gym={panda_mujoco_gym.__file__}")
    print(
        f"timesteps={args.timesteps:,} n_envs={args.n_envs} "
        f"vec_env={args.vec_env} max_episode_steps={args.max_episode_steps} "
        f"n_substeps={args.n_substeps} device={args.device}"
    )
    print(
        f"batch={args.batch_size} lr={args.learning_rate} "
        f"learning_starts={args.learning_starts} eval_freq={args.eval_freq}"
    )

    try:
        callbacks = [SuccessMetricCallback(), eval_callback]
        if args.curriculum:
            callbacks.insert(0, TaskCurriculumCallback())

        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name="tqc_pnp_sparse_opt",
            progress_bar=args.progress_bar,
        )
        final_path = save_dir / "final_model"
        model.save(str(final_path))
        print(f"Saved final model to {final_path}.zip")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
