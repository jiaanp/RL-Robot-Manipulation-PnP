"""优化奖励函数版训练 — 解决成功率卡在 ~70% 的问题

奖励函数改动 (3 处)：
1. push 激活从硬门槛改成指数平滑 → 不再需要"正好贴到 12cm"才给信号
2. 新增 proximity_bonus → 方块靠近目标就给分，填补"推动→成功"之间的空白
3. push 权重 15→25, 动作惩罚减半 → 鼓励更积极地推动
"""

import os
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

from rizon4_push import (
    XarmEnv, PushMetricsCallback, delete_flag_file,
    write_flag_file, check_flag_file,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="stable_baselines3.common.on_policy_algorithm",
)


class RewardOptimizedEnv(XarmEnv):
    """继承原环境，只覆盖 _calc_reward 来优化奖励函数"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 覆盖奖励参数
        self.push_reward_activation_dist = 0.18  # 从 0.12 放宽到 0.18
        self.push_reward_weight = 25.0            # 从 15 提高到 25
        self.action_penalty_weight = 0.005        # 从 0.01 减半
        self.proximity_bonus_threshold = 0.12     # 新增：方块进入 12cm 范围开始给额外分
        self.proximity_bonus_weight = 8.0         # 新增：接近奖励权重

    def _calc_reward(
        self,
        ee_pos: np.ndarray,
        cube_pos: np.ndarray,
        cube_vel: np.ndarray,
        action: np.ndarray,
        step_count: int,
    ):
        """优化后的奖励函数

        Rt = Rreach + Rpush_smooth + Rproximity + Rsuccess - Raction - Rvel - Rcollision

        改动点:
        - Rpush 改为指数平滑激活，不再用硬门槛
        - 新增 Rproximity: 方块靠近目标就给分
        - 动作惩罚减半，鼓励更积极的动作
        """
        ee_cube_dist = float(np.linalg.norm(cube_pos - ee_pos))
        prev_cube_goal_dist = float(np.linalg.norm(self.goal - self.prev_cube_pos))
        cube_goal_dist = float(np.linalg.norm(self.goal - cube_pos))
        cube_speed = float(np.linalg.norm(cube_vel))
        manipulability, singularity_penalty = self._get_singularity_penalty()

        # --- Rreach: 接近预推点（和原来一样）---
        prev_pre_push_point = self._get_pre_push_point(self.prev_cube_pos)
        prev_ee_prepush_dist = float(np.linalg.norm(prev_pre_push_point - self.prev_ee_pos))
        pre_push_point = self._get_pre_push_point(cube_pos)
        ee_prepush_dist = float(np.linalg.norm(pre_push_point - ee_pos))
        reach_reward = self.reach_reward_weight * (prev_ee_prepush_dist - ee_prepush_dist)

        # --- Rpush: 指数平滑激活（核心改动 ①）---
        # 原来: if ee_cube_dist <= 0.12: push_reward = ...
        # 现在: 平滑过渡，末端离方块越近，push 信号越强
        push_alpha = float(np.exp(-ee_cube_dist / 0.06))
        # ee_cube_dist=0.05m → alpha=0.43 | 0.10m→0.19 | 0.15m→0.08 | 0.20m→0.04
        push_reward = self.push_reward_weight * push_alpha * (prev_cube_goal_dist - cube_goal_dist)

        # --- Rproximity: 接近目标额外奖励（核心改动 ②）---
        # 填补"方块在路上一半"和"成功"之间的空白
        proximity_bonus = 0.0
        if cube_goal_dist < self.proximity_bonus_threshold:
            # 越近越多，0.12m 时 ≈ 0, 0.05m 时 ≈ 0.56
            proximity_bonus = self.proximity_bonus_weight * (
                self.proximity_bonus_threshold - cube_goal_dist
            )

        # --- Raction: 减半（改动 ③）---
        action_penalty = -self.action_penalty_weight * float(np.sum(np.square(action)))

        # --- Rvel: 和原来一样 ---
        velocity_penalty = -self.velocity_penalty_weight * cube_speed * float(
            np.exp(-self.velocity_penalty_decay * cube_goal_dist)
        )

        # --- Rsuccess: 和原来一样 ---
        success_bonus = 0.0
        terminated = False
        if cube_goal_dist <= self.goal_threshold and cube_speed <= self.goal_speed_threshold:
            terminated = True
            success_bonus = self.success_bonus_value

        total_reward = (
            reach_reward
            + push_reward
            + proximity_bonus
            + action_penalty
            + velocity_penalty
            + success_bonus
            - 0.05 * self._count_non_cube_contacts()
        )

        truncated = step_count >= self.max_steps and not terminated
        return (
            np.float32(total_reward),
            terminated,
            cube_goal_dist,
            cube_speed,
            truncated,
            manipulability,
            singularity_penalty,
        )


def main():
    # ========== 路径配置 ==========
    MODEL_SAVE_PATH = "runs/rizon4_push_v6/rizon4_pos_push_v6"
    TENSORBOARD_LOG = "runs/rizon4_push_v6/tensorboard/"
    BEST_MODEL_DIR = "runs/rizon4_push_v6/best_model"
    EVAL_LOG_DIR = "runs/rizon4_push_v6/eval_logs"

    N_ENVS = 32
    TOTAL_TIMESTEPS = 6_000_000  # 奖励更密集，不需要跑那么久
    MODEL_XML_PATH = "description/mjcf/scene.xml"

    # ========== 创建环境 ==========
    env_kwargs = {
        "visualize": False,
        "obs_noise_scale": 0.001,
        "model_xml_path": MODEL_XML_PATH,
        "frame_skip": 5,
        "delta_scale": 0.05,
        "max_steps": 200,
    }

    env = make_vec_env(
        env_id=lambda: RewardOptimizedEnv(**env_kwargs),
        n_envs=N_ENVS,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )

    eval_env = Monitor(
        XarmEnv(
            visualize=False,
            max_steps=300,
            obs_noise_scale=0.0,
            model_xml_path=MODEL_XML_PATH,
            frame_skip=5,
            delta_scale=0.05,
        )
    )

    for d in [os.path.dirname(MODEL_SAVE_PATH), TENSORBOARD_LOG, BEST_MODEL_DIR, EVAL_LOG_DIR]:
        if d:
            os.makedirs(d, exist_ok=True)

    eval_freq = max(65536 // max(N_ENVS, 1), 1)
    callback = CallbackList([
        PushMetricsCallback(),
        EvalCallback(
            eval_env=eval_env,
            best_model_save_path=BEST_MODEL_DIR,
            log_path=EVAL_LOG_DIR,
            eval_freq=eval_freq,
            n_eval_episodes=20,
            deterministic=True,
            render=False,
            verbose=1,
            warn=False,
        ),
    ])

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=2048,
        batch_size=2048,
        n_epochs=8,
        gamma=0.995,
        gae_lambda=0.98,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.005,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=TENSORBOARD_LOG,
    )

    print(f"=== 优化奖励函数训练 ===")
    print(f"改动: push平滑激活 | proximity_bonus | push权重15→25 | 动作惩罚减半")
    print(f"并行环境: {N_ENVS} | 总步数: {TOTAL_TIMESTEPS:,}")
    print(f"TensorBoard: {TENSORBOARD_LOG}")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True,
        )
        model.save(MODEL_SAVE_PATH)
        print(f"模型已保存至: {MODEL_SAVE_PATH}")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    delete_flag_file()
    main()
