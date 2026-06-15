"""Rizon4 机械臂推方块任务的强化学习环境和训练代码

该脚本实现了一个使用 MuJoCo 物理引擎的强化学习环境，其中 Rizon4 机械臂需要通过推动方块到达指定目标。
主要包括：
- XarmEnv: Gym 环境类，定义了机械臂的物理模型、奖励函数、约束条件等
- PushMetricsCallback: PPO 训练回调，用于记录自定义指标
- train_ppo: 多进程 PPO 训练函数
- test_ppo: 推理测试函数
"""

import os
import time
import warnings
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 忽略 stable-baselines3 的冗余 UserWarning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="stable_baselines3.common.on_policy_algorithm",
)


def write_flag_file(flag_filename: str = "rl_visu_flag") -> bool:
    """创建标志文件，用于控制是否启用可视化
    
    Args:
        flag_filename: 标志文件名，默认为 "rl_visu_flag"
        
    Returns:
        如果成功创建返回 True，否则返回 False
    """
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w", encoding="utf-8") as f:
            f.write("This is a flag file")
        return True
    except Exception:
        return False


def check_flag_file(flag_filename: str = "rl_visu_flag") -> bool:
    """检查标志文件是否存在
    
    Args:
        flag_filename: 标志文件名，默认为 "rl_visu_flag"
        
    Returns:
        标志文件存在返回 True，否则返回 False
    """
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)


def delete_flag_file(flag_filename: str = "rl_visu_flag") -> bool:
    """删除标志文件
    
    Args:
        flag_filename: 标志文件名，默认为 "rl_visu_flag"
        
    Returns:
        删除成功或文件不存在返回 True，否则返回 False
    """
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception:
        return False

class PushMetricsCallback(BaseCallback):
    """自定义训练回调类，用于记录推方块任务的关键指标
    
    在每个训练步骤后从环境的 info 中提取自定义指标，并将其记录到 TensorBoard，
    便于监控训练过程中以下关键指标：
    - cube_distance_to_goal: 方块到目标点的距离
    - cube_speed: 方块运动速度
    - manipulability: 末端可操作性指标
    - singularity_penalty: 奇异位姿惩罚项
    - contact_quality: 推动姿态与接触质量
    - contact_flag: 推板与方块的真实接触标志
    - engagement_score: 连续阶段混合权重
    - plate_alignment: 推板朝向与推动方向的对齐度
    - cube_v_parallel: 方块沿目标方向的速度
    - cube_v_lateral: 方块横向速度
    """
    
    def _on_step(self) -> bool:
        """在每个训练步骤后被调用
        
        从环境返回的 info 中提取各项指标并记录平均值
        
        Returns:
            True 继续训练，False 停止训练
        """
        # 从环境通过 info 字典返回的信息中获取指标
        infos = self.locals.get("infos", [])
        for info in infos:
            # 记录方块到目标点的距离（平均值）
            if "cube_distance_to_goal" in info:
                self.logger.record_mean(
                    "custom/cube_distance_to_goal_mean",
                    float(info["cube_distance_to_goal"]),
                )
            # 记录方块运动速度（平均值）
            if "cube_speed" in info:
                self.logger.record_mean("custom/cube_speed_mean", float(info["cube_speed"]))

            if "contact_flag" in info:
                self.logger.record_mean(
                    "custom/contact_flag_rate",
                    float(info["contact_flag"]),
                )

        return True


class XarmEnv(gym.Env):
    """
    Rizon4 Push 环境。
    agent 输出前 6 个关节的角度增量，末端将 cube 推到地面目标点。
    """

    def __init__(
        self,
        visualize: bool = False,
        max_steps: int = 300,
        obs_noise_scale: float = 0.001,
        model_xml_path: str = "description/mjcf/scene.xml",
        frame_skip: int = 5,
        delta_scale: float = 0.05,
        home_joint_pos: Optional[np.ndarray] = None,
        fixed_cube_xy: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        self.model_xml_path = model_xml_path
        self.model = mujoco.MjModel.from_xml_path(self.model_xml_path)
        self.data = mujoco.MjData(self.model)

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 2.5
            self.handle.cam.azimuth = 135.0
            self.handle.cam.elevation = -25.0
            self.handle.cam.lookat = np.array([0.25, -0.20, 0.60], dtype=np.float64)

        # 获取 MuJoCo 模型的结构信息
        self.num_actuators = int(self.model.nu)  # 控制执行器数（joint7已通过equality锁定，不再有被动关节）
        
        # 获取各种 MuJoCo 元素的 ID（用于后续查询和操作）
        self.end_effector_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef")  # 末端位置
        self.push_plate_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "push_plate")  # 推板几何体
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")  # 方块几何体
        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")  # 方块自由关节
        
        # 获取方块在状态向量中的地址（用于快速读取和修改状态）
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])  # 位置向量地址
        self.cube_dof_adr = int(self.model.jnt_dofadr[self.cube_joint_id])  # 速度向量地址
        
        # 获取控制关节的 ID 和地址
        self.control_joint_ids = self.model.actuator_trnid[:self.num_actuators, 0].astype(np.int32)
        self.control_qpos_adr = self.model.jnt_qposadr[self.control_joint_ids].astype(np.int32)  # 位置地址数组
        self.control_dof_adr = self.model.jnt_dofadr[self.control_joint_ids].astype(np.int32)  # 速度地址数组
        
        # 获取控制范围并计算范围跨度（用于规范化）
        self.ctrl_ranges = self.model.actuator_ctrlrange[:self.num_actuators].copy().astype(np.float32)
        self.joint_range_low = self.ctrl_ranges[:, 0]  # 最小范围
        self.joint_range_span = self.ctrl_ranges[:, 1] - self.joint_range_low  # 范围宽度

        # home 位姿
        default_home = np.array(
            [0, 1.0, 0, 1.0, 0, 0],
            dtype=np.float32,
        )
        if home_joint_pos is None:
            home_joint_pos = default_home
        self.home_joint_pos = np.asarray(home_joint_pos, dtype=np.float32)
        if self.home_joint_pos.shape != (self.num_actuators,):
            raise ValueError(
                f"home_joint_pos shape mismatch: expected {(self.num_actuators,)}, "
                f"got {self.home_joint_pos.shape}"
            )

        # 模拟器参数
        self.frame_skip = frame_skip  # 每个上位action执行的模拟步数
        self.delta_scale = delta_scale  # 关节增量控制的最大幅度
        self.obs_noise_scale = obs_noise_scale  # 观测噪声标准差
        self.max_steps = max_steps  # 最大步数

        # 目标和采样相关的参数
        self.goal_size = 0.03
        self.sample_x_range = np.array([-0.30, 0.55], dtype=np.float32)
        self.sample_y_range = np.array([-0.35, 0.45], dtype=np.float32)
        self.base_center_xy = np.array([0.0, 0.0], dtype=np.float32)
        self.base_radius_range = (0.28, 0.65)
        self.cube_goal_distance_range = (0.12, 0.35)
        self.cube_height = 0.051
        self.push_offset = 0.08  # 推板中心到方块中心距离，需要略大于cube_half+plate_half
        self.goal_threshold = 0.05  # 稍微放宽成功判定
        self.goal_speed_threshold = 0.03  # 成功时要求方块保持较小速度

        default_fixed_cube_xy = np.array([0.38, 0.0], dtype=np.float32)
        if fixed_cube_xy is None:
            fixed_cube_xy = default_fixed_cube_xy
        self.fixed_cube_xy = np.asarray(fixed_cube_xy, dtype=np.float32)
        if self.fixed_cube_xy.shape != (2,):
            raise ValueError(
                f"fixed_cube_xy shape mismatch: expected {(2,)}, got {self.fixed_cube_xy.shape}"
            )
        if not self._is_xy_in_sampling_region(self.fixed_cube_xy):
            raise ValueError(f"fixed_cube_xy {self.fixed_cube_xy.tolist()} 不在有效采样区域内。")

        # 观测归一化参数
        self.position_obs_center = np.array(
            [
                0.5 * (self.sample_x_range[0] + self.sample_x_range[1]),
                0.5 * (self.sample_y_range[0] + self.sample_y_range[1]),
                0.40,
            ],
            dtype=np.float32,
        )
        self.position_obs_half_extent = np.array(
            [
                0.5 * (self.sample_x_range[1] - self.sample_x_range[0]),
                0.5 * (self.sample_y_range[1] - self.sample_y_range[0]),
                0.40,
            ],
            dtype=np.float32,
        )
        self.relative_obs_scale = np.array([0.50, 0.50, 0.40], dtype=np.float32)
        self.joint_velocity_obs_scale = 5.0
        self.linear_velocity_obs_scale = 1.0

        # 简化后的分段奖励参数
        self.reach_reward_weight = 10.0
        self.push_reward_weight = 15.0
        self.action_penalty_weight = 0.01
        self.push_reward_activation_dist = 0.12
        self.pre_push_offset = 0.03
        self.velocity_penalty_weight = 0.01
        self.velocity_penalty_decay = 5.0
        self.success_bonus_value = 100.0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actuators,),
            dtype=np.float32,
        )
        self.obs_size = 37
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32,
        )

        self.goal = np.zeros(3, dtype=np.float32)
        self.pre_push_direction = np.zeros(3, dtype=np.float32)
        self.target_qpos = np.zeros(self.num_actuators, dtype=np.float32)
        self.prev_ee_pos = np.zeros(3, dtype=np.float32)
        self.prev_cube_pos = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.np_random = np.random.default_rng(None)

        self.manipulability_threshold = 0.005
        self.singularity_penalty_weight = 2.0
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def _get_control_joint_pos(self) -> np.ndarray:
        """获取控制关节的当前位置
        
        Returns:
            当前关节位置数组，形状为 (num_actuators,)
        """
        return self.data.qpos[self.control_qpos_adr].copy().astype(np.float32)

    def _get_control_joint_vel(self) -> np.ndarray:
        """获取控制关节的当前速度
        
        Returns:
            当前关节速度数组，形状为 (num_actuators,)
        """
        return self.data.qvel[self.control_dof_adr].copy().astype(np.float32)

    def _get_ee_pos(self) -> np.ndarray:
        """获取末端执行器（eef）在世界坐标系中的位置
        
        Returns:
            末端位置 [x, y, z]，shape (3,)
        """
        return self.data.site(self.end_effector_site_id).xpos.copy().astype(np.float32)

    def _get_cube_pos(self) -> np.ndarray:
        """获取方块当前位置
        
        Returns:
            方块位置 [x, y, z]，shape (3,)
        """
        return self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3].copy().astype(np.float32)

    def _get_cube_quat(self) -> np.ndarray:
        """获取方块当前姿态四元数。"""
        return self.data.qpos[self.cube_qpos_adr + 3:self.cube_qpos_adr + 7].copy().astype(np.float32)

    def _get_cube_velocity(self, cube_pos: np.ndarray) -> np.ndarray:
        """计算方块的速度（根据前两步位置差计算）
        
        Args:
            cube_pos: 当前方块位置
            
        Returns:
            方块速度，shape (3,)
        """
        dt = max(self.model.opt.timestep * self.frame_skip, 1e-6)
        return ((cube_pos - self.prev_cube_pos) / dt).astype(np.float32)

    def _set_pre_push_direction(self, cube_pos: np.ndarray) -> None:
        """在每个 episode 开始时固定预推方向。"""
        direction_vector = self.goal - cube_pos
        direction_norm = float(np.linalg.norm(direction_vector))
        if direction_norm > 1e-4:
            self.pre_push_direction = (direction_vector / direction_norm).astype(np.float32)
        else:
            self.pre_push_direction = np.zeros(3, dtype=np.float32)

    def _get_pre_push_point(self, cube_pos: np.ndarray) -> np.ndarray:
        """使用固定的预推方向计算方块背后的预推点。"""
        return (cube_pos - self.pre_push_direction * self.pre_push_offset).astype(np.float32)

    def _get_ee_velocity(self, ee_pos: np.ndarray) -> np.ndarray:
        """计算末端执行器速度（根据前两步位置差计算）。"""
        dt = max(self.model.opt.timestep * self.frame_skip, 1e-6)
        return ((ee_pos - self.prev_ee_pos) / dt).astype(np.float32)

    def _set_cube_pose(self, cube_pos: np.ndarray) -> None:
        self.data.qpos[self.cube_qpos_adr:self.cube_qpos_adr + 3] = cube_pos
        self.data.qpos[self.cube_qpos_adr + 3:self.cube_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.data.qvel[self.cube_dof_adr:self.cube_dof_adr + 6] = 0.0

    def _get_joint_limit_normalized(self, qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """获取关节限位的归一化状态
        
        将关节位置映射到 [-1, 1] 范围，其中 -1 和 1 分别表示运动范围的两端。
        
        Args:
            qpos: 可选的关节位置数组；若为 None，则从当前仿真状态读取

        Returns:
            关节限位归一化值，shape (num_actuators,)
        """
        if qpos is None:
            qpos = self._get_control_joint_pos()
        normalized = 2.0 * (qpos - self.joint_range_low) / self.joint_range_span - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _normalize_joint_velocity(self, joint_vel: np.ndarray) -> np.ndarray:
        """将关节速度裁剪并归一化到 [-1, 1]。"""
        return np.clip(joint_vel / self.joint_velocity_obs_scale, -1.0, 1.0).astype(np.float32)

    def _normalize_linear_velocity(self, linear_vel: np.ndarray) -> np.ndarray:
        """将笛卡尔线速度裁剪并归一化到 [-1, 1]。"""
        return np.clip(linear_vel / self.linear_velocity_obs_scale, -1.0, 1.0).astype(np.float32)

    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """将世界坐标位置归一化到 [-1, 1]。"""
        normalized = (pos - self.position_obs_center) / self.position_obs_half_extent
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _normalize_relative_vector(self, vec: np.ndarray) -> np.ndarray:
        """将相对位移向量按统一尺度归一化到 [-1, 1]。"""
        normalized = vec / self.relative_obs_scale
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _get_manipulability(self) -> float:
        """计算末端可操作性指标（Manipulability）
        
        Manipulability = sqrt(det(J @ J.T))，其中 J 是雅可比矩阵。
        指标值越大表示末端执行器在当前配置下的操作灵活性越好，
        值越小表示越接近奇异点。用于惩罚低可操作性配置。
        
        Returns:
            可操作性标量值
        """
        self._jacp[:] = 0
        self._jacr[:] = 0
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.end_effector_site_id)
        J = self._jacp[:, self.control_dof_adr]
        JJT = J @ J.T
        det_val = np.linalg.det(JJT)
        return float(np.sqrt(max(det_val, 0.0)))

    def _get_singularity_penalty(self, manipulability: Optional[float] = None) -> tuple[float, float]:
        """计算奇异位姿惩罚

        当末端可操作性低于阈值时，使用归一化后的平方亏损作为惩罚。
        这样在接近奇异点时惩罚会平滑增大，同时保持有上界，避免奖励剧烈震荡。

        Args:
            manipulability: 可选的外部传入可操作性，避免重复计算

        Returns:
            (manipulability, singularity_penalty)
        """
        if manipulability is None:
            manipulability = self._get_manipulability()

        threshold = max(self.manipulability_threshold, 1e-6)
        deficit_ratio = max(threshold - manipulability, 0.0) / threshold
        singularity_penalty = -self.singularity_penalty_weight * (deficit_ratio ** 2)
        return float(manipulability), float(singularity_penalty)

    def _is_xy_in_sampling_region(self, point_xy: np.ndarray) -> bool:
        """检查平面点是否在有效的采样区域内
        
        采样区域由矩形范围和圆形范围同时约束，属于一个"环形弧形"区域。
        
        Args:
            point_xy: 二维平面坐标 [x, y]
            
        Returns:
            该点是否在采样区域内
        """
        x_ok = self.sample_x_range[0] <= point_xy[0] <= self.sample_x_range[1]
        y_ok = self.sample_y_range[0] <= point_xy[1] <= self.sample_y_range[1]
        if not (x_ok and y_ok):
            return False

        radius = float(np.linalg.norm(point_xy - self.base_center_xy))
        return self.base_radius_range[0] <= radius <= self.base_radius_range[1]

    def _sample_ground_xy(self) -> np.ndarray:
        """在有效的采样区域内随机采样一个点
        
        最多尝试2000次采样，直到找到有效点。
        
        Returns:
            采样到的二维点坐标
            
        Raises:
            RuntimeError: 如果无法在可达地面区域采样到有效点
        """
        for _ in range(2000):
            candidate = self.np_random.uniform(
                low=[self.sample_x_range[0], self.sample_y_range[0]],
                high=[self.sample_x_range[1], self.sample_y_range[1]],
            ).astype(np.float32)
            if self._is_xy_in_sampling_region(candidate):
                return candidate
        raise RuntimeError("无法在可达地面区域采样到有效点。")

    def _sample_cube_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """固定方块初始位置，并随机采样目标位置
        
        确保：
        1. 固定方块位置在采样区域内
        2. 方块到目标的距离在指定范围内
        3. 起始推点和目标推点都在有效采样区域内
        
        Returns:
            (cube_pos, goal_pos): 方块初始位置和目标位置
            
        Raises:
            RuntimeError: 如果无法采样到满足约束的有效配置
        """
        cube_xy = self.fixed_cube_xy.copy()
        for _ in range(5000):
            goal_xy = self._sample_ground_xy()
            cube_to_goal = goal_xy - cube_xy
            planar_distance = float(np.linalg.norm(cube_to_goal))
            if not (self.cube_goal_distance_range[0] <= planar_distance <= self.cube_goal_distance_range[1]):
                continue

            push_dir_xy = cube_to_goal / max(planar_distance, 1e-6)
            start_push_xy = cube_xy - push_dir_xy * self.push_offset
            goal_push_xy = goal_xy - push_dir_xy * self.push_offset
            if not self._is_xy_in_sampling_region(start_push_xy):
                continue
            if not self._is_xy_in_sampling_region(goal_push_xy):
                continue

            cube_pos = np.array([cube_xy[0], cube_xy[1], self.cube_height], dtype=np.float32)
            goal_pos = np.array([goal_xy[0], goal_xy[1], self.cube_height], dtype=np.float32)
            return cube_pos, goal_pos

        raise RuntimeError("无法采样到满足 push 约束的 cube 和目标点。")

    def _render_scene(self) -> None:
        """可视化场景中的目标点
        
        在 MuJoCo 可视化窗口中绘制一个蓝色球体表示目标位置。
        """
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 0
        self.handle.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32),
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """重置环境并返回初始观测
        
        固定方块初始位置，并随机采样满足约束的目标位置。
        初始化关节位置为家位置，物理数据和所有状态变量。
        
        Args:
            seed: 随机数生成器种子
            options: 其他选项（当前未使用）
            
        Returns:
            obs: 初始观测值
            info: 信息字典，包含初始距离、速度等信息
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.control_qpos_adr] = self.home_joint_pos
        self.data.qvel[self.control_dof_adr] = 0.0

        cube_pos, self.goal = self._sample_cube_and_goal()
        self._set_cube_pose(cube_pos)

        self.target_qpos = self.home_joint_pos.copy()
        self.data.ctrl[:self.num_actuators] = self.target_qpos
        mujoco.mj_forward(self.model, self.data)

        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        self._set_pre_push_direction(cube_pos)
        manipulability, singularity_penalty = self._get_singularity_penalty()
        initial_cube_goal_dist = float(np.linalg.norm(self.goal[:2] - cube_pos[:2]))

        self.prev_ee_pos = ee_pos.copy()
        self.prev_cube_pos = cube_pos.copy()
        self.step_count = 0

        if self.visualize:
            self._render_scene()

        obs = self._get_observation()
        info = {
            "is_success": False,
            "cube_distance_to_goal": initial_cube_goal_dist,
            "cube_speed": 0.0,
            "cube_fell": False,
            "contact_flag": False,
            "manipulability": float(manipulability),
            "singularity_penalty": float(singularity_penalty),
            "simulation_unstable": False,
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """构造环境观测状态
        
        观测包括 37 维归一化特征：
        - 关节位置 (6维)：按关节限位映射到 [-1, 1]
        - 关节速度 (6维)：按固定尺度裁剪到 [-1, 1]
        - 末端位置 (3维)：工作空间坐标归一化
        - 末端线速度 (3维)：笛卡尔速度归一化
        - 方块位置 (3维)：工作空间坐标归一化
        - 方块线速度 (3维)：笛卡尔速度归一化
        - 方块姿态 (4维)：四元数
        - 目标位置 (3维)：工作空间坐标归一化
        - 末端到方块的相对向量 (3维)：归一化
        - 方块到目标的相对向量 (3维)：归一化
        
        Returns:
            观测值数组，shape (37,)
        """
        joint_pos = self._get_control_joint_pos()
        joint_vel = self._get_control_joint_vel()

        if self.obs_noise_scale > 0.0:
            joint_pos += self.np_random.uniform(
                -self.obs_noise_scale,
                self.obs_noise_scale,
                size=joint_pos.shape,
            ).astype(np.float32)
            joint_vel += self.np_random.uniform(
                -self.obs_noise_scale,
                self.obs_noise_scale,
                size=joint_vel.shape,
            ).astype(np.float32)

        joint_pos_obs = self._get_joint_limit_normalized(joint_pos)
        joint_vel_obs = self._normalize_joint_velocity(joint_vel)

        ee_pos = self._get_ee_pos()
        ee_vel = self._get_ee_velocity(ee_pos)
        cube_pos = self._get_cube_pos()
        cube_vel = self._get_cube_velocity(cube_pos)
        cube_quat = self._get_cube_quat()
        goal_pos = self.goal.copy().astype(np.float32)
        ee_to_cube = (cube_pos - ee_pos).astype(np.float32)
        cube_to_goal = (self.goal - cube_pos).astype(np.float32)

        obs = np.concatenate(
            [
                joint_pos_obs,
                joint_vel_obs,
                self._normalize_position(ee_pos),
                self._normalize_linear_velocity(ee_vel),
                self._normalize_position(cube_pos),
                self._normalize_linear_velocity(cube_vel),
                cube_quat,
                self._normalize_position(goal_pos),
                self._normalize_relative_vector(ee_to_cube),
                self._normalize_relative_vector(cube_to_goal),
            ]
        ).astype(np.float32)
        return obs

    def _has_push_plate_contact(self) -> bool:
        """检查推板与方块是否发生真实接触。"""
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            is_push_plate_cube_contact = (
                (contact.geom1 == self.push_plate_geom_id and contact.geom2 == self.cube_geom_id)
                or (contact.geom2 == self.push_plate_geom_id and contact.geom1 == self.cube_geom_id)
            )
            if is_push_plate_cube_contact:
                return True
        return False
    def _count_non_cube_contacts(self) -> int:
        """统计不涉及方块的碰撞数量
        
        用于检测机械臂是否与环境中的其他物体（如地面）发生碰撞。
        返回值越大表示与方块无关的碰撞越多，会被惩罚化。
        
        Returns:
            不涉及方块的碰撞数量
        """
        count = 0
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            if contact.geom1 != self.cube_geom_id and contact.geom2 != self.cube_geom_id:
                count += 1
        return count
    
    def _calc_reward(
        self,
        ee_pos: np.ndarray,
        cube_pos: np.ndarray,
        cube_vel: np.ndarray,
        action: np.ndarray,
        step_count: int,
    ) -> tuple[np.float32, bool, float, float, bool, float, float]:
        """计算奖励函数
        
        采用简化后的组合奖励：
        Rt = Rreach + Rpush + Rsuccess - Raction - Rvel
        - Rreach: 鼓励末端持续朝方块背对目标的预推点靠近
        - Rpush: 当末端足够接近方块时，奖励方块持续朝目标移动
        - Rsuccess: 方块进入目标容差范围时的一次性大奖励
        - Raction: 惩罚过大的动作输出
        - Rvel: 仅在接近目标时逐步增强的速度惩罚，鼓励平稳停靠
        
        Args:
            ee_pos: 末端当前位置
            cube_pos: 方块当前位置
            cube_vel: 方块当前速度
            action: 当前action
            
        Returns:
            (reward, terminated, cube_goal_dist, cube_speed, truncated,
             manipulability, singularity_penalty)
        """
        ee_cube_dist = float(np.linalg.norm(cube_pos - ee_pos))
        prev_cube_goal_dist = float(np.linalg.norm(self.goal - self.prev_cube_pos))
        cube_goal_dist = float(np.linalg.norm(self.goal - cube_pos))
        cube_speed = float(np.linalg.norm(cube_vel))
        manipulability, singularity_penalty = self._get_singularity_penalty()

        prev_pre_push_point = self._get_pre_push_point(self.prev_cube_pos)
        prev_ee_prepush_dist = float(np.linalg.norm(prev_pre_push_point - self.prev_ee_pos))
        pre_push_point = self._get_pre_push_point(cube_pos)
        ee_prepush_dist = float(np.linalg.norm(pre_push_point - ee_pos))

        reach_reward = self.reach_reward_weight * (prev_ee_prepush_dist - ee_prepush_dist)
        push_reward = 0.0
        if ee_cube_dist <= self.push_reward_activation_dist:
            push_reward = self.push_reward_weight * (prev_cube_goal_dist - cube_goal_dist)

        action_penalty = -self.action_penalty_weight * float(np.sum(np.square(action)))

        velocity_penalty = -self.velocity_penalty_weight * cube_speed * float(
            np.exp(-self.velocity_penalty_decay * cube_goal_dist)
        )
        success_bonus = 0.0
        terminated = False

        if cube_goal_dist <= self.goal_threshold and cube_speed <= self.goal_speed_threshold:
            terminated = True
            success_bonus = self.success_bonus_value

        total_reward = (
            reach_reward
            + push_reward
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

    def _has_invalid_state(self) -> bool:
        """检查物理仿真是否出现数值不稳定
        
        检查位置、速度、加速度、控制信号是否包含 NaN 或 Inf。
        如果仿真变得不稳定，会立即停止该步并给予大惩罚。
        
        Returns:
            仿真状态是否无效（包含NaN或Inf）
        """
        return not (
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
            and np.all(np.isfinite(self.data.qacc))
            and np.all(np.isfinite(self.data.ctrl))
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        """执行一步环境交互
        
        Args:
            action: 智能体输出的action，范围 [-1, 1]，对应关节增量控制
            
        Returns:
            obs: 观测值，包括关节位置、速度、末端位置、方块位置等
            reward: 奖励值
            terminated: 任务是否完成（达成目标）
            truncated: 是否超过最大步数
            info: 包含诊断信息的字典
        """
        self.step_count += 1
        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        delta_qpos = clipped_action * self.delta_scale
        self.target_qpos = np.clip(
            self.target_qpos + delta_qpos,
            self.ctrl_ranges[:, 0],
            self.ctrl_ranges[:, 1],
        ).astype(np.float32)

        simulation_unstable = False
        push_contact_during_step = False
        for _ in range(self.frame_skip):
            self.data.ctrl[:self.num_actuators] = self.target_qpos
            mujoco.mj_step(self.model, self.data)
            push_contact_during_step = push_contact_during_step or self._has_push_plate_contact()
            if self._has_invalid_state():
                simulation_unstable = True
                break

        if simulation_unstable:
            obs = np.zeros(self.obs_size, dtype=np.float32)
            info = {
                "is_success": False,
                "cube_distance_to_goal": float("inf"),
                "cube_speed": float("inf"),
                "cube_fell": False,
                "contact_flag": False,
                "simulation_unstable": True,
            }
            return obs, np.float32(-50.0), False, True, info

        # Get current state information
        ee_pos = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        cube_vel = self._get_cube_velocity(cube_pos)
        push_contact = push_contact_during_step or self._has_push_plate_contact()

        (
            reward,
            terminated,
            cube_goal_dist,
            cube_speed,
            truncated,
            manipulability,
            singularity_penalty,
        ) = self._calc_reward(
            ee_pos,
            cube_pos,
            cube_vel,
            clipped_action,
            self.step_count,
        )

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01)

        obs = self._get_observation()
        info = {
            "is_success": bool(terminated),
            "cube_distance_to_goal": float(cube_goal_dist),
            "cube_speed": float(cube_speed),
            "cube_fell": False,
            "contact_flag": bool(push_contact),
            "manipulability": float(manipulability),
            "singularity_penalty": float(singularity_penalty),
            "simulation_unstable": False,
        }

        self.prev_ee_pos = ee_pos.copy()
        self.prev_cube_pos = cube_pos.copy()
        return obs, reward, terminated, truncated, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 400_000,
    model_save_path: str = "runs/rizon4_push/rizon4_pos_push",
    visualize: bool = False,
    resume_from: Optional[str] = None,
    model_xml_path: str = "description/mjcf/scene.xml",
    frame_skip: int = 5,
    delta_scale: float = 0.05,
    tensorboard_log: str = "./tensorboard/rizon4_pos_push_v2/",
    eval_freq: int = 65536,
    n_eval_episodes: int = 10,
    best_model_save_path: Optional[str] = None,
    eval_log_path: Optional[str] = None,
) -> None:
    """使用 PPO 算法进行多进程并行训练
    
    创建多个并行环境进行采样，周期性地在独立的评估环境上评估模型，
    并保存最佳模型检查点。支持从已有模型继续训练。
    
    Args:
        n_envs: 并行环境数量（多进程）
        total_timesteps: 本次训练累计执行的环境交互步数
        model_save_path: 最终模型保存路径
        visualize: 是否启用可视化（仅第一个环境）
        resume_from: 从某个已有模型继续训练，为 None 则从头开始
        model_xml_path: MuJoCo 场景文件路径
        frame_skip: 每个 action 执行的模拟步数
        delta_scale: 控制增量的最大幅度
        tensorboard_log: TensorBoard 日志目录
        eval_freq: 评估频率（每 N 个总步数进行一次评估）
        n_eval_episodes: 每次评估的轮数
        best_model_save_path: 最佳模型保存目录
        eval_log_path: 评估日志目录
    """
    env_kwargs = {
        "visualize": visualize,
        "obs_noise_scale": 0.001,
        "model_xml_path": model_xml_path,
        "frame_skip": frame_skip,
        "delta_scale": delta_scale,
        "max_steps": 200,  # 从 300 增加到 500，给推动更多时间
    }

    env = make_vec_env(
        env_id=lambda: XarmEnv(**env_kwargs),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )

    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)

    if best_model_save_path is None:
        best_model_save_path = os.path.join(model_dir or ".", "best_model")
    if eval_log_path is None:
        eval_log_path = os.path.join(model_dir or ".", "eval_logs")
    os.makedirs(best_model_save_path, exist_ok=True)
    os.makedirs(eval_log_path, exist_ok=True)

    eval_env = Monitor(
        XarmEnv(
            visualize=False,
            max_steps=300,
            obs_noise_scale=0.0,
            model_xml_path=model_xml_path,
            frame_skip=frame_skip,
            delta_scale=delta_scale,
        )
    )
    effective_eval_freq = max(eval_freq // max(n_envs, 1), 1)
    callback = CallbackList(
        [
            PushMetricsCallback(),
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=best_model_save_path,
                log_path=eval_log_path,
                eval_freq=effective_eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
                warn=False,
            ),
        ]
    )

    if resume_from is not None:
        print(f"从模型 {resume_from} 恢复训练")
        model = PPO.load(resume_from, env=env)
    else:
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
            batch_size=512,     # 从 256 增加到 512
            n_epochs=10,
            gamma=0.995,        # 从 0.99 提高到 0.995，更重视远期回报
            gae_lambda=0.98,    # 从默认 0.95 提高到 0.98
            learning_rate=3e-4, 
            clip_range=0.2,
            ent_coef=0.005,     # 从 0.01 降低到 0.005，减少随机性
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=tensorboard_log,
        )

    print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
    print(f"评估频率: 每累计约 {effective_eval_freq * max(n_envs, 1)} 环境步评估一次")
    print(f"best_model 自动保存目录: {best_model_save_path}")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
        model.save(model_save_path)
        print(f"模型已保存至: {model_save_path}")
    finally:
        env.close()
        eval_env.close()


def test_ppo(
    model_path: str = "assets/model/rl_push_checkpoint/rizon4_pos_push_v1",
    total_episodes: int = 5,
    model_xml_path: str = "description/mjcf/scene.xml",
) -> None:
    """加载已训练的 PPO 模型进行推理和可视化测试
    
    在启用可视化的环境中运行多个完整的推任务轮次，
    统计成功率和性能指标。
    
    Args:
        model_path: 已训练模型文件路径（不需要 .zip 扩展名）
        total_episodes: 测试总轮数
        model_xml_path: MuJoCo 场景文件路径
    """
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    env = None
    try:
        env = XarmEnv(
            visualize=True,
            max_steps=200,
            obs_noise_scale=0.0,
            model_xml_path=model_xml_path,
            frame_skip=5,
            delta_scale=0.05,
        )

        success_count = 0
        final_distances = []
        final_speeds = []
        episode_lengths = []
        print(f"测试轮数: {total_episodes}")

        for ep in range(total_episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            episode_reward = 0.0
            step_count = 0
            info = {}

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                done = terminated or truncated
                step_count += 1
                time.sleep(0.02)

            final_distance = float(info["cube_distance_to_goal"])
            final_speed = float(info["cube_speed"])
            final_distances.append(final_distance)
            final_speeds.append(final_speed)
            episode_lengths.append(step_count)
            if info["is_success"]:
                success_count += 1
            print(
                f"轮次 {ep + 1:2d} | 总奖励: {episode_reward:7.2f} | "
                f"最终距离: {final_distance:.4f} | 最终速度: {final_speed:.4f} | "
                f"步数: {step_count:4d} | "
                f"结果: {'成功' if info['is_success'] else '失败'}"
            )
            time.sleep(0.25)

        success_rate = (success_count / total_episodes) * 100
        print(f"总成功率: {success_rate:.1f}%")
        print(f"平均最终距离: {float(np.mean(final_distances)):.4f}")
        print(f"平均最终速度: {float(np.mean(final_speeds)):.4f}")
        print(f"平均步数: {float(np.mean(episode_lengths)):.1f}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    delete_flag_file()
    TRAIN_MODE = True  # 设置为 True 以启用训练模式，False 以启用测试模式

    # ---- 训练配置 ----
    # 保存路径：用 v3 新目录，不覆盖原来的 v2 结果
    MODEL_PATH = "assets/model/rl_push_checkpoint_v3/rizon4_pos_push_v3"
    TENSORBOARD_DIR = "./tensorboard/rizon4_pos_push_v3/"
    BEST_MODEL_DIR = "assets/model/rl_push_checkpoint_v3/best_model"
    EVAL_LOG_DIR = "assets/model/rl_push_checkpoint_v3/eval_logs"
    RESUME_MODEL_PATH = None  # 设为已有 .zip 路径可从断点继续训练

    if TRAIN_MODE:
        train_ppo(
            n_envs=32,
            total_timesteps=5_000_000,
            model_save_path=MODEL_PATH,
            visualize=False,
            resume_from=RESUME_MODEL_PATH,
            model_xml_path="description/mjcf/scene.xml",
            frame_skip=5,
            delta_scale=0.05,
            tensorboard_log=TENSORBOARD_DIR,
            best_model_save_path=BEST_MODEL_DIR,
            eval_log_path=EVAL_LOG_DIR,
        )
    else:
        try:
            test_ppo(
                model_path=MODEL_PATH,
                total_episodes=20,
                model_xml_path="description/mjcf/scene.xml",
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"{exc}，请先训练或修改 MODEL_PATH。")
