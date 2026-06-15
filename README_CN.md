# Franka RL — TQC+HER 机械臂操作

在 MuJoCo 仿真环境中，使用 TQC+HER 训练 Franka Panda 机械臂完成抓取 / 推 / 滑动物体的任务。

[English](README.md)

## 环境要求

```
mujoco>=3.7.0
gymnasium>=1.3.0
gymnasium-robotics>=1.4.2
stable-baselines3>=2.8.0
sb3-contrib>=2.8.0
glfw>=2.7.0
numpy
```

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── panda_mujoco_gym/          # MuJoCo 环境
│   ├── envs/
│   │   ├── panda_env.py       # 基类（位置增量控制）
│   │   ├── pick_and_place.py
│   │   ├── push.py
│   │   └── slide.py
│   └── assets/                # XML 模型 + mesh 文件
├── panda_rl/
│   ├── PickAndPlaceSparse/
│   │   └── TQC/
│   │       ├── train.py       # 训练脚本（支持课程学习）
│   │       ├── watch.py       # 可视化脚本
│   │       └── runs/
│   │           └── tqc_pnp_sparse_curriculum_v2/
│   │               ├── final_model.zip        # 训练好的模型（100% 成功率）
│   │               ├── best_model/            # 最佳评估模型
│   │               └── eval_logs/             # 评估日志
│   ├── PushSparse/
│   └── SlideSparse/
└── docs/                      # 演示 GIF
```

## 快速开始

### 训练

```bash
cd panda_rl/PickAndPlaceSparse/TQC

# 基础训练
python train.py --timesteps 1_000_000

# 课程学习（推荐）
python train.py --timesteps 1_000_000 --curriculum

# 从检查点恢复
python train.py --load-model runs/xxx/best_model/best_model.zip
```

### 评估

```bash
python watch.py --model runs/tqc_pnp_sparse_curriculum_v2/final_model.zip
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 10 | 评估 episode 数 |
| `--sleep` | 0.02 | 每步延迟秒数（越小越快） |
| `--model` | `runs/.../final_model.zip` | 模型路径 |

> **提示**：MuJoCo 窗口首次打开可能偏小，手动拖大一次后会自动记住大小。

## 超参数

| Parameter | Value |
| --- | --- |
| 算法 | TQC + HER |
| 策略 | MultiInputPolicy |
| 网络结构 | [256, 256, 256] |
| 批次大小 | 512 |
| 缓冲大小 | 1,000,000 |
| 学习率 | 7e-4 |
| Polyak 更新 (τ) | 0.05 |
| 折扣因子 (γ) | 0.95 |
| HER 策略 | Future (n_sampled_goal=4) |
| 评论家数量 | 2（TQC 特有） |
| 每评论家丢弃分位数 | 2（TQC 特有） |
| 最大 episode 步数 | 100 |
| n_substeps（训练） | 15 |
| n_substeps（评估） | 25 |

## 课程学习

```python
PHASES = [
    (0,       {"obj_xy_range": 0.08, "goal_xy_range": 0.12, ...}),   # 简单
    (80_000,  {"obj_xy_range": 0.12, "goal_xy_range": 0.16, ...}),   # 中等
    (180_000, {"obj_xy_range": 0.18, "goal_xy_range": 0.20, ...}),
    (350_000, {"obj_xy_range": 0.22, "goal_xy_range": 0.22, ...}),
    (600_000, {"obj_xy_range": 0.30, "goal_xy_range": 0.30, ...}),   # 完整任务
]
```

## 训练结果

| 版本 | 训练步数 | 成功率 | 备注 |
| --- | --- | --- | --- |
| curriculum_v2 | 1,000,000 | **100%** | 课程学习，最终评估 20/20 |
| curriculum | 500,000 | ~50% | 早期版本 |
| fast | 未完成 | — | 训练中断 |

---

## Rizon4 推方块 — PPO

使用 PPO 算法在 Rizon4 机械臂上实现推方块任务，v6 版本优化了奖励函数。

[English](README.md#rizon4-push--ppo)

### 结构

```
└── rizon4_push/
    ├── rizon4_push.py            # 环境代码 (XarmEnv + train_ppo)
    ├── train_v6_reward.py        # 优化奖励函数的训练脚本
    └── description/              # URDF/MJCF + mesh 文件 (Rizon4 模型)
```

### 训练

```bash
cd rizon4_push
python train_v6_reward.py
```

配置：PPO，32 并行环境，600 万步，针对推方块任务的奖励优化。

### v6 奖励设计要点

| 改动 | 说明 |
| --- | --- |
| Push 激活 | 硬门槛 12cm → 指数平滑，不再卡激活条件 |
| Proximity 奖励 | 方块靠近目标即给分，填补推动→成功之间的空白 |
| Push 权重 | 15→25，动作惩罚减半，鼓励积极推动 |

### 结果

| 版本 | 成功率 |
| --- | --- |
| v6_reward (初始) | ~70% |
| v6_reward (续训 80 万步) | ~90%+ |


## 成功率对比 (TQC+HER vs SAC+HER)

| 环境 | TQC+HER | SAC+HER |
| --- | --- | --- |
| FrankaPushSparse-v0 | 100.00% | 73.33% |
| FrankaPushDense-v0 | 100.00% | 0.00% |
| FrankaSlideSparse-v0 | 0.00% | 0.00% |
| FrankaSlideDense-v0 | 20.00% | 0.00% |
| FrankaPickAndPlaceSparse-v0 | 100.00% | 0.00% |
| FrankaPickAndPlaceDense-v0 | 33.33% | 46.67% |
