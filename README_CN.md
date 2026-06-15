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

## 成功率对比 (TQC+HER vs SAC+HER)

| 环境 | TQC+HER | SAC+HER |
| --- | --- | --- |
| FrankaPushSparse-v0 | 100.00% | 73.33% |
| FrankaPushDense-v0 | 100.00% | 0.00% |
| FrankaSlideSparse-v0 | 0.00% | 0.00% |
| FrankaSlideDense-v0 | 20.00% | 0.00% |
| FrankaPickAndPlaceSparse-v0 | 100.00% | 0.00% |
| FrankaPickAndPlaceDense-v0 | 33.33% | 46.67% |
