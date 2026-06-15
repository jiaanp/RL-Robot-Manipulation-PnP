# Franka RL — TQC+HER for Panda Manipulation

MuJoCo 仿真环境中使用 TQC+HER 训练 Franka Panda 机械臂完成 Push / Slide / PickAndPlace 任务。

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
├── panda_mujoco_gym/          # MuJoCo 环境 (FrankaPush/Slide/PickAndPlace)
│   ├── envs/
│   │   ├── panda_env.py       # 基类 (位置增量控制)
│   │   ├── pick_and_place.py
│   │   ├── push.py
│   │   └── slide.py
│   └── assets/                # XML 模型 + mesh 文件
├── panda_rl/
│   ├── PickAndPlaceSparse/
│   │   └── TQC/
│   │       ├── train.py       # 训练脚本 (课程学习)
│   │       ├── watch.py       # 可视化脚本
│   │       └── runs/
│   │           └── tqc_pnp_sparse_curriculum_v2/
│   │               ├── final_model.zip        # 最终模型 (100% 成功率)
│   │               ├── best_model/            # 最佳评估模型
│   │               └── eval_logs/             # 评估日志
│   ├── PushSparse/
│   └── SlideSparse/
└── docs/                      # 演示 GIF
```

## 训练

```bash
cd panda_rl/PickAndPlaceSparse/TQC

# 基础训练
python train.py --timesteps 1_000_000

# 课程学习 (推荐)
python train.py --timesteps 1_000_000 --curriculum

# 查看效果
python watch.py --model runs/tqc_pnp_sparse_curriculum_v2/final_model.zip
```

## 超参数

| Parameter | Value |
| --- | --- |
| Algorithm | TQC + HER |
| Policy | MultiInputPolicy |
| Network size | [256, 256, 256] |
| Batch size | 512 |
| Buffer size | 1,000,000 |
| Learning rate | 7e-4 |
| Polyak update (τ) | 0.05 |
| Discount factor (γ) | 0.95 |
| HER strategy | Future |
| n_sampled_goal | 4 |
| n_critics | 2 |
| Top quantiles to drop | 2 |
| Max episode steps | 100 |
| n_substeps (train) | 15 |
| n_substeps (eval) | 25 |

## 课程学习 (Curriculum)

```python
PHASES = [
    (0,       {"obj_xy_range": 0.08, "goal_xy_range": 0.12, ...}),  # 简单
    (80_000,  {"obj_xy_range": 0.12, "goal_xy_range": 0.16, ...}),  # 渐进
    (180_000, {"obj_xy_range": 0.18, "goal_xy_range": 0.20, ...}),
    (350_000, {"obj_xy_range": 0.22, "goal_xy_range": 0.22, ...}),
    (600_000, {"obj_xy_range": 0.30, "goal_xy_range": 0.30, ...}),  # 完整
]
```

## 训练结果 (PickAndPlaceSparse)

| 版本 | 训练步数 | 成功率 | 备注 |
| --- | --- | --- | --- |
| curriculum_v2 | 1,000,000 | **100%** | 课程学习, 最终评估 20/20 |
| curriculum | 500,000 | ~50% | 早期版本 |
| fast | 未完成 | — | — |

## Watch 参数

```bash
python watch.py --help

# 常用:
--episodes 5           # 评估次数
--sleep 0.02           # 步间延迟 (越小越快)
--model <path>         # 模型路径
```

**提示**: MuJoCo 窗口首次打开可能较小，手动拖大后会自动记住大小。
