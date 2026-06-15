# Franka RL — TQC+HER for Panda Manipulation

Train Franka Panda robot with TQC+HER in MuJoCo for Push / Slide / PickAndPlace tasks.

[中文版](README_CN.md)

## Requirements

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

## Project Structure

```
├── panda_mujoco_gym/          # MuJoCo environments
│   ├── envs/
│   │   ├── panda_env.py       # Base class (delta position control)
│   │   ├── pick_and_place.py
│   │   ├── push.py
│   │   └── slide.py
│   └── assets/                # XML models + meshes
├── panda_rl/
│   ├── PickAndPlaceSparse/
│   │   └── TQC/
│   │       ├── train.py       # Training script (w/ curriculum)
│   │       ├── watch.py       # Visualization script
│   │       └── runs/
│   │           └── tqc_pnp_sparse_curriculum_v2/
│   │               ├── final_model.zip        # Trained model (100% SR)
│   │               ├── best_model/
│   │               └── eval_logs/
│   ├── PushSparse/
│   └── SlideSparse/
└── docs/                      # Demo GIFs
```

## Quick Start

### Training

```bash
cd panda_rl/PickAndPlaceSparse/TQC

# Basic training
python train.py --timesteps 1_000_000

# With curriculum learning (recommended)
python train.py --timesteps 1_000_000 --curriculum

# Resume from checkpoint
python train.py --load-model runs/xxx/best_model/best_model.zip
```

### Evaluation

```bash
python watch.py --model runs/tqc_pnp_sparse_curriculum_v2/final_model.zip
```

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 10 | Number of evaluation episodes |
| `--sleep` | 0.02 | Delay between steps (smaller = faster) |
| `--model` | `runs/.../final_model.zip` | Path to model |

> **Tip**: The MuJoCo viewer window may open small initially. Drag the corner to resize — MuJoCo remembers it.

## Hyperparameters

| Parameter | Value |
| --- | --- |
| Algorithm | TQC + HER |
| Policy | MultiInputPolicy |
| Network | [256, 256, 256] |
| Batch size | 512 |
| Buffer size | 1,000,000 |
| Learning rate | 7e-4 |
| Polyak update (τ) | 0.05 |
| Discount factor (γ) | 0.95 |
| HER strategy | Future (n_sampled_goal=4) |
| n_critics | 2 (TQC) |
| Top quantiles to drop | 2 (TQC) |
| Max episode steps | 100 |
| n_substeps (train) | 15 |
| n_substeps (eval) | 25 |

## Curriculum Learning

```python
PHASES = [
    (0,       {"obj_xy_range": 0.08, "goal_xy_range": 0.12, ...}),   # Easy
    (80_000,  {"obj_xy_range": 0.12, "goal_xy_range": 0.16, ...}),   # Medium
    (180_000, {"obj_xy_range": 0.18, "goal_xy_range": 0.20, ...}),
    (350_000, {"obj_xy_range": 0.22, "goal_xy_range": 0.22, ...}),
    (600_000, {"obj_xy_range": 0.30, "goal_xy_range": 0.30, ...}),   # Full task
]
```

## Results (PickAndPlaceSparse)

| Version | Steps | Success Rate | Notes |
| --- | --- | --- | --- |
| curriculum_v2 | 1,000,000 | **100%** | Curriculum, final eval 20/20 |
| curriculum | 500,000 | ~50% | Early version |
| fast | incomplete | — | — |

## Success Rate Benchmark (TQC+HER vs SAC+HER)

| Environment | TQC+HER | SAC+HER |
| --- | --- | --- |
| FrankaPushSparse-v0 | 100.00% | 73.33% |
| FrankaPushDense-v0 | 100.00% | 0.00% |
| FrankaSlideSparse-v0 | 0.00% | 0.00% |
| FrankaSlideDense-v0 | 20.00% | 0.00% |
| FrankaPickAndPlaceSparse-v0 | 100.00% | 0.00% |
| FrankaPickAndPlaceDense-v0 | 33.33% | 46.67% |
