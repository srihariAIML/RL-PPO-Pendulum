# PPO Pendulum-v1 Reinforcement Learning Project

## Project Overview

This project implements Proximal Policy Optimization (PPO) for continuous control in Gymnasium's `Pendulum-v1` environment.

The goal is to learn a policy that applies continuous torque to control the pendulum toward the upright position while considering movement and control effort.

## Project Structure

```text
RL-PPO-Pendulum/
├── README.md
├── requirements.txt
├── ppo_agent.py
├── train_ppo.py
├── evaluate_ppo.py
├── results/
│   ├── reward_curve.png
│   ├── evaluation_results.txt
│   └── README.txt
└── models/
    └── ppo_pendulum.pt
```

## Environment

`Pendulum-v1` provides three observation values: cosine of the angle, sine of the angle, and angular velocity. The action is a continuous torque value from -2 to 2.

## Installation

Use Python 3.10 or newer if possible.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Agent

```bash
python train_ppo.py
```

The training script saves the trained model and episode-reward data under `models/` and `results/`.

## Evaluate the Agent

```bash
python evaluate_ppo.py
```

To attempt visual rendering on a local computer:

```bash
python evaluate_ppo.py --render
```

## PPO Components

The implementation includes an actor network, critic network, Gaussian continuous-action policy, PPO clipping, Generalized Advantage Estimation (GAE), entropy regularization, value loss, and mini-batch updates.

## Actual Evaluation Results

The completed 10-episode evaluation produced:

- Mean return: **-1194.15**
- Standard deviation: **47.13**
- Best return: **-1141.12**
- Worst return: **-1279.83**

The results indicate moderate learning improvement with noticeable variability; the policy did not completely converge.

The actual evaluation episodes are recorded in `results/evaluation_results.txt`, and the training reward curve is provided in `results/reward_curve.png`.

## Reproducibility

The training script accepts a random seed, for example:

```bash
python train_ppo.py --seed 42
```

For stronger experimental validation, additional random seeds and longer training runs can be used in future work.

## Academic Note

This repository contains the student implementation used for the reinforcement learning project. The PPO implementation follows standard PPO concepts from the research literature and is organized specifically for this project.
