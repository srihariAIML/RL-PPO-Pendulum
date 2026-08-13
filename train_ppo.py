import argparse
import csv
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppo_agent import PPOAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    args = parser.parse_args()

    set_seed(args.seed)

    env = gym.make("Pendulum-v1")
    env.action_space.seed(args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=float(env.action_space.low[0]),
        action_high=float(env.action_space.high[0]),
        hidden_size=64,
        learning_rate=3e-4,
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        update_epochs=10,
        minibatch_size=64,
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    obs, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_rewards = []
    global_steps = 0

    while global_steps < args.total_steps:
        rollout = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "log_probs": [],
            "values": [],
        }

        for _ in range(args.rollout_steps):
            action, log_prob, value = agent.get_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rollout["obs"].append(obs.copy())
            rollout["actions"].append(action.copy())
            rollout["rewards"].append(float(reward))
            rollout["dones"].append(float(done))
            rollout["log_probs"].append(float(log_prob))
            rollout["values"].append(float(value))

            episode_reward += reward
            global_steps += 1
            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()

            if global_steps >= args.total_steps:
                break

        _, _, last_value = agent.get_action(obs, deterministic=True)
        stats = agent.update(rollout, last_value)

        if len(episode_rewards) and global_steps % args.rollout_steps < 10:
            recent = episode_rewards[-10:]
            print(
                f"Steps {global_steps:7d} | "
                f"Episodes {len(episode_rewards):4d} | "
                f"AvgReward(10) {np.mean(recent):8.2f} | "
                f"PolicyLoss {stats['policy_loss']:8.4f} | "
                f"ValueLoss {stats['value_loss']:8.4f} | "
                f"Entropy {stats['entropy']:7.3f}"
            )

    model_path = "models/ppo_pendulum.pt"
    agent.save(model_path)

    with open("results/training_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, reward in enumerate(episode_rewards, 1):
            writer.writerow([i, reward])

    if episode_rewards:
        plt.figure(figsize=(9, 5))
        plt.plot(episode_rewards, alpha=0.35, label="Episode reward")
        if len(episode_rewards) >= 10:
            moving = np.convolve(
                episode_rewards, np.ones(10) / 10, mode="valid"
            )
            plt.plot(
                range(10, len(episode_rewards) + 1),
                moving,
                label="10-episode average",
            )
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("PPO Training Reward — Pendulum-v1")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/reward_curve.png", dpi=200)
        plt.close()

    env.close()
    print(f"\nSaved model: {model_path}")
    print("Saved results: results/training_results.csv")
    print("Saved graph: results/reward_curve.png")


if __name__ == "__main__":
    main()
