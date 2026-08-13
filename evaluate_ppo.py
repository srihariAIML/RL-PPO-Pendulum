import argparse
import numpy as np
import gymnasium as gym

from ppo_agent import PPOAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = gym.make(
        "Pendulum-v1",
        render_mode="human" if args.render else None,
    )

    agent = PPOAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        action_dim=int(np.prod(env.action_space.shape)),
        action_low=float(env.action_space.low[0]),
        action_high=float(env.action_space.high[0]),
        hidden_size=64,
    )
    agent.load("models/ppo_pendulum.pt")

    rewards = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=1000 + episode)
        done = False
        total_reward = 0.0

        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"Episode {episode + 1:02d}: return = {total_reward:.2f}")

    print("\nEvaluation summary")
    print(f"Mean return: {np.mean(rewards):.2f}")
    print(f"Std. deviation: {np.std(rewards):.2f}")
    print(f"Best return: {np.max(rewards):.2f}")
    print(f"Worst return: {np.min(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
