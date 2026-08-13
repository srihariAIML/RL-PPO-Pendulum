import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Gaussian policy for the one-dimensional continuous Pendulum action."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean(h)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std


class Critic(nn.Module):
    """State-value function."""

    def __init__(self, obs_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: float = -2.0,
        action_high: float = 2.0,
        hidden_size: int = 64,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.action_low = action_low
        self.action_high = action_high
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.actor = Actor(obs_dim, action_dim, hidden_size).to(self.device)
        self.critic = Critic(obs_dim, hidden_size).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=learning_rate
        )

    def get_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(obs_t)
            dist = Normal(mean, std)
            raw_action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(raw_action).sum(-1)
            value = self.critic(obs_t)

        action = torch.clamp(
            raw_action, self.action_low, self.action_high
        )
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def _distribution(self, obs):
        mean, std = self.actor(obs)
        return Normal(mean, std)

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = (
                rewards[t]
                + self.gamma * next_value * next_non_terminal
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout, last_value):
        obs = np.asarray(rollout["obs"], dtype=np.float32)
        actions = np.asarray(rollout["actions"], dtype=np.float32)
        rewards = np.asarray(rollout["rewards"], dtype=np.float32)
        dones = np.asarray(rollout["dones"], dtype=np.float32)
        old_log_probs = np.asarray(rollout["log_probs"], dtype=np.float32)
        values = np.asarray(rollout["values"], dtype=np.float32)

        advantages, returns = self.compute_gae(
            rewards, values, dones, last_value
        )

        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_t = torch.as_tensor(
            old_log_probs, dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = len(obs)
        indices = np.arange(n)

        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]

                dist = self._distribution(obs_t[mb])
                new_log_probs = dist.log_prob(actions_t[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_log_probs - old_log_t[mb])
                unclipped = ratio * adv_t[mb]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * adv_t[mb]

                actor_loss = -torch.min(unclipped, clipped).mean()

                values_pred = self.critic(obs_t[mb])
                critic_loss = 0.5 * (returns_t[mb] - values_pred).pow(2).mean()

                total_actor_loss = (
                    actor_loss - self.entropy_coef * entropy
                )

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())

        return {
            "policy_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
        }

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
