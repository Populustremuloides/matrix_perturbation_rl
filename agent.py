# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import Actor, Critic
from replay_buffer.py import ReplayBuffer

class DDPGAgent:
    def __init__(self, action_space, state_space, max_action, min_action, lr_actor=1e-3, lr_critic=1e-3,
                 gamma=0.99, tau=0.005, buffer_size=100000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.max_action = max_action
        self.min_action = min_action
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy()

        # Add exploration noise
        action += noise_scale * np.random.randn(1)
        action = action.clip(self.min_action, self.max_action)
        return action

    def train(self, batch_size=64):
        if self.replay_buffer.size < batch_size:
            return

        # Sample a batch of experiences
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Compute target actions and Q-values
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            next_action = next_action.clamp(self.min_action, self.max_action)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Critic loss
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

