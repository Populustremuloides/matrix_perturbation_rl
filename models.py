# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)  # Output is B_{i,i}

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # We can use tanh to bound the action between -1 and 1, or scale as needed
        return self.out(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Input is state and action concatenated
        self.fc1 = nn.Linear(1 + 1, 64)  # State and action
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)  # Q-value output

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

