import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    Simple MLP policy network
    """
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)