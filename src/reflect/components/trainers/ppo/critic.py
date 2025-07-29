import numpy as np
import torch.nn as nn
from reflect.components.trainers.ppo.utils import layer_init


class PPOCritic(nn.Module):
    def __init__(
            self,
            input_dim,
            num_layers=3,
            hidden_dim=512,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        layers = [
            layer_init(nn.Linear(np.array(input_dim).prod(), hidden_dim)),
            nn.Tanh(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
            ])
        layers.extend([
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        ])
        self.critic = nn.Sequential(*layers)

    def forward(self, x):
        return self.critic(x)