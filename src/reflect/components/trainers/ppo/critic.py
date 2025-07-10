import numpy as np
import torch.nn as nn
from reflect.components.trainers.ppo.utils import layer_init


class Critic(nn.Module):
    def __init__(
            self,
            input_dim
        ):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(input_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        return self.critic(x)