import torch
from torch.distributions import Normal
import numpy as np
import torch.nn as nn
from reflect.components.trainers.ppo.utils import layer_init


class PPOActor(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
        ):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(input_dim).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(output_dim)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(output_dim)))

    def forward(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs