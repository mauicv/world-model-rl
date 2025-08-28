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
            num_layers=3,
            hidden_dim=512,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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
            layer_init(nn.Linear(hidden_dim, np.prod(output_dim)), std=0.01),
        ])
        self.actor_mean = nn.Sequential(*layers)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(output_dim)))

    def forward(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        action_mean = torch.tanh(action_mean)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs