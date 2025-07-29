import torch
import torch.nn.functional as F
from torch.distributions import Normal

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


class TD3Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, noise_std=0.0, num_layers=3, hidden_dim=512):
        super().__init__()
        layers = [
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ELU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ELU(),
            ])
        output_layer = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.uniform_(
            output_layer.weight,
            -WEIGHTS_FINAL_INIT,
            WEIGHTS_FINAL_INIT
        )
        torch.nn.init.uniform_(
            output_layer.bias,
            -BIAS_FINAL_INIT,
            BIAS_FINAL_INIT
        )
        self.noise_std = noise_std
        self.layers = torch.nn.Sequential(*layers, output_layer)


    def forward(self, x, deterministic=True):
        x = self.layers(x)
        mean = torch.tanh(x)
        if deterministic:
            return mean
        else:
            return Normal(mean, self.noise_std)
