import torch
import torch.nn.functional as F
from torch.distributions import Normal

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


class TD3Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, num_layers=3, hidden_dim=512):
        super().__init__()
        layers = [
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ELU(),
            *[
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ELU(),
            ] * (num_layers - 1),
        ]
        output_layer = torch.nn.Linear(hidden_dim, 1)
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
        self.layers = torch.nn.Sequential(*layers, output_layer)

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=-1)
        return self.layers(x)