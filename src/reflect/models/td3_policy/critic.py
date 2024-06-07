import torch
import torch.nn.functional as F
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_space, num_layers=4, hidden_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]
        self.num_layers=num_layers
        self.hidden_dim=hidden_dim

        layers = []
        layers.append(torch.nn.Linear(
            self.state_dim + self.action_dim, hidden_dim
        ))
        for _ in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU()
            ])

        final_layer = torch.nn.Linear(hidden_dim, 1)
        layers.append(final_layer)
        self.layers = torch.nn.Sequential(*layers)

        torch.nn.init.uniform_(
            final_layer.weight,
            -WEIGHTS_FINAL_INIT,
            WEIGHTS_FINAL_INIT
        )
        torch.nn.init.uniform_(
            final_layer.bias,
            -BIAS_FINAL_INIT,
            BIAS_FINAL_INIT
        )

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=-1)
        return self.layers(x)
