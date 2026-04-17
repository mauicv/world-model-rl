import torch
import torch.nn as nn

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

class MLPActor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, num_layers: int = 2, hidden_dim: int = 512):
        super().__init__()
        layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.layers = nn.Sequential(*layers)
        torch.nn.init.uniform_(
            self.layers[-1].weight,
            -WEIGHTS_FINAL_INIT,
            WEIGHTS_FINAL_INIT
        )
        torch.nn.init.uniform_(
            self.layers[-1].bias,
            -BIAS_FINAL_INIT,
            BIAS_FINAL_INIT
        )

    def forward(self, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return torch.tanh(self.layers(z))
