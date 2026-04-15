import torch
import torch.nn as nn


class MLPActor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, num_layers: int = 2, hidden_dim: int = 512):
        super().__init__()
        layers = [
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        ]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
            ])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return torch.tanh(self.layers(z))
