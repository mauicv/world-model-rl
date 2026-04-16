import torch
import torch.nn as nn


class MLPDynamicModel(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            num_layers: int = 3,
            hidden_dim: int = 512,
        ):
        super().__init__()
        trunk_layers = [
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
        ]
        for _ in range(num_layers - 1):
            trunk_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
        self.trunk = nn.Sequential(*trunk_layers)
        self.z_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, a: torch.Tensor):
        x = torch.cat([z, a], dim=-1)
        x = self.trunk(x)
        return self.z_head(x), self.reward_head(x), torch.sigmoid(self.done_head(x))
