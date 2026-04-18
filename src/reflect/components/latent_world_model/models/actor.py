import torch
import torch.nn as nn

from reflect.components.latent_world_model.models.mlp import MLP, orthogonal_init_fn
BIAS_FINAL_INIT = 3e-4

class MLPActor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, num_layers: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.mlp = MLP(
            input_dim=latent_dim,
            output_dim=hidden_dim,
            num_layers=2,
            hidden_dim=hidden_dim,
            output_activation=nn.Tanh,
        )
        self.output_layer = torch.nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh(),
        )
        self.apply(orthogonal_init_fn)

    def forward(self, z: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.output_layer(self.mlp(z))
