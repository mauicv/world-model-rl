import torch
import torch.nn as nn

from reflect.components.latent_world_model.models.mlp import MLP, orthogonal_init_fn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 3, hidden_dim: int = 512, output_activation: nn.Module = nn.Identity):
        super().__init__()
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=nn.ELU,
            output_activation=output_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
