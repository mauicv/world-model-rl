import torch
import torch.nn as nn

import torch.nn as nn


def orthogonal_init_fn(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int = 3,
            hidden_dim: int = 512,
            activation: nn.Module = nn.Mish,
            output_activation: nn.Module = nn.Identity,
            layernorm: bool = True,
            orthogonal_init: bool = True,
        ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"
        trunk_layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
            activation(),
        ]
        for _ in range(num_layers - 2):
            trunk_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layernorm else nn.Identity(),
                activation(),
            ])

        trunk_layers.extend([
            nn.Linear(hidden_dim, output_dim),
            layernorm and nn.LayerNorm(output_dim) or nn.Identity(),
            output_activation(),
        ])
        
        self.trunk = nn.Sequential(*trunk_layers)
        if orthogonal_init:
            self.apply(orthogonal_init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)
