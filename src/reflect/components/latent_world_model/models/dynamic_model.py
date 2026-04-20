import torch
import torch.nn as nn

from reflect.components.latent_world_model.models.mlp import MLP

class MLPDynamicModel(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            num_layers: int = 2,
            hidden_dim: int = 512,
            predict_done: bool = True,
        ):
        super().__init__()

        self._reward_model = MLP(
            input_dim=latent_dim + action_dim,
            output_dim=1,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=nn.ELU,
        )

        self._done_model = MLP(
            input_dim=latent_dim + action_dim,
            output_dim=1,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=nn.ELU,
        ) if predict_done else None

        self._z_model = MLP(
            input_dim=latent_dim + action_dim,
            output_dim=latent_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            activation=nn.ELU,
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor):
        x = torch.cat([z, a], dim=-1)
        z_out = self._z_model(x)
        reward_out = self._reward_model(x)
        done_out = torch.sigmoid(self._done_model(x)) if self._done_model is not None else None
        return z_out, reward_out, done_out

