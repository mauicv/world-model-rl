import torch
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.models.world_model.head import Head
from reflect.models.world_model.embedder import Embedder


class DynamicsModel(torch.nn.Module):
    def __init__(
            self,
            hdn_dim,
            num_heads,
            latent_dim=32,
            num_cat=32,
            t_dim=48,
            a_size=1,
            input_dim=1024,
            layers=10,
            dropout=0.05
        ):
        super(DynamicsModel, self).__init__()

        self.gpt = GPT(
            dropout=dropout,
            hidden_dim=hdn_dim,
            num_heads=num_heads,
            embedder=Embedder(
                z_dim=input_dim,
                a_size=a_size,
                hidden_dim=hdn_dim
            ),
            head=Head(
                latent_dim=latent_dim,
                num_cat=num_cat,
                hidden_dim=hdn_dim
            ),
            layers=[
                TransformerLayer(
                    hidden_dim=hdn_dim,
                    attn=RelativeAttention(
                        hidden_dim=hdn_dim,
                        num_heads=num_heads,
                        num_positions=t_dim,
                        dropout=dropout
                    ),
                    mlp=MLP(
                        hidden_dim=hdn_dim,
                        dropout=dropout
                    )
                ) for _ in range(layers)
            ]
        )

    def forward(self, x, mask=None): return self.gpt(x, mask=mask)
