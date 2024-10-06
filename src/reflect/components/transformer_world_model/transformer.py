from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.components.transformer_world_model.head import Head
from reflect.components.transformer_world_model.embedder import Embedder
from pytfex.transformer.gpt import GPT
import torch



def get_causal_mask(l):
    mask = torch.tril(torch.ones(l, l))
    masked_indices = mask[None, None, :l, :l] == 0
    return masked_indices


class PytfexTransformer(torch.nn.Module):
    def __init__(
            self,
            num_ts: int,
            num_cat: int=32,
            dropout: float=0.1,
            num_heads: int=8,
            latent_dim: int=32,
            action_size: int=6,
            num_layers: int=12,
            hdn_dim: int=512
        ):
        super().__init__()
        self.num_ts = num_ts
        self.num_cat = num_cat
        self.latent_dim = latent_dim
        self.mask = get_causal_mask(self.num_ts * 3)

        self.dynamic_model = GPT(
            dropout=dropout,
            hidden_dim=hdn_dim,
            num_heads=num_heads,
            embedder=Embedder(
                z_dim=latent_dim*num_cat,
                a_size=action_size,
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
                        num_positions=3*num_ts,
                        dropout=dropout
                    ),
                    mlp=MLP(
                        hidden_dim=hdn_dim,
                        dropout=dropout
                    )
                ) for _ in range(num_layers)
            ]
        )

    def _step(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
        ):
        z_dist, new_r, new_d = self.dynamic_model((
            z[:, -self.num_ts:],
            a[:, -self.num_ts:],
            r[:, -self.num_ts:]
        ))

        new_r = new_r[:, -1].reshape(-1, 1, 1)
        r = torch.cat([r, new_r], dim=1)

        new_d = new_d[:, -1].reshape(-1, 1, 1)
        d = torch.cat([d, new_d], dim=1)

        return z_dist, r, d

    def step(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
        ):
        z_dist, new_r, new_d = self._step(z, a, r, d)
        new_z = z_dist.sample()
        new_z = new_z[:, -1].reshape(-1, 1, self.num_cat * self.latent_dim)
        new_z = torch.cat([z, new_z], dim=1)
        return new_z, new_r, new_d

    def rstep(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
        ):
        z_dist, new_r, new_d = self._step(z, a, r, d)
        new_z = z_dist.rsample()
        new_z = new_z[:, -1].reshape(-1, 1, self.num_cat * self.latent_dim)
        new_z = torch.cat([z, new_z], dim=1)
        return new_z, new_r, new_d

    def forward(self, z, a, r):
        self.mask = self.mask.to(z.device)
        return self.dynamic_model(
            (z, a, r),
            mask=self.mask
        )
