from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.components.transformer_world_model.head import StackHead, ConcatHead, AddHead
from reflect.components.transformer_world_model.embedder import StackEmbedder, ConcatEmbedder, AddEmbedder
from pytfex.transformer.gpt import GPT
import torch
from typing import Literal, Optional



def get_causal_mask(l):
    mask = torch.tril(torch.ones(l, l))
    masked_indices = mask[None, None, :l, :l] == 0
    return masked_indices


head_embedder_class_map = {
    'stack': (StackHead, StackEmbedder, 3, 1),
    'concat': (ConcatHead, ConcatEmbedder, 1, 3),
    'add': (AddHead, AddEmbedder, 1, 1),
}


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
            hdn_dim: int=512,
            embedding_type: Literal['stack', 'add', 'concat']='stack'
        ):
        super().__init__()
        self.num_ts = num_ts
        self.num_cat = num_cat
        self.latent_dim = latent_dim
        head_cls, embedder_cls, ts_adjuster, hdn_adj = head_embedder_class_map[embedding_type]
        num_ts = self.num_ts * ts_adjuster
        internal_hdn_dim = hdn_adj * hdn_dim
        self.mask = get_causal_mask(num_ts)

        self.dynamic_model = GPT(
            dropout=dropout,
            hidden_dim=hdn_dim,
            num_heads=num_heads,
            embedder=embedder_cls(
                z_dim=latent_dim*num_cat,
                a_size=action_size,
                hidden_dim=hdn_dim
            ),
            head=head_cls(
                latent_dim=latent_dim,
                num_cat=num_cat,
                hidden_dim=hdn_dim
            ),
            layers=[
                TransformerLayer(
                    hidden_dim=internal_hdn_dim,
                    attn=RelativeAttention(
                        hidden_dim=internal_hdn_dim,
                        num_heads=num_heads,
                        num_positions=num_ts,
                        dropout=dropout
                    ),
                    mlp=MLP(
                        hidden_dim=internal_hdn_dim,
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
            h: Optional[torch.Tensor] = None,
        ):
        new_h, z_dist, new_r, new_d = self.dynamic_model((
            z[:, -self.num_ts:],
            a[:, -self.num_ts:],
            r[:, -self.num_ts:]
        ))

        if h is not None:
            new_h = torch.cat([h, new_h[:, [-1]]], dim=1)

        new_r = new_r[:, -1].reshape(-1, 1, 1)
        r = torch.cat([r, new_r], dim=1)

        new_d = new_d[:, -1].reshape(-1, 1, 1)
        d = torch.cat([d, new_d], dim=1)

        return new_h, z_dist, r, d

    def step(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            h: Optional[torch.Tensor] = None,
        ):
        h, z_dist, new_r, new_d = self._step(z, a, r, d, h)
        new_z = z_dist.sample()
        new_z = new_z[:, -1].reshape(-1, 1, self.num_cat * self.latent_dim)
        new_z = torch.cat([z, new_z], dim=1)
        return h, new_z, new_r, new_d

    def rstep(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            h: Optional[torch.Tensor] = None,
        ):
        h, z_dist, new_r, new_d = self._step(z, a, r, d, h)
        new_z = z_dist.rsample()
        new_z = new_z[:, -1].reshape(-1, 1, self.num_cat * self.latent_dim)
        new_z = torch.cat([z, new_z], dim=1)
        return h, new_z, new_r, new_d

    def forward(self, z, a, r):
        self.mask = self.mask.to(z.device)
        return self.dynamic_model(
            (z, a, r),
            mask=self.mask
        )
