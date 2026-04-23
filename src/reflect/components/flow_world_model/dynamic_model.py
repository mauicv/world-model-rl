from reflect.components.flow_world_model.cross_attention import (
    AttnActivation,
    CrossAttention,
)

from torch.distributions import Normal
from torch import nn
import torch
import math


def time_embedding(t, embed_dim=64, max_freq=1000.0):
    if t.dim() == 1:
        t = t[:, None]
    half = embed_dim // 2
    freqs = torch.exp(
        torch.linspace(0, math.log(max_freq), half, device=t.device)
    )[None, :]

    angles = 2 * math.pi * t * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb

def make_layer_basic(in_size, out_size, use_layer_norm):
    layers = []
    layers.append(nn.Linear(in_size, out_size))
    if use_layer_norm:
        layers.append(nn.LayerNorm(out_size))
    layers.append(nn.SiLU()) 
    return layers


class DynamicFlowModel(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            conditioning_dim,
            output_dim,
            time_embed_dim,
            hidden_dim,
            depth,
            use_layer_norm,
            num_positions
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.use_layer_norm = use_layer_norm
        self.depth = depth
        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim
        self.num_positions = num_positions

        self.pos_embed = PositionEmbeddingLayer(hidden_dim, num_positions)
        self.time_mlp = nn.Sequential(
            *make_layer_basic(
                self.time_embed_dim,
                self.time_embed_dim,
                use_layer_norm=self.use_layer_norm
            )
        )

        layers = [
            *make_layer_basic(
                self.conditioning_dim + self.output_dim + self.time_embed_dim,
                self.hidden_dim,
                use_layer_norm=self.use_layer_norm
            )
        ]

        for i in range(depth - 1):
            layers.extend(
                make_layer_basic(
                    self.hidden_dim,
                    self.hidden_dim,
                    use_layer_norm=self.use_layer_norm
                )
            )

        self.x_output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cond, x, t):
        b = x.shape[0]
        t = time_embedding(t, self.time_embed_dim)
        t = self.time_mlp(t)
        x = torch.cat([
            x_cond.reshape(b, -1),
            t.reshape(b, -1),
            x.reshape(b, -1)
        ], dim=-1)
        x = self.mlp(x)
        u = self.x_output_layer(x)
        u = u.reshape(b, self.output_dim)
        return u


def make_layer_transformer_mlp(in_size, hidden_dim, out_size, use_layer_norm):
    layers = []
    layers.append(nn.Linear(in_size, hidden_dim))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.SiLU()) 
    layers.append(nn.Linear(hidden_dim, out_size))
    if use_layer_norm:
        layers.append(nn.LayerNorm(out_size))
    layers.append(nn.SiLU()) 
    return layers


class CrossAttentionLayer(torch.nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            use_layer_norm,
            dropout=0.01,
            attn_activation: AttnActivation = "softmax",
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln_x_attn = nn.LayerNorm(hidden_dim)
            self.ln_c_attn = nn.LayerNorm(hidden_dim)
            self.ln_x_mlp = nn.LayerNorm(hidden_dim)
        else:
            self.ln_x_attn = nn.Identity()
            self.ln_c_attn = nn.Identity()
            self.ln_x_mlp = nn.Identity()
        self.attention = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attn_activation=attn_activation,
        )
        self.mlp = nn.Sequential(
            *make_layer_transformer_mlp(
                hidden_dim,
                4*hidden_dim,
                hidden_dim,
                use_layer_norm=use_layer_norm
            )
        )

    def forward(self, x_cond, x):
        x = x + self.attention(
            self.ln_c_attn(x_cond),
            self.ln_x_attn(x)
        )
        x = x + self.mlp(
            self.ln_x_mlp(x)
        )
        return x


class TimeEmbeddingLayer(torch.nn.Module):
    def __init__(self, hidden_dim, use_layer_norm):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.time_mlp = nn.Sequential(
            *make_layer_basic(
                self.hidden_dim,
                self.hidden_dim,
                use_layer_norm=self.use_layer_norm
            )
        )

    def forward(self, t):
        t = time_embedding(t, self.hidden_dim)
        t = self.time_mlp(t)
        return t


class PositionEmbeddingLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_positions):
        super().__init__()
        self.pos_embed = torch.nn.Embedding(
            num_positions,
            hidden_dim
        )

    def forward(self, x):
        pos = (torch
            .arange(0, x.shape[1])
            .expand(x.shape[0], -1)
            .to(x.device)
        )
        pos = self.pos_embed(pos)
        return x + pos


class DynamicAttentionalFlowModel(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            conditioning_dim,
            output_dim,
            num_heads,
            hidden_dim,
            num_positions,
            depth,
            use_layer_norm,
            dropout=0.01,
            attn_activation: AttnActivation = "softmax",
        ):
        super().__init__()
        self.input_dim = input_dim
        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.depth = depth
        self.num_positions = num_positions

        self.pos_embed = PositionEmbeddingLayer(hidden_dim, num_positions)
        self.time_dim_proj = TimeEmbeddingLayer(hidden_dim, use_layer_norm)

        self.cond_dim_proj = nn.Sequential(
            *make_layer_basic(
                self.conditioning_dim,
                self.hidden_dim,
                use_layer_norm=self.use_layer_norm
            )
        )

        self.x_dim_proj = nn.Sequential(
            *make_layer_basic(
                self.input_dim,
                self.hidden_dim,
                use_layer_norm=self.use_layer_norm
            )
        )

        self.layers = nn.ModuleList()
        for i in range(depth - 1):
            self.layers.append(
                CrossAttentionLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    use_layer_norm=self.use_layer_norm,
                    dropout=dropout,
                    attn_activation=attn_activation,
                )
            )

        self.x_output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x_cond, x, t):
        b = x.shape[0]
        t = self.time_dim_proj(t)
        x_cond = self.cond_dim_proj(x_cond)
        x_cond = self.pos_embed(x_cond)
        x = self.x_dim_proj(x)
        x = x + t
        x_cond = x_cond + t
        for layer in self.layers:
            x = layer(x_cond, x)
        u = self.x_output_layer(x)
        u = u.reshape(b, self.output_dim)
        return u
