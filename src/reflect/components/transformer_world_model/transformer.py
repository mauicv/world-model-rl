from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from reflect.components.actor import Actor
from pytfex.transformer.attention import RelativeAttention
from reflect.components.transformer_world_model.head import Head
from reflect.components.transformer_world_model.state import ImaginedRollout, Sequence
from reflect.components.transformer_world_model.embedder import Embedder
from reflect.components.general import DenseModel
import torch.distributions as D
import torch


def get_causal_mask(l):
    mask = torch.tril(torch.ones(l, l))
    masked_indices = mask[None, None, :l, :l] == 0
    return masked_indices


class Transformer(torch.nn.Module):
    def __init__(
            self,
            hdn_dim: int,
            num_heads: int,
            discrete_latent_dim: int,
            continuous_latent_dim: int,
            num_cat: int,
            num_ts: int,
            layers: int,
            dropout: float,
            action_size: int,
            predictor: DenseModel,
            reward_model: DenseModel,
            done_model: DenseModel,
        ) -> None:
        super().__init__()

        self.hdn_dim=hdn_dim
        self.num_heads=num_heads
        self.discrete_latent_dim=discrete_latent_dim
        self.continuous_latent_dim=continuous_latent_dim
        self.num_cat=num_cat
        self.num_ts=num_ts
        self.input_dim=self.discrete_latent_dim * self.num_cat + self.continuous_latent_dim
        self.layers=layers
        self.dropout=dropout
        self.action_size=action_size

        self.mask = get_causal_mask(self.num_ts * 3)

        self.model = GPT(
            dropout=self.dropout,
            hidden_dim=self.hdn_dim,
            num_heads=self.num_heads,
            embedder=Embedder(
                z_dim=self.input_dim,
                a_size=self.action_size,
                hidden_dim=self.hdn_dim
            ),
            head=Head(
                discrete_latent_dim=self.discrete_latent_dim,
                continuous_latent_dim=self.continuous_latent_dim,
                num_cat=self.num_cat,
                hidden_dim=self.hdn_dim,
                predictor=predictor,
                reward_model=reward_model,
                done_model=done_model,
            ),
            layers=[
                TransformerLayer(
                    hidden_dim=self.hdn_dim,
                    attn=RelativeAttention(
                        hidden_dim=self.hdn_dim,
                        num_heads=self.num_heads,
                        num_positions=self.num_ts * 3,
                        dropout=self.dropout
                    ),
                    mlp=MLP(
                        hidden_dim=self.hdn_dim,
                        dropout=self.dropout
                    )
                ) for _ in range(self.layers)
            ]
        )

    def to(self, device):
        self.mask = self.mask.to(device)
        return self.model.to(device)

    def forward(self, input: Sequence) -> Sequence:
        state_dist = self.model(
            input.first(ts=self.num_ts).to_sar(),
            mask=self.mask
        )
        return Sequence.from_distribution(state=state_dist)

    def step(
            self,
            input: ImaginedRollout
        ) -> ImaginedRollout:
        state = self.model(input.to_ts_tuple(ts=self.num_ts))
        return input.append(state_distribution=state.last(ts=1))

    def imagine_rollout(
            self,
            initial_state: ImaginedRollout,
            actor: Actor,
            n_steps: int,
        ):
        # TODO: Not using KV cache here. Would be better to use it.
        rollout = initial_state
        for _ in range(n_steps):
            rollout = self.step(rollout)
            last_state = rollout.dist_features[:, -1].detach()
            action = actor(last_state, deterministic=True)
            rollout = rollout.append_action(action=action)
        return rollout