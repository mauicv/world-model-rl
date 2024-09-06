from typing import Optional
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from reflect.components.actor import Actor
from pytfex.transformer.attention import RelativeAttention
from reflect.components.transformer_world_model.head import Head
from reflect.components.transformer_world_model.state import ImaginedRollout, Sequence
from reflect.components.transformer_world_model.embedder import Embedder
from reflect.components.general import DenseModel
from reflect.components.base_state import BaseState
import torch.distributions as D
from dataclasses import dataclass
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
            latent_dim: int,
            num_cat: int,
            num_ts: int,
            input_dim: int,
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
        self.latent_dim=latent_dim
        self.num_cat=num_cat
        self.num_ts=num_ts
        self.input_dim=input_dim
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
                latent_dim=self.latent_dim,
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
        state_logits, reward, done, hdn_state = self.model(
            input.first(ts=self.num_ts).to_sar(),
            mask=self.mask
        )
        return Sequence.from_sard(
            state=state_logits,
            reward=reward,
            done=done,
            hdn_state=hdn_state
        )

    def step(
            self,
            input: ImaginedRollout
        ) -> ImaginedRollout:
        next_state_logits, next_reward, next_done, hdn_state = self.model(
            input.to_ts_tuple(ts=self.num_ts)
        )
        return input.append(
            state_logits=next_state_logits,
            reward_mean=next_reward,
            done_mean=next_done,
            hdn_state=hdn_state
        )

    def imagine_rollout(
            self,
            initial_state: ImaginedRollout,
            actor: Actor,
            n_steps: int,
        ):
        # TODO: Not using KV cache here. Would be better to use it.
        rollout: ImaginedRollout = initial_state
        for _ in range(n_steps):
            rollout = self.step(rollout)
            last_state = rollout.state_logits[:, -1].detach()
            action = actor(last_state, deterministic=True)
            rollout = rollout.append_action(action=action)
        return rollout