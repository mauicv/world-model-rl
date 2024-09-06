from typing import Optional
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.components.transformer_world_model.head import Head
from reflect.components.transformer_world_model.embedder import Embedder
from reflect.components.general import DenseModel
from reflect.components.base_state import BaseState
import torch.distributions as D
from dataclasses import dataclass
import torch


def create_z_dist(logits, temperature=1):
    assert temperature > 0
    dist = D.OneHotCategoricalStraightThrough(logits=logits / temperature)
    return D.Independent(dist, 1)


def create_norm_dist(mean):
    return D.Independent(D.Normal(mean, torch.ones_like(mean)), 1)


@dataclass
class ImaginedRollout(BaseState):
    state: torch.Tensor
    state_logits: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    observations: Optional[torch.Tensor]=None
    hdn_state: Optional[torch.Tensor]=None

    def to_ts_tuple(self, ts):
        return (
            self.state[:, -ts:],
            self.action[:, -ts:],
            self.reward[:, -ts:]
        )

    def append(self, state_logits, done_mean, reward_mean, hdn_state=None):
        b, *_ = state_logits.shape
        prev_state_logits = state_logits[:, [-1]]
        state = (
            create_z_dist(prev_state_logits)
            .rsample()
            .reshape(b, 1, -1)
        )
        reward = create_norm_dist(reward_mean[:, [-1]]).rsample()
        done = create_norm_dist(done_mean[:, [-1]]).rsample()
        prev_state_logits = prev_state_logits.reshape(b, 1, -1)
        if hdn_state is not None:
            if self.hdn_state is None:
                self.hdn_state = torch.zeros_like(hdn_state)
            hdn_state=torch.cat([self.hdn_state, hdn_state[:, [-1]]], dim=1)
        return ImaginedRollout(
            state=torch.cat([self.state, state], dim=1),
            state_logits=torch.cat([self.state_logits, prev_state_logits], dim=1),
            action=self.action,
            reward=torch.cat([self.reward, reward], dim=1),
            done=torch.cat([self.done, done], dim=1),
            hdn_state=hdn_state
        )

    def append_action(self, action):
        return ImaginedRollout(
            state=self.state,
            action=torch.cat([self.action, action[:, None, :]], dim=1),
            reward=self.reward,
            done=self.done,
            state_logits=self.state_logits,
            hdn_state=self.hdn_state
        )

    def to_decoder_input(self):
        return torch.cat([self.state, self.hdn_state], dim=-1)


@dataclass
class Sequence(BaseState):
    state_dist: D.Distribution
    reward: D.Distribution
    done: D.Distribution
    hdn_state: Optional[torch.Tensor]=None
    state_sample: Optional[torch.Tensor]=None
    action: Optional[torch.Tensor]=None

    @classmethod
    def from_sard(cls, state, reward, done, hdn_state=None, action=None):
        b, t, *_ = state.shape
        state_dist = create_z_dist(state)
        state_sample = state_dist.rsample().reshape(b, t, -1)
        return cls(
            state_dist=state_dist,
            hdn_state=hdn_state,
            state_sample=state_sample,
            reward=D.Independent(D.Normal(reward, torch.ones_like(reward)), 1),
            done=D.Independent(D.Normal(done, torch.ones_like(done)), 1),
            action=action
        )

    def range(self, ts_start, ts_end):
        z = self.state_dist.base_dist.logits[:, ts_start:ts_end]
        s = self.state_sample[:, ts_start:ts_end]
        r = self.reward.base_dist.mean[:, ts_start:ts_end]
        d = self.done.base_dist.mean[:, ts_start:ts_end]
        a = self.action[:, ts_start:ts_end]
        hdn_state = self.hdn_state[:, ts_start:ts_end] if self.hdn_state is not None else None
        return Sequence(
            state_dist=create_z_dist(z),
            state_sample=s,
            reward=D.Independent(D.Normal(r, torch.ones_like(r)), 1),
            done=D.Independent(D.Normal(d, torch.ones_like(d)), 1),
            action=a,
            hdn_state=hdn_state
        )

    def to_sar(self):
        return (
            self.state_sample,
            self.action,
            self.reward.base_dist.mean
        )

    def first(self, ts):
        return self.range(0, ts)

    def last(self, ts):
        return self.range(-ts, None)

    def to_initial_state(self):
        b, t, *_ = self.state_sample.shape
        state_logits = self.state_dist.base_dist.logits.reshape(b * t, 1, -1)
        state = self.state_sample.reshape(b * t, 1, *self.state_sample.shape[2:])
        action = self.action.reshape(b * t, 1, *self.action.shape[2:])
        done = self.done.base_dist.mean.reshape(b * t, 1, *self.done.base_dist.mean.shape[2:])
        reward = self.reward.base_dist.mean.reshape(b * t, 1, *self.reward.base_dist.mean.shape[2:])
        return ImaginedRollout(
            state_logits=state_logits,
            state=state,
            action=action,
            done=done,
            reward=reward
        )

    def to_decoder_input(self):
        return torch.cat([self.state_sample, self.hdn_state], dim=-1)
