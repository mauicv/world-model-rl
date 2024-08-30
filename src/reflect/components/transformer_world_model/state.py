from typing import Optional
from reflect.components.base_state import BaseState
from dataclasses import dataclass
import torch.distributions as D
import torch
from reflect.components.transformer_world_model.distribution import create_z_dist, create_norm_dist


@dataclass
class ImaginedRollout(BaseState):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    observations: Optional[torch.Tensor]=None

    def to_ts_tuple(self, ts):
        return (
            self.state[:, -ts:],
            self.action[:, -ts:],
            self.reward[:, -ts:]
        )

    def append(self, state_distribution: 'StateDistribution'):
        sample = state_distribution.rsample()
        return ImaginedRollout(
            state=torch.cat([self.state, sample.features], dim=1),
            action=self.action,
            reward=torch.cat([self.reward, sample.reward], dim=1),
            done=torch.cat([self.done, sample.done], dim=1)
        )

    def append_action(self, action):
        return ImaginedRollout(
            state=self.state,
            action=torch.cat([self.action, action[:, None, :]], dim=1),
            reward=self.reward,
            done=self.done
        )


@dataclass
class StateSample:
    continuous_state: torch.Tensor
    discrete_state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

    @classmethod
    def from_sard(cls, continuous_state, discrete_state, reward, done):
        return cls(
            continuous_state=continuous_state,
            discrete_state=discrete_state,
            reward=reward,
            done=done
        )

    def range(self, start, end):
        return StateSample(
            continuous_state=self.continuous_state[:, start:end],
            discrete_state=self.discrete_state[:, start:end],
            reward=self.reward[:, start:end],
            done=self.done[:, start:end]
        )

    @property
    def features(self):
        return torch.cat([
            self.continuous_state,
            self.discrete_state,
        ], dim=-1)



@dataclass
class StateDistribution:
    continuous_state: D.Distribution
    discrete_state: D.Distribution
    reward_dist: D.Distribution
    done_dist: D.Distribution

    @classmethod
    def from_sard(cls, continuous_mean, continuous_std, discrete, reward, done):
        return cls(
            continuous_state=create_norm_dist(continuous_mean, continuous_std),
            discrete_state=create_z_dist(logits=discrete),
            reward_dist=create_norm_dist(reward),
            done_dist=create_norm_dist(done) 
        )

    def range(self, start, end):
        continuous_mean = self.continuous_state.base_dist.loc[:, start:end]
        continuous_std = self.continuous_state.base_dist.scale[:, start:end]
        discrete = self.discrete_state.base_dist.logits[:, start:end]
        reward = self.reward_dist.base_dist.loc[:, start:end]
        done = self.done_dist.base_dist.loc[:, start:end]
        return StateDistribution(
            continuous_state=create_norm_dist(continuous_mean, continuous_std),
            discrete_state=create_z_dist(discrete),
            reward_dist=create_norm_dist(reward),
            done_dist=create_norm_dist(done)
        )

    def rsample(self):
        b, t, _ = self.continuous_state.mean.shape
        continuous_sample = self.continuous_state.rsample()
        discrete_sample = self.discrete_state.rsample().reshape(b, t, -1)
        reward = self.reward_dist.rsample()
        done = self.done_dist.rsample()
        return StateSample(
            continuous_state=continuous_sample,
            discrete_state=discrete_sample,
            reward=reward,
            done=done
        )

    def last(self, ts):
        return self.range(-ts, None)

    def first(self, ts):
        return self.range(0, ts)


@dataclass
class Sequence(BaseState):
    state_distribution: Optional[StateDistribution]=None
    state_sample: Optional[StateSample]=None
    action: Optional[torch.Tensor]=None

    @classmethod
    def from_sard(cls, continuous_state, discrete_state, reward, done, action=None):
        state = StateSample.from_sard(
            continuous_state,
            discrete_state,
            reward,
            done
        )
        return cls(
            state_sample=state,
            action=action
        )

    @classmethod
    def from_distribution(cls, state: StateDistribution):
        return cls(
            state_distribution=state,
            state_sample=state.rsample()
        )

    def range(self, ts_start, ts_end):
        state_dist = self.state_distribution.range(ts_start, ts_end) if self.state_distribution is not None else None
        state_sample = self.state_sample.range(ts_start, ts_end) if self.state_sample is not None else None
        a = self.action[:, ts_start:ts_end]
        return Sequence(
            state_distribution=state_dist,
            state_sample=state_sample,
            action=a
        )

    def to_sar(self):
        return (
            self.state_sample.features,
            self.action,
            self.state_sample.reward,
        )

    def first(self, ts):
        return self.range(0, ts)

    def last(self, ts):
        return self.range(-ts, None)

    def to_initial_state(self):
        b, t, *_ = self.state_sample.shape
        state = self.state_sample.reshape(b * t, 1, *self.state_sample.shape[2:])
        action = self.action.reshape(b * t, 1, *self.action.shape[2:])
        done = self.done.base_dist.mean.reshape(b * t, 1, *self.done.base_dist.mean.shape[2:])
        reward = self.reward.base_dist.mean.reshape(b * t, 1, *self.reward.base_dist.mean.shape[2:])
        
        return ImaginedRollout(
            state=state,
            action=action,
            done=done,
            reward=reward
        )
