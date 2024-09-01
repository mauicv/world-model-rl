from typing import Optional
from reflect.components.base_state import BaseState
from dataclasses import dataclass
import torch.distributions as D
import torch
from reflect.components.transformer_world_model.distribution import create_z_dist, create_norm_dist


@dataclass
class ImaginedRollout(BaseState):
    state_features: torch.Tensor
    dist_features: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    observations: Optional[torch.Tensor]=None

    def to_ts_tuple(self, ts):
        return (
            self.state_features[:, -ts:],
            self.action[:, -ts:],
            self.reward[:, -ts:]
        )

    def append(self, state_distribution: 'StateDistribution'):
        sample = state_distribution.rsample()
        return ImaginedRollout(
            dist_features=torch.cat([self.dist_features, state_distribution.features], dim=1),
            state_features=torch.cat([self.state_features, sample.features], dim=1),
            action=self.action,
            reward=torch.cat([self.reward, sample.reward], dim=1),
            done=torch.cat([self.done, sample.done], dim=1)
        )

    def append_action(self, action):
        return ImaginedRollout(
            dist_features=self.dist_features,
            state_features=self.state_features,
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
    reward_dist: Optional[D.Distribution]=None
    done_dist: Optional[D.Distribution]=None

    @classmethod
    def from_sard(cls, continuous_mean, continuous_std, discrete, reward, done):
        continuous_std = torch.nn.functional.softplus(continuous_std) + 0.1
        return cls(
            continuous_state=create_norm_dist(continuous_mean, continuous_std),
            discrete_state=create_z_dist(logits=discrete),
            reward_dist=create_norm_dist(reward),
            done_dist=create_norm_dist(done) 
        )

    @classmethod
    def from_dist(cls, continuous_mean, continuous_std, discrete):
        continuous_std = torch.nn.functional.softplus(continuous_std) + 0.1
        return cls(
            continuous_state=create_norm_dist(continuous_mean, continuous_std),
            discrete_state=create_z_dist(logits=discrete),
        )

    @property
    def features(self):
        b, t, _ = self.continuous_state.base_dist.loc.shape
        return torch.cat([
            self.discrete_state.base_dist.logits.reshape(b, t, -1),
            self.continuous_state.base_dist.mean,
            self.continuous_state.base_dist.scale,
        ], dim=-1)

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
    def from_distribution(cls, state: StateDistribution, action: Optional[torch.Tensor]=None):
        return cls(
            state_distribution=state,
            state_sample=state.rsample(),
            action=action
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

    def rsample(self):
        assert self.state_distribution, "No state distribution to sample from"
        self.state_sample = self.state_distribution.rsample()

    def to_initial_state(self):
        state_features = self.state_sample.features
        dist_features = self.state_distribution.features
        done_state = self.state_sample.done
        reward_state = self.state_sample.reward
        b, t, *_ = state_features.shape
        state_features = state_features.reshape(b * t, 1, *state_features.shape[2:])
        dist_features = dist_features.reshape(b * t, 1, *dist_features.shape[2:])
        action = self.action.reshape(b * t, 1, *self.action.shape[2:])
        done = done_state.reshape(b * t, 1, *done_state.shape[2:])
        reward = reward_state.reshape(b * t, 1, *reward_state.shape[2:])
        return ImaginedRollout(
            dist_features=dist_features,
            state_features=state_features,
            action=action,
            done=done,
            reward=reward
        )
