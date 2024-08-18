from typing import Tuple
import torch
from dataclasses import dataclass
import torch.distributions as D


@dataclass
class InternalState:
    deter_state: torch.Tensor
    stoch_state: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def detach(self) -> 'InternalState':
        return InternalState(
            deter_state=self.deter_state.detach(),
            stoch_state=self.stoch_state.detach(),
            mean=self.mean.detach(),
            std=self.std.detach()
        )

    @property
    def shapes(self) -> Tuple[int, int]:
        return self.deter_state.shape, self.stoch_state.shape
    
    def get_features(self):
        return torch.cat([
            self.deter_state,
            self.stoch_state
        ], dim=-1)

    def to(self, device: torch.device):
        self.deter_state=self.deter_state.to(device)
        self.stoch_state=self.stoch_state.to(device)
        self.mean=self.mean.to(device)
        self.std=self.std.to(device)


@dataclass
class InternalStateSequence:
    deter_states: torch.Tensor
    stoch_states: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor

    @classmethod
    def from_init(cls, init_state: InternalState):
        return cls(
            deter_states=init_state.deter_state.unsqueeze(1),
            stoch_states=init_state.stoch_state.unsqueeze(1),
            means=init_state.mean.unsqueeze(1),
            stds=init_state.std.unsqueeze(1)
        )

    @property
    def shapes(self) -> Tuple[int, int]:
        return self.deter_states.shape, self.stoch_states.shape

    def append_(self, other: InternalState):
        self.deter_states = torch.cat([self.deter_states, other.deter_state.unsqueeze(1)], dim=1)
        self.stoch_states = torch.cat([self.stoch_states, other.stoch_state.unsqueeze(1)], dim=1)
        self.means = torch.cat([self.means, other.mean.unsqueeze(1)], dim=1)
        self.stds = torch.cat([self.stds, other.std.unsqueeze(1)], dim=1)

    def __getitem__(self, index):
        return InternalState(
            deter_state=self.deter_states[:, index],
            stoch_state=self.stoch_states[:, index],
            mean=self.means[:, index],
            std=self.stds[:, index]
        )

    def get_last(self):
        return self[-1]

    def get_features(self):
        return torch.cat([
            self.deter_states[:, 1:],
            self.stoch_states[:, 1:]
        ], dim=-1)

    def get_dist(self):
        normal = D.Normal(
            self.means[:, 1:],
            self.stds[:, 1:]
        )
        return D.Independent(normal, 1)

    def flatten_batch_time(self) -> InternalState:
        deter_states = self.deter_states[:, 1:]
        stoch_states = self.stoch_states[:, 1:]
        means = self.means[:, 1:]
        stds = self.stds[:, 1:]
        return InternalState(
            deter_state=deter_states.reshape(-1, *deter_states.shape[2:]),
            stoch_state=stoch_states.reshape(-1, *stoch_states.shape[2:]),
            mean=means.reshape(-1, *means.shape[2:]),
            std=stds.reshape(-1, *stds.shape[2:])
        )

    def to(self, device):
        self.deter_states = self.deter_states.to(device)
        self.stoch_states = self.stoch_states.to(device)
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)
