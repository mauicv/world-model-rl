from typing import Tuple
import torch
from dataclasses import dataclass
import torch.distributions as D
from reflect.components.rssm_world_model.state.base import Base


@dataclass
class InternalState:
    deter_state: torch.Tensor
    stoch_state: torch.Tensor
    logits: torch.Tensor

    def detach(self) -> 'InternalState':
        return InternalState(
            deter_state=self.deter_state.detach(),
            stoch_state=self.stoch_state.detach(),
            logits=self.logits.detach(),
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
        self.logits=self.logits.to(device)


@dataclass
class InternalStateSequence:
    deter_states: torch.Tensor
    stoch_states: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def from_init(cls, init_state: InternalState):
        return cls(
            deter_states=init_state.deter_state.unsqueeze(1),
            stoch_states=init_state.stoch_state.unsqueeze(1),
            logits=init_state.logits.unsqueeze(1),
        )

    @property
    def shapes(self) -> Tuple[int, int]:
        return self.deter_states.shape, self.stoch_states.shape

    def append_(self, other: InternalState):
        self.deter_states = torch.cat([self.deter_states, other.deter_state.unsqueeze(1)], dim=1)
        self.stoch_states = torch.cat([self.stoch_states, other.stoch_state.unsqueeze(1)], dim=1)
        self.logits = torch.cat([self.logits, other.logits.unsqueeze(1)], dim=1)

    def __getitem__(self, index):
        return InternalState(
            deter_state=self.deter_states[:, index],
            stoch_state=self.stoch_states[:, index],
            logits=self.logits[:, index],
        )

    def get_last(self):
        return self[-1]

    def get_features(self):
        return torch.cat([
            self.deter_states[:, 1:],
            self.stoch_states[:, 1:]
        ], dim=-1)

    def get_dist(self):
        return D.Independent(D.Categorical(
            self.logits[:, 1:],
        ), 1)

    def flatten_batch_time(self) -> InternalState:
        deter_states = self.deter_states[:, 1:]
        stoch_states = self.stoch_states[:, 1:]
        logits = self.logits[:, 1:]
        return InternalState(
            deter_state=deter_states.reshape(-1, *deter_states.shape[2:]),
            stoch_state=stoch_states.reshape(-1, *stoch_states.shape[2:]),
            logits=logits.reshape(-1, *logits.shape[2:]),
        )

    def to(self, device):
        self.deter_states = self.deter_states.to(device)
        self.stoch_states = self.stoch_states.to(device)
        self.logits = self.logits.to(device)
