from typing import Tuple
import torch
from dataclasses import dataclass
import torch.distributions as D
from reflect.components.base_state import BaseState


@dataclass
class InternalStateDiscrete(BaseState):
    deter_state: torch.Tensor
    stoch_state: torch.Tensor
    logits: torch.Tensor
    
    def get_features(self):
        return torch.cat([
            self.deter_state,
            self.stoch_state
        ], dim=-1)

    @classmethod
    def from_logits(
            cls,
            deter_state: torch.Tensor,
            logits: torch.Tensor,
            temperature: float = 1.0,
        ):
        dist = D.Independent(D.OneHotCategoricalStraightThrough(
            logits=logits / temperature
        ), 1)
        stoch_state: torch.Tensor = dist.rsample()
        b, c, p = stoch_state.shape
        stoch_state = stoch_state.reshape(b, c * p)
        return cls(
            deter_state=deter_state,
            stoch_state=stoch_state,
            logits=logits,
        )

    def to_sequence(self):
        return InternalStateDiscreteSequence.from_init(self)

@dataclass
class InternalStateDiscreteSequence(BaseState):
    deter_states: torch.Tensor
    stoch_states: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def from_init(cls, init_state: InternalStateDiscrete):
        return cls(
            deter_states=init_state.deter_state.unsqueeze(1),
            stoch_states=init_state.stoch_state.unsqueeze(1),
            logits=init_state.logits.unsqueeze(1),
        ) 

    def append_(self, other: InternalStateDiscrete):
        self.deter_states = torch.cat([self.deter_states, other.deter_state.unsqueeze(1)], dim=1)
        self.stoch_states = torch.cat([self.stoch_states, other.stoch_state.unsqueeze(1)], dim=1)
        self.logits = torch.cat([self.logits, other.logits.unsqueeze(1)], dim=1)

    def __getitem__(self, index):
        return InternalStateDiscrete(
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

    def get_dist(self, temperature=1.0):
        dist = D.OneHotCategoricalStraightThrough(logits=self.logits[:, 1:] / temperature)
        return D.Independent(dist, 1)

    def flatten_batch_time(self) -> InternalStateDiscrete:
        deter_states = self.deter_states[:, 1:]
        stoch_states = self.stoch_states[:, 1:]
        logits = self.logits[:, 1:]
        return InternalStateDiscrete(
            deter_state=deter_states.reshape(-1, *deter_states.shape[2:]),
            stoch_state=stoch_states.reshape(-1, *stoch_states.shape[2:]),
            logits=logits.reshape(-1, *logits.shape[2:]),
        )
