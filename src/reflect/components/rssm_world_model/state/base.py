from typing import Tuple
import torch
from dataclasses import dataclass
import torch.distributions as D


@dataclass
class Base:
    deter_state: torch.Tensor
    stoch_state: torch.Tensor

    def detach(self) -> 'Base':
        cls_args = {k: v.detach() for k, v in self.__dict__.items()}
        return self.__class__(**cls_args)

    @property
    def shapes(self) -> Tuple[int, int]:
        return tuple(f'{k}:{v.shape}' for k, v in self.__dict__.items())
    
    def get_features(self):
        return torch.cat([
            self.deter_state,
            self.stoch_state
        ], dim=-1)

    def to(self, device: torch.device):
        for k, v in self.__dict__.items():
            setattr(self, k, v.to(device))
