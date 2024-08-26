from typing import Tuple
import torch
from dataclasses import dataclass
import torch.distributions as D

def get_shape(v):
    if isinstance(v, torch.Tensor):
        return v.shape
    if isinstance(v, D.Independent):
        if isinstance(v.base_dist, D.Normal):
            return v.base_dist.loc.shape
        if isinstance(v.base_dist, D.OneHotCategoricalStraightThrough):
            return v.base_dist.logits.shape


def detach(v):
    if isinstance(v, torch.Tensor):
        return v.detach()
    if isinstance(v, D.Independent):
        if isinstance(v.base_dist, D.Normal):
            return D.Independent(D.Normal(v.base_dist.loc.detach(), v.base_dist.scale.detach()), 1)
        if isinstance(v.base_dist, D.OneHotCategoricalStraightThrough):
            return D.Independent(D.OneHotCategoricalStraightThrough(v.base_dist.logits.detach()), 1)
    else:
        return v


@dataclass
class BaseState:
    def detach(self) -> 'BaseState':
        cls_args = {k: detach(v) for k, v in self.__dict__.items()}
        return self.__class__(**cls_args)

    @property
    def shapes(self) -> Tuple[int, int]:
        return tuple(get_shape(v) for k, v in self.__dict__.items())

    def to(self, device: torch.device):
        for k, v in self.__dict__.items():
            setattr(self, k, v.to(device))
