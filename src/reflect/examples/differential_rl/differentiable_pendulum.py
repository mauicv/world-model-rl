import torch
from dataclasses import dataclass
from typing import Optional
import numpy as np


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


@dataclass
class EnvState:
    th: torch.Tensor
    thdot: torch.Tensor
    batch_size: int

    def to_observation(self):
        return torch.stack([
            self.th.sin(),
            self.th.cos(),
            self.thdot
        ], dim=-1)


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class DiffPendulumEnv:
    def __init__(
            self,
            g=10.0,
            max_speed=8,
            max_torque=2.0,
            dt=0.05,
            m=1.0,
            l=1.0,
            seed=None
        ):
        self.g_force = torch.tensor(g, dtype=torch.float)
        self.max_speed = torch.tensor(max_speed, dtype=torch.float)
        self.max_torque = torch.tensor(max_torque, dtype=torch.float)
        self.dt = torch.tensor(dt, dtype=torch.float)
        self.m = torch.tensor(m, dtype=torch.float)
        self.l = torch.tensor(l, dtype=torch.float)
        self.state = None
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def reset(self, batch_size=1):
        high_th = torch.tensor(DEFAULT_X)
        high_thdot = torch.tensor(DEFAULT_Y)
        low_th = -high_th
        low_thdot = -high_thdot
        th = (
            torch.rand(batch_size, generator=self.rng)
            * (high_th - low_th)
            + low_th
        )
        thdot = (
            torch.rand(batch_size, generator=self.rng)
            * (high_thdot - low_thdot)
            + low_thdot
        )
        self.state = EnvState(th=th, thdot=thdot, batch_size=batch_size)
        observation = self.state.to_observation()
        return observation, torch.zeros_like(th)

    def step(self, action):
        assert self.state is not None, "Call reset before step"
        assert action.shape[0] == self.state.batch_size, "Action shape mismatch"
        u = action.clamp(-self.max_torque, self.max_torque)
        th, thdot = self.state.th, self.state.thdot
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        new_thdot = (
            thdot
            + (3 * self.g_force / (2 * self.l) * th.sin()
            + 3.0 / (self.m * self.l**2) * u) * self.dt
        )
        new_thdot = new_thdot.clamp(-self.max_speed, self.max_speed)
        new_th = th + new_thdot * self.dt
        reward = -costs.view(self.state.batch_size, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)
        self.state = EnvState(
            th=new_th,
            thdot=new_thdot,
            batch_size=self.state.batch_size
        )
        observation = self.state.to_observation()
        return observation, reward, done, {}
