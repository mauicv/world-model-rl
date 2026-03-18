from typing import Optional
from dataclasses import dataclass
from reflect.components.flow_world_model.dynamic_model import DynamicFlowModel
from reflect.components.models.actor import Actor
from itertools import chain
from reflect.components.base import Base
from reflect.utils import FreezeParameters

import torch
from reflect.utils import (
    AdamOptim,
)

done_loss_fn = torch.nn.BCELoss()


@dataclass
class WorldModelTrainingParams:
    dynamic_coeff: float = 1.0
    reward_coeff: float = 10.0
    done_coeff: float = 1.0
    time_coeff: float = 0.0
    lr: float = 1e-3
    grad_clip: float = 1.0


@dataclass
class WorldModelLosses:
    flow_loss: float
    rel_err: float
    grad_norm: float


class WorldModel(Base):
    model_list = [
        'dynamic_model',
        'dynamic_model_opt'
    ]
    def __init__(
            self,
            dynamic_model: DynamicFlowModel,
            observation_dim: int,
            action_dim: int,
            environment_action_bound: float,
            params: Optional[WorldModelTrainingParams] = None,
        ):
        super().__init__()
        if params is None:
            params = WorldModelTrainingParams()
        self.params = params
        self.dynamic_model = dynamic_model
        self.environment_action_bound = environment_action_bound
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.dynamic_model_opt = AdamOptim(
            self.dynamic_model.parameters(),
            lr=params.lr,
            grad_clip=params.grad_clip
        )

    def update(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            params: Optional[WorldModelTrainingParams] = None,
            noise_scale: float = 0.05,
        ):
        if params is None:
            params = self.params
        b, l, do = o.shape
        _, _, da = a.shape
        n = self.dynamic_model.num_positions
        x_cond = torch.cat([o[:, -2-n:-1], a[:, -2-n:-1], r[:, -2-n:-1], d[:, -2-n:-1]], dim=-1)
        x_real = torch.cat([o[:, [-1]], r[:, [-1]], d[:, [-1]]], dim=-1)
        x_last = torch.cat([o[:, [-2]], r[:, [-2]], d[:, [-2]]], dim=-1)
        t = torch.rand(b, 1, 1, device=x_real.device)
        x_sample = torch.randn_like(x_last, device=x_real.device)*noise_scale + x_last
        x_interp = (1 - t) * x_sample + t * x_real
        v = (x_real - x_sample).reshape(b, do + 2)
        u = self.dynamic_model.forward(x_cond, x_interp, t)
        assert u.shape == v.shape
        loss = ((u - v) ** 2).mean()
        grad_norm = self.dynamic_model_opt.backward(loss)
        self.dynamic_model_opt.update_parameters()

        rel_err = (
            (u - v).norm(dim=1)
            / (v.norm(dim=1) + 1e-4)
        ).mean()

        return WorldModelLosses(
            flow_loss=loss.cpu().detach().item(),
            grad_norm=grad_norm.cpu().detach().item(),
            rel_err=rel_err.cpu().detach().item(),
        )

    def get_conditioning(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, d: torch.Tensor):
        n = self.dynamic_model.num_positions
        x_cond = torch.cat([o[:, -n:], a[:, -n:], r[:, -n:], d[:, -n:]], dim=-1)
        return x_cond

    def get_initial_x(self, o: torch.Tensor, r: torch.Tensor, d: torch.Tensor, noise_scale: float = 0.05):
        x_last = torch.cat([o[:, [-1]], r[:, [-1]], d[:, [-1]]], dim=-1)
        x_sample = torch.randn_like(x_last, device=x_last.device)*noise_scale + x_last
        return x_sample

    def split_x(self, x: torch.Tensor):
        o = x[:, :, :self.observation_dim]
        r = x[:, :, self.observation_dim:self.observation_dim+1]
        d = x[:, :, self.observation_dim+1:]
        return o, r, d

    def _step_flow(
            self,
            x_cond: torch.Tensor,
            x: torch.Tensor,
            t: torch.Tensor,
            delta: float,
            step_type: str = 'euler',
        ):    
        if step_type == 'euler':
            u = self.dynamic_model.forward(x_cond, x, t)
            u = u.unsqueeze(1)
            x_next = x + u * delta
            return x_next, t + delta
        else:
            raise ValueError(f'Invalid step type: {step_type}')

    def step_dynamics(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            step_type: str = 'euler',
            num_flow_steps: int = 10,
            noise_scale: float = 0.05,
        ):
        assert o.shape[-2] == self.dynamic_model.num_positions
        assert a.shape[-2] == self.dynamic_model.num_positions
        assert r.shape[-2] == self.dynamic_model.num_positions
        assert d.shape[-2] == self.dynamic_model.num_positions
        
        delta = float(1/num_flow_steps)
        x_cond = self.get_conditioning(o, a, r, d)
        x = self.get_initial_x(o, r, d, noise_scale=noise_scale)
        t = torch.zeros(x.shape[0], 1, 1, device=x.device)
        for _ in range(num_flow_steps):
            x, t = self._step_flow(x_cond, x, t, delta, step_type)
        o, r, d = self.split_x(x)
        return o, r, d

    def prediction_error_per_sample(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            step_type: str = 'euler',
            num_flow_steps: int = 10,
            noise_scale: float = 0.05,
        ):
        n = self.dynamic_model.num_positions
        assert o.shape[1] >= n + 1
        assert a.shape[1] >= n + 1
        assert r.shape[1] >= n + 1
        assert d.shape[1] >= n + 1

        o_cond = o[:, -(n + 1):-1]
        a_cond = a[:, -(n + 1):-1]
        r_cond = r[:, -(n + 1):-1]
        d_cond = d[:, -(n + 1):-1]

        with torch.no_grad():
            o_pred, r_pred, d_pred = self.step_dynamics(
                o=o_cond,
                a=a_cond,
                r=r_cond,
                d=d_cond,
                step_type=step_type,
                num_flow_steps=num_flow_steps,
                noise_scale=noise_scale,
            )

        x_pred = torch.cat([o_pred, r_pred, d_pred], dim=-1).squeeze(1)
        x_target = torch.cat([o[:, [-1]], r[:, [-1]], d[:, [-1]]], dim=-1).squeeze(1)
        per_sample_error = ((x_pred - x_target) ** 2).mean(dim=1)
        return per_sample_error

    def imagine_rollout(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            actor: Actor,
            num_timesteps: int,
            num_flow_steps: int = 100,
            noise_scale: float = 0.05,
            disable_gradients: bool=False,
        ):
        with torch.set_grad_enabled(not disable_gradients):
            with FreezeParameters([self.dynamic_model]):
                for _ in range(num_timesteps):
                    num_pos = self.dynamic_model.num_positions
                    _o, _a, _r, _d = o[:, -num_pos:], a[:, -num_pos:], r[:, -num_pos:], d[:, -num_pos:]
                    _no, _nr, _nd = self.step_dynamics(_o, _a, _r, _d, num_flow_steps=num_flow_steps, noise_scale=noise_scale)
                    action = actor(_no.detach(), deterministic=True)
                    a = torch.cat([a, action], dim=1)
                    o = torch.cat([o, _no], dim=1)
                    r = torch.cat([r, _nr], dim=1)
                    d = torch.cat([d, _nd], dim=1)
        return o, a, r, d

        