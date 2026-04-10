from typing import Optional
from dataclasses import dataclass
from reflect.components.flow_world_model.dynamic_model import DynamicFlowModel
from reflect.components.base import Base
from reflect.utils import FreezeParameters

import torch
from reflect.utils import AdamOptim


@dataclass
class WorldModelTrainingParams:
    dynamic_coeff: float = 1.0
    lr: float = 1e-3
    grad_clip: float = 1.0


@dataclass
class WorldModelLosses:
    flow_loss: float
    rel_err: float
    grad_norm: float


class WorldModel(Base):
    model_list = ['dynamic_model', 'dynamic_model_opt']

    def __init__(
            self,
            dynamic_model: DynamicFlowModel,
            observation_dim: int,
            action_dim: int,
            environment_action_bound: float,
            params: Optional[WorldModelTrainingParams] = None,
            num_flow_steps: int = 10,
            noise_scale: float = 0.05,
        ):
        super().__init__()
        if params is None:
            params = WorldModelTrainingParams()
        self.params = params
        self.dynamic_model = dynamic_model
        self.environment_action_bound = environment_action_bound
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps
        self.noise_scale = noise_scale
        self.dynamic_model_opt = AdamOptim(
            self.dynamic_model.parameters(),
            lr=params.lr,
            grad_clip=params.grad_clip
        )

    def update(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            params: Optional[WorldModelTrainingParams] = None,
            noise_scale: Optional[float] = None,
            x_source: Optional[torch.Tensor] = None,
        ):
        if params is None:
            params = self.params
        if noise_scale is None:
            noise_scale = self.noise_scale
        b, l, do = o.shape
        n = self.dynamic_model.num_positions
        x_cond = torch.cat(
            [o[:, -2-n:-1], a[:, -2-n:-1]],
            dim=-1
        )
        x_real = o[:, [-1]]
        x_0 = x_source.detach() if x_source is not None else o[:, [-2]]
        t = torch.rand(b, 1, 1, device=x_real.device)
        x_sample = torch.randn_like(x_0) * noise_scale + x_0
        x_interp = (1 - t) * x_sample + t * x_real
        v = (x_real - x_sample).reshape(b, do)
        u = self.dynamic_model.forward(x_cond, x_interp, t)
        assert u.shape == v.shape, f"u shape {u.shape} != v shape {v.shape}"
        flow_loss = ((u - v) ** 2).mean()
        total_loss = params.dynamic_coeff * flow_loss

        rel_err = (
            (u - v).norm(dim=1)
            / (v.norm(dim=1) + 1e-4)
        ).mean()

        grad_norm = self.dynamic_model_opt.backward(total_loss)
        self.dynamic_model_opt.update_parameters()

        return WorldModelLosses(
            flow_loss=flow_loss.cpu().detach().item(),
            rel_err=rel_err.cpu().detach().item(),
            grad_norm=grad_norm.cpu().detach().item(),
        )

    def get_conditioning(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
        ):
        n = self.dynamic_model.num_positions
        x_cond = torch.cat([o[:, -n:], a[:, -n:]], dim=-1)
        return x_cond

    def get_initial_x(
            self,
            o: torch.Tensor,
            noise_scale: Optional[float] = None,
            x_source: Optional[torch.Tensor] = None,
        ):
        if noise_scale is None:
            noise_scale = self.noise_scale
        x_0 = x_source if x_source is not None else o[:, [-1]]
        return torch.randn_like(x_0) * noise_scale + x_0

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
            step_type: str = 'euler',
            num_flow_steps: Optional[int] = None,
            noise_scale: Optional[float] = None,
            x_source: Optional[torch.Tensor] = None,
        ):
        if num_flow_steps is None:
            num_flow_steps = self.num_flow_steps
        if noise_scale is None:
            noise_scale = self.noise_scale

        delta = float(1 / num_flow_steps)
        x_cond = self.get_conditioning(o, a)
        x = self.get_initial_x(o, noise_scale=noise_scale, x_source=x_source)
        t = torch.zeros(x.shape[0], 1, 1, device=x.device)
        for _ in range(num_flow_steps):
            x, t = self._step_flow(x_cond, x, t, delta, step_type)
        return x

    def correct(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            o_decoded: torch.Tensor,
            **kwargs,
        ):
        """Refine a decoded observation using accumulated rollout context.

        Args:
            o: corrected observations accumulated so far [b, t, obs_dim]
            a: actions accumulated so far [b, t, a_dim]
            o_decoded: latest decoded obs to refine [b, 1, obs_dim]

        Returns:
            corrected observation [b, 1, obs_dim]
        """
        n = self.dynamic_model.num_positions
        o_ctx = o[:, -n:]
        a_ctx = a[:, -n:]
        if o_ctx.shape[1] < n:
            pad = n - o_ctx.shape[1]
            o_ctx = torch.cat([o_ctx[:, [0]].expand(-1, pad, -1), o_ctx], dim=1)
            a_ctx = torch.cat([a_ctx[:, [0]].expand(-1, pad, -1), a_ctx], dim=1)
        return self.step_dynamics(
            o=o_ctx,
            a=a_ctx,
            x_source=o_decoded,
            **kwargs,
        )
