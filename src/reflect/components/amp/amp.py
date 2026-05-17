from dataclasses import dataclass

import torch
import torch.autograd as autograd

from reflect.components.latent_world_model.models.mlp import MLP
from reflect.utils import AdamOptim


@dataclass
class AMPLosses:
    discriminator_loss: float
    gradient_penalty: float
    grad_norm: float


class AMP:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        lr: float = 1e-4,
        grad_penalty_coef: float = 10.0,
        grad_clip: float = 1.0,
    ):
        self.grad_penalty_coef = grad_penalty_coef
        self.discriminator = MLP(
            input_dim=state_dim * 2,
            output_dim=1,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
        self.optim = AdamOptim(
            self.discriminator.parameters(),
            lr=lr,
            grad_clip=grad_clip,
        )

    def update(
        self,
        ref_pairs: tuple[torch.Tensor, torch.Tensor],
        env_pairs: tuple[torch.Tensor, torch.Tensor],
    ) -> AMPLosses:
        ref_input = self._pack_pair(*ref_pairs).detach().requires_grad_(True)
        d_ref = self.discriminator(ref_input)

        grads = autograd.grad(
            outputs=d_ref.sum(),
            inputs=ref_input,
            create_graph=True,
            retain_graph=True,
        )[0]
        gp = self.grad_penalty_coef * grads.pow(2).mean()

        l_ref = 0.5 * ((d_ref - 1) ** 2).mean()

        env_input = self._pack_pair(*env_pairs)
        d_env = self.discriminator(env_input)
        l_env = 0.5 * ((d_env + 1) ** 2).mean()

        loss = l_ref + l_env + gp
        grad_norm = self.optim.backward(loss)
        self.optim.update_parameters()

        return AMPLosses(
            discriminator_loss=(l_ref + l_env).item(),
            gradient_penalty=gp.item(),
            grad_norm=grad_norm.item(),
        )

    def compute_reward(self, states: torch.Tensor) -> torch.Tensor:
        # states: (b, t, state_dim) -> returns (b, t-1, 1)
        s_i = states[:, :-1]
        s_next = states[:, 1:]
        b, t, state_dim = s_i.shape
        pairs = self._pack_pair(
            s_i.reshape(b * t, state_dim),
            s_next.reshape(b * t, state_dim),
        )
        with torch.no_grad():
            d = self.discriminator(pairs)
        rewards = torch.clamp(1 - 0.25 * (d - 1) ** 2, min=0)
        return rewards.reshape(b, t, 1)

    def _pack_pair(self, s_i: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        return torch.cat([s_i, s_next], dim=-1)
