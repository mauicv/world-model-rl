import copy
from typing import Optional
from dataclasses import dataclass
from reflect.components.models.actor import Actor
from itertools import chain
from reflect.components.base import Base
from reflect.utils import FreezeParameters, AdamOptim

import torch
import torch.nn.functional as F



@dataclass
class LatentWorldModelTrainingParams:
    consistency_coeff: float = 1.0
    reward_coeff: float = 1.0
    done_coeff: float = 1.0
    rollout_discount: float = 0.5


@dataclass
class LatentWorldModelLosses:
    consistency_loss: float
    reward_loss: float
    done_loss: float
    grad_norm: float


class LatentWorldModel(Base):
    model_list = [
        'encoder',
        'dynamic_model',
        'optim'
    ]

    def __init__(
            self,
            encoder: torch.nn.Module,
            dynamic_model: torch.nn.Module,
            params: Optional[LatentWorldModelTrainingParams] = None,
            ema_tau: float = 0.005,
            environment_action_bound: float = 1.0,
            use_delta: bool = False,
            learning_rate: float = 3e-4,
        ):
        super().__init__()
        if params is None:
            params = LatentWorldModelTrainingParams()
        self.params = params
        self.encoder = encoder
        self.ema_encoder = copy.deepcopy(encoder)
        self.ema_encoder.requires_grad_(False)
        self.ema_tau = ema_tau
        self.dynamic_model = dynamic_model
        self.environment_action_bound = environment_action_bound
        self.use_delta = use_delta
        optim_param = chain(encoder.parameters(), self.dynamic_model.parameters())
        self.optim = AdamOptim(
            optim_param,
            lr=learning_rate,
            # eps=1e-5,
            grad_clip=100
        )

    def _step(self, z: torch.Tensor, a: torch.Tensor):
        z_out, r, d = self.dynamic_model(z, a)
        if self.use_delta:
            z_out = z + z_out
        return z_out, r, d

    def _update_ema_encoder(self):
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_encoder.parameters(),
                self.encoder.parameters()
            ):
                ema_param.data = self.ema_tau * param.data + (1 - self.ema_tau) * ema_param.data

    def encode(self, state):
        return self.encoder(state)

    def update(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            params: Optional[LatentWorldModelTrainingParams] = None,
            return_init_states: bool = False,
            no_update: bool = False,
        ):
        """
        o: (b, t, *s_dim)
        a: (b, t, a_dim)
        r: (b, t, 1)
        d: (b, t, 1)
        """
        if params is None:
            params = self.params

        b, t, *_ = o.shape

        # Encode first observation as rollout starting point
        z = self.encoder(o[:, 0])  # (b, latent_dim)

        # EMA targets for all subsequent observations
        with torch.no_grad():
            target_z = self.ema_encoder(o[:, 1:])  # (b, t-1, latent_dim)

        # Roll forward t-1 steps using own predictions
        predicted_zs = []
        predicted_rs = []
        predicted_ds = []
        for h in range(t - 1):
            z, r_pred, d_pred = self._step(z, a[:, h])
            predicted_zs.append(z)
            predicted_rs.append(r_pred)
            predicted_ds.append(d_pred)

        predicted_zs = torch.stack(predicted_zs, dim=1)  # (b, t-1, latent_dim)
        predicted_rs = torch.stack(predicted_rs, dim=1)  # (b, t-1, 1)
        predicted_ds = torch.stack(predicted_ds, dim=1)  # (b, t-1, 1)

        # Discount weights: [γ^0, γ^1, ..., γ^(t-2)] — earlier steps weighted more
        discount = params.rollout_discount
        weights = torch.tensor(
            [discount ** h for h in range(t - 1)],
            dtype=predicted_zs.dtype,
            device=predicted_zs.device,
        )  # (t-1,)

        # Cosine similarity loss between predicted and EMA target latents
        cosine_sim = F.cosine_similarity(predicted_zs, target_z, dim=-1)  # (b, t-1)
        consistency_loss = ((1 - cosine_sim) * weights).mean()

        # Reward MSE loss
        reward_loss = (F.mse_loss(predicted_rs, r[:, 1:], reduction='none').squeeze(-1) * weights).mean()

        # Done BCE loss
        done_loss = (
            F.binary_cross_entropy(predicted_ds, d[:, 1:].float(), reduction='none').squeeze(-1) * weights
        ).mean()

        loss = (
            params.consistency_coeff * consistency_loss
            + params.reward_coeff * reward_loss
            + params.done_coeff * done_loss
        )

        grad_norm = self.optim.backward(loss)
        if no_update:
            self.optim.zero_grad()
        else:
            self.optim.update_parameters()

        self._update_ema_encoder()

        losses = LatentWorldModelLosses(
            consistency_loss=consistency_loss.item(),
            reward_loss=reward_loss.item(),
            done_loss=done_loss.item(),
            grad_norm=grad_norm.item(),
        )

        if return_init_states:
            with torch.no_grad():
                z_all = self.encoder(o)  # (b, t, latent_dim)
            return losses, (z_all.detach(), a.detach(), r.detach(), d.detach())
        return losses

    def imagine_rollout(
            self,
            z: torch.Tensor,
            actor: Actor,
            num_timesteps: int = 25,
            disable_gradients: bool = False,
        ):
        """
        z: (b, latent_dim) - starting latent states

        Returns z, a, r, d each of shape (b, num_timesteps+1, dim).
        The TD3 trainer's [:, :-1] / [:, 1:] slicing then gives
        num_timesteps valid (current, next) transition pairs.
        Rewards and dones are zero-padded at the final timestep.
        """
        with torch.set_grad_enabled(not disable_gradients):
            with FreezeParameters([self.dynamic_model, self.encoder]):
                states = [z]
                actions = []
                rewards = []
                dones = []

                z_current = z
                for _ in range(num_timesteps):
                    a = actor(z_current.detach(), deterministic=True)
                    if self.environment_action_bound is not None:
                        a = torch.clamp(
                            a,
                            -self.environment_action_bound,
                            self.environment_action_bound,
                        )
                    z_next, r, d = self._step(z_current, a)
                    actions.append(a)
                    rewards.append(r)
                    dones.append(d)
                    states.append(z_next)
                    z_current = z_next

                # Final action at z_T (used by actor update, not critic)
                a_final = actor(z_current.detach(), deterministic=True)
                if self.environment_action_bound is not None:
                    a_final = torch.clamp(
                        a_final,
                        -self.environment_action_bound,
                        self.environment_action_bound,
                    )
                actions.append(a_final)

                # Pad r and d at the end — final state has no outgoing transition
                rewards.append(torch.zeros_like(rewards[-1]))
                dones.append(torch.zeros_like(dones[-1]))

                return (
                    torch.stack(states, dim=1),   # (b, T+1, latent_dim)
                    torch.stack(actions, dim=1),  # (b, T+1, action_dim)
                    torch.stack(rewards, dim=1),  # (b, T+1, 1)
                    torch.stack(dones, dim=1),    # (b, T+1, 1)
                )
