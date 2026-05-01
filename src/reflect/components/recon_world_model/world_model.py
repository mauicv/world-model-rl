from dataclasses import dataclass
from itertools import chain
from typing import Optional

import torch
import torch.nn.functional as F

from reflect.components.base import Base
from reflect.components.models.actor import Actor
from reflect.utils import AdamOptim, FreezeParameters


@dataclass
class ReconWorldModelTrainingParams:
    consistency_coeff: float = 1.0
    reward_coeff: float = 1.0
    done_coeff: float = 1.0
    recon_coeff: float = 1.0
    recon_threshold: float = 0.5
    rollout_discount: float = 0.5


@dataclass
class ReconWorldModelLosses:
    consistency_loss: float
    reward_loss: float
    done_loss: float
    recon_loss: float
    recon_gate_mean: float
    grad_norm: float
    effective_rank: float
    mean_latent_std: float


class ReconWorldModel(Base):
    model_list = [
        'encoder',
        'decoder',
        'dynamic_model',
        'optim',
    ]

    def __init__(
            self,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            dynamic_model: torch.nn.Module,
            params: Optional[ReconWorldModelTrainingParams] = None,
            environment_action_bound: float = 1.0,
            use_delta: bool = False,
            learning_rate: float = 3e-4,
        ):
        super().__init__()
        if params is None:
            params = ReconWorldModelTrainingParams()
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.dynamic_model = dynamic_model
        self.environment_action_bound = environment_action_bound
        self.use_delta = use_delta
        self.optim = AdamOptim(
            chain(encoder.parameters(), decoder.parameters(), dynamic_model.parameters()),
            lr=learning_rate,
            grad_clip=10,
            weight_decay=1e-6,
        )

    def _latent_collapse_metrics(self, z: torch.Tensor):
        z_flat = z.reshape(-1, z.shape[-1]).detach()
        mean_latent_std = z_flat.std(dim=0).mean()
        z_centered = z_flat - z_flat.mean(dim=0)
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        p = S / (S.sum() + 1e-8)
        effective_rank = torch.exp(-(p * torch.log(p + 1e-8)).sum())
        return effective_rank.item(), mean_latent_std.item()

    def _step(self, z: torch.Tensor, a: torch.Tensor):
        z_out, r, d = self.dynamic_model(z, a)
        if self.use_delta:
            z_out = z + z_out
        return z_out, r, d

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def update(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            params: Optional[ReconWorldModelTrainingParams] = None,
            return_init_states: bool = False,
            no_update: bool = False,
        ):
        """
        o: (b, t, obs_dim)
        a: (b, t, a_dim)
        r: (b, t, 1)
        d: (b, t, 1)
        """
        if params is None:
            params = self.params

        b, t, *_ = o.shape

        # Encode rollout start
        z = self.encoder(o[:, 0])                  # (b, latent_dim)

        # Single encoder pass for o[:, 1:].
        # z_for_recon keeps grad for the reconstruction path.
        # target_z is detached — stop-grad consistency target, no EMA.
        z_for_recon = self.encoder(o[:, 1:])        # (b, t-1, latent_dim)
        target_z = z_for_recon.detach()             # (b, t-1, latent_dim)

        # Roll forward t-1 steps
        predicted_zs, predicted_rs, predicted_ds = [], [], []
        for h in range(t - 1):
            z, r_pred, d_pred = self._step(z, a[:, h])
            predicted_zs.append(z)
            predicted_rs.append(r_pred)
            if d_pred is not None:
                predicted_ds.append(d_pred)

        predicted_zs = torch.stack(predicted_zs, dim=1)                              # (b, t-1, latent_dim)
        predicted_rs = torch.stack(predicted_rs, dim=1)                              # (b, t-1, 1)
        predicted_ds = torch.stack(predicted_ds, dim=1) if predicted_ds else None    # (b, t-1, 1)

        weights = torch.tensor(
            [params.rollout_discount ** h for h in range(t - 1)],
            dtype=predicted_zs.dtype,
            device=predicted_zs.device,
        )  # (t-1,)

        # Consistency loss
        pred_norm = F.normalize(predicted_zs, dim=-1, p=2)
        target_norm = F.normalize(target_z, dim=-1, p=2)
        cosine_dist = 2 - 2 * (pred_norm * target_norm).sum(dim=-1)  # (b, t-1)
        consistency_loss = (cosine_dist * weights).sum(dim=1).mean()

        # Reward loss
        reward_loss = (
            F.mse_loss(predicted_rs, r[:, 1:], reduction='none').squeeze(-1) * weights
        ).sum(dim=1).mean()

        # Done loss
        done_loss = torch.tensor(0.0, device=o.device)
        if predicted_ds is not None:
            done_loss = (
                F.binary_cross_entropy(predicted_ds, d[:, 1:].float(), reduction='none')
                .squeeze(-1) * weights
            ).sum(dim=1).mean()

        # Soft gate: 1 when cosine_dist=0 (perfect prediction), 0 at threshold
        recon_gate = F.relu(params.recon_threshold - cosine_dist) / params.recon_threshold  # (b, t-1)

        # Gated reconstruction loss — gradient flows through z_for_recon into encoder
        recon_obs = self.decoder(z_for_recon)                                        # (b, t-1, obs_dim)
        recon_loss_per = F.mse_loss(recon_obs, o[:, 1:], reduction='none').mean(dim=-1)  # (b, t-1)
        recon_loss = (recon_gate * recon_loss_per * weights).sum(dim=1).mean()

        loss = (
            params.consistency_coeff * consistency_loss.clamp(max=1e4)
            + params.reward_coeff    * reward_loss.clamp(max=1e4)
            + params.done_coeff      * done_loss.clamp(max=1e4)
            + params.recon_coeff     * recon_loss.clamp(max=1e4)
        )
        horizon = t - 1
        loss.register_hook(lambda grad: grad * (1 / horizon))

        grad_norm = self.optim.backward(loss)
        if no_update:
            self.optim.zero_grad()
        else:
            self.optim.update_parameters()

        effective_rank, mean_latent_std = self._latent_collapse_metrics(predicted_zs)
        losses = ReconWorldModelLosses(
            consistency_loss=consistency_loss.item(),
            reward_loss=reward_loss.item(),
            done_loss=done_loss.item(),
            recon_loss=recon_loss.item(),
            recon_gate_mean=recon_gate.mean().item(),
            grad_norm=grad_norm.item(),
            effective_rank=effective_rank,
            mean_latent_std=mean_latent_std,
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
        z: (b, latent_dim) — starting latent states

        Returns z, a, r, d each of shape (b, num_timesteps+1, dim).
        """
        with torch.set_grad_enabled(not disable_gradients):
            with FreezeParameters([self.dynamic_model, self.encoder]):
                states = [z]
                actions, rewards, dones = [], [], []

                z_current = z
                for _ in range(num_timesteps):
                    a_t = actor(z_current.detach(), deterministic=True)
                    if self.environment_action_bound is not None:
                        a_t = torch.clamp(a_t, -self.environment_action_bound, self.environment_action_bound)
                    z_next, r_t, d_t = self._step(z_current, a_t)
                    actions.append(a_t)
                    rewards.append(r_t)
                    dones.append(d_t)
                    states.append(z_next)
                    z_current = z_next

                a_final = actor(z_current.detach(), deterministic=True)
                if self.environment_action_bound is not None:
                    a_final = torch.clamp(a_final, -self.environment_action_bound, self.environment_action_bound)
                actions.append(a_final)
                rewards.append(torch.zeros_like(rewards[-1]))
                dones.append(torch.zeros_like(dones[-1]))

                return (
                    torch.stack(states, dim=1),
                    torch.stack(actions, dim=1),
                    torch.stack(rewards, dim=1),
                    torch.stack(dones, dim=1),
                )
