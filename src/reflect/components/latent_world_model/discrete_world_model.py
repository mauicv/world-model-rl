import copy
from typing import Optional, List
from dataclasses import dataclass
from reflect.components.models.actor import Actor
from itertools import chain
from reflect.components.base import Base
from reflect.components.latent_world_model.fsq import FSQ
from reflect.utils import FreezeParameters, AdamOptim

import torch
import torch.nn.functional as F



@dataclass
class DiscreteLatentWorldModelLosses:
    consistency_loss: float
    reward_loss: float
    done_loss: float
    grad_norm: float
    effective_rank: float
    mean_latent_std: float


class DiscreteLatentWorldModel(Base):
    model_list = [
        'encoder',
        'dynamic_model',
        'optim'
    ]

    def __init__(
            self,
            encoder: torch.nn.Module,
            dynamic_model: torch.nn.Module,
            latent_dim: int,
            ema_tau: float = 0.005,
            environment_action_bound: float = 1.0,
            learning_rate: float = 3e-4,
            consistency_coeff: float = 1.0,
            reward_coeff: float = 1.0,
            done_coeff: float = 1.0,
            rollout_discount: float = 0.5,
            fsq_levels: List[int] = [8, 8],
        ):
        super().__init__()
        self.fsq = FSQ(fsq_levels)
        self.num_groups = latent_dim // self.fsq.num_channels
        self.codebook_size = self.fsq.codebook_size
        self.consistency_coeff = consistency_coeff
        self.reward_coeff = reward_coeff
        self.done_coeff = done_coeff
        self.rollout_discount = rollout_discount
        self.encoder = encoder
        self.ema_encoder = copy.deepcopy(encoder)
        self.ema_encoder.requires_grad_(False)
        self.ema_tau = ema_tau
        self.dynamic_model = dynamic_model
        self.environment_action_bound = environment_action_bound
        optim_param = chain(encoder.parameters(), self.dynamic_model.parameters())
        self.optim = AdamOptim(
            optim_param,
            lr=learning_rate,
            # eps=1e-5,
            grad_clip=10,
            weight_decay=1e-6
        )

    def _latent_collapse_metrics(self, z: torch.Tensor):
        # z: (b, t, latent_dim) or (b, latent_dim)
        z_flat = z.reshape(-1, z.shape[-1]).detach()
        mean_latent_std = z_flat.std(dim=0).mean()
        z_centered = z_flat - z_flat.mean(dim=0)
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        p = S / (S.sum() + 1e-8)
        effective_rank = torch.exp(-(p * torch.log(p + 1e-8)).sum())
        return effective_rank.item(), mean_latent_std.item()

    def _update_ema_encoder(self):
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_encoder.parameters(),
                self.encoder.parameters()
            ):
                ema_param.data = self.ema_tau * param.data + (1 - self.ema_tau) * ema_param.data

    def _step(self, z: torch.Tensor, a: torch.Tensor, tau: float = 1.0, hard: bool = True):
        """
        z: (b, latent_dim) - flat quantized codes
        a: (b, action_dim)

        Returns:
            logits: (b, num_groups, codebook_size)
            codes:  (b, latent_dim) - gumbel-sampled next step codes
            r:      (b, 1)
            d:      (b, 1) or None
        """
        z_raw, r, d = self.dynamic_model(z, a)
        logits = z_raw.view(*z_raw.shape[:-1], self.num_groups, self.codebook_size)
        onehot = F.gumbel_softmax(logits, tau=tau, hard=hard)  # (b, num_groups, codebook_size)
        codebook = self.fsq.implicit_codebook.to(z.device)     # (codebook_size, num_channels)
        codes = (onehot @ codebook).flatten(-2)                # (b, latent_dim)
        return logits, codes, r, d

    def encode(self, state):
        return self.encoder(state)

    def update(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            return_init_states: bool = False,
            no_update: bool = False,
        ):
        """
        o: (b, t, *s_dim)
        a: (b, t, a_dim)
        r: (b, t, 1)
        d: (b, t, 1)
        """
        b, t, *_ = o.shape

        # Encode first observation as rollout starting point
        enc = self.encoder(o[:, 0])
        assert torch.all(enc < 1) and torch.all(enc > -1), "Encoder output must be between -1 and 1"
        z = self.fsq(enc)['codes']  # (b, latent_dim)

        # EMA targets for all subsequent observations
        with torch.no_grad():
            s_dim = o.shape[2:]
            ema_enc = self.ema_encoder(o[:, 1:].reshape(b * (t - 1), *s_dim))
            target_indices = self.fsq(ema_enc)['indices'].view(b, t - 1, self.num_groups)  # (b, t-1, num_groups)

        # Roll forward t-1 steps using own predictions
        predicted_logits = []
        predicted_zs = []
        predicted_rs = []
        predicted_ds = []
        for h in range(t - 1):
            logits, z, r_pred, d_pred = self._step(z, a[:, h])
            predicted_logits.append(logits)
            predicted_zs.append(z)
            predicted_rs.append(r_pred)
            if d_pred is not None:
                predicted_ds.append(d_pred)

        predicted_logits = torch.stack(predicted_logits, dim=1)  # (b, t-1, num_groups, codebook_size)
        predicted_zs = torch.stack(predicted_zs, dim=1)          # (b, t-1, latent_dim)
        predicted_rs = torch.stack(predicted_rs, dim=1)          # (b, t-1, 1)
        predicted_ds = torch.stack(predicted_ds, dim=1) if predicted_ds else None  # (b, t-1, 1)

        # Discount weights: [γ^0, γ^1, ..., γ^(t-2)] — earlier steps weighted more
        discount = self.rollout_discount
        weights = torch.tensor(
            [discount ** h for h in range(t - 1)],
            dtype=predicted_logits.dtype,
            device=predicted_logits.device,
        )  # (t-1,)

        # CE loss: logits (b, t-1, num_groups, codebook_size) vs indices (b, t-1, num_groups)
        logits_flat = predicted_logits.reshape(b * (t - 1) * self.num_groups, self.codebook_size)
        targets_flat = target_indices.reshape(b * (t - 1) * self.num_groups).long()
        ce_per_group = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        ce_per_step = ce_per_group.view(b, t - 1, self.num_groups).mean(dim=-1)  # (b, t-1)
        consistency_loss = (ce_per_step * weights).sum(dim=1).mean()

        # Reward MSE loss — same structure: sum over time, mean over batch
        reward_loss = (F.mse_loss(predicted_rs, r[:, 1:], reduction='none').squeeze(-1) * weights).sum(dim=1).mean()

        done_loss = torch.tensor(0.0)
        if predicted_ds is not None:
            done_loss = (
                F.binary_cross_entropy(predicted_ds, d[:, 1:].float(), reduction='none')
                .squeeze(-1) * weights
            ).sum(dim=1).mean()

        loss = (
            self.consistency_coeff * consistency_loss.clamp(max=1e4)
            + self.reward_coeff * reward_loss.clamp(max=1e4)
            + self.done_coeff * done_loss.clamp(max=1e4)
        )
        # Scale gradients by 1/horizon — matches TCRL's register_hook convention
        horizon = t - 1
        loss.register_hook(lambda grad: grad * (1 / horizon))

        grad_norm = self.optim.backward(loss)
        if no_update:
            self.optim.zero_grad()
        else:
            self.optim.update_parameters()

        self._update_ema_encoder()

        effective_rank, mean_latent_std = self._latent_collapse_metrics(predicted_zs)
        losses = DiscreteLatentWorldModelLosses(
            consistency_loss=consistency_loss.item(),
            reward_loss=reward_loss.item(),
            done_loss=done_loss.item(),
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
                    _, z_next, r, d = self._step(z_current, a)
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
