from typing import Optional
from dataclasses import dataclass
from reflect.components.models.encoder import ConvEncoder
from reflect.components.models.decoder import ConvDecoder
from reflect.components.transformer_world_model.transformer import PytfexTransformer
from reflect.components.models.actor import Actor
from itertools import chain
from reflect.components.base import Base
from reflect.utils import FreezeParameters

import torch
from reflect.utils import (
    recon_loss_fn,
    reg_loss_fn,
    # cross_entropy_loss_fn,
    reward_loss_fn,
    AdamOptim,
    create_z_dist,
    kl_divergence_loss_fn
)

done_loss_fn = torch.nn.BCELoss()


@dataclass
class WorldModelTrainingParams:
    reg_coeff: float = 0.0
    recon_coeff: float = 1.0
    dynamic_coeff: float = 1.0
    reward_coeff: float = 10.0
    done_coeff: float = 1.0


@dataclass
class WorldModelLosses:
    recon_loss: float
    reg_loss: float
    dynamic_loss: float
    reward_loss: float
    done_loss: float
    dynamic_grad_norm: float
    observation_grad_norm: float

    recon_loss_per_timestep: Optional[torch.Tensor] = None
    dynamic_loss_per_timestep: Optional[torch.Tensor] = None
    reward_loss_per_timestep: Optional[torch.Tensor] = None


class WorldModel(Base):
    model_list = [
        'encoder',
        'decoder',
        'dynamic_model',
        'observation_model_opt',
        'dynamic_model_opt'
    ]
    def __init__(
            self,
            encoder: ConvEncoder,
            decoder: ConvDecoder,
            dynamic_model: PytfexTransformer,
            num_cat: int=32,
            num_latent: int=32,
            params: Optional[WorldModelTrainingParams] = None,
        ):
        super().__init__()
        if params is None:
            params = WorldModelTrainingParams()
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.dynamic_model = dynamic_model
        self.num_cat = num_cat
        self.num_latent = num_latent
        observation_parameters = chain(encoder.parameters(), decoder.parameters())
        self.observation_model_opt = AdamOptim(
            observation_parameters,
            lr=0.0001,
            eps=1e-5,
            grad_clip=100
        )
        self.dynamic_model_opt = AdamOptim(
            self.dynamic_model.parameters(),
            lr=0.0001,
            eps=1e-5,
            grad_clip=100
        )

    def encode(self, image):
        image = image.reshape(
            image.shape[0] * image.shape[1],
            *image.shape[2:]
        )
        z = self.encoder(image)
        z_logits = z.reshape(-1, self.num_latent, self.num_cat)
        z_dist = create_z_dist(z_logits)
        z_sample = z_dist.rsample()
        return z_sample, z_logits

    def decode(self, z_sample, h):
        decoder_input = torch.cat([z_sample, h], dim=-1)
        return self.decoder(decoder_input)

    def update(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            training_mask: Optional[torch.Tensor] = None,
            params: Optional[WorldModelTrainingParams] = None,
            return_init_states: bool=False,
            no_update: bool=False,
        ):
        if params is None:
            params = self.params
        if training_mask is None:
            training_mask = torch.ones(o.shape[:2]).to(o.device)
        b, t, *_  = o.shape

        z, z_logits = self.encode(o)
        z = z.reshape(b, t, -1)
        reg_loss = reg_loss_fn(z_logits)

        # Dynamic Models
        _, num_z, num_c = z_logits.shape
        z_logits = z_logits.reshape(b, t, num_z, num_c)

        r_targets = r[:, 1:].detach()
        d_targets = d[:, 1:].detach()
        z_logits = z_logits[:, 1:]
        next_z_dist = create_z_dist(z_logits.detach())
        
        z_inputs, r_inputs, a_inputs, training_mask_inputs = (
            z[:, :-1].detach(),
            r[:, :-1].detach(),
            a[:, :-1].detach(),
            training_mask[:, :-1].detach()
        )
        # z_inputs = z_inputs * training_mask_inputs[:, :, None]
        r_inputs = r_inputs * training_mask_inputs[:, :, None]
        h, z_pred, r_pred, d_pred = self.dynamic_model(
            z_inputs, a_inputs, r_inputs,
        )

        training_mask_targets = training_mask[:, 1:].detach()
        r_pred = r_pred * training_mask_targets[:, :, None]
        d_pred = d_pred * training_mask_targets[:, :, None]
        dynamic_loss, dynamic_loss_per_timestep = kl_divergence_loss_fn(z_pred, next_z_dist)

        reward_loss, reward_loss_per_timestep = reward_loss_fn(r_targets, r_pred)
        done_loss = done_loss_fn(d_pred, d_targets.float())

        # Update observation_model and dynamic_model
        dyn_loss = (
            params.dynamic_coeff * dynamic_loss 
            + params.reward_coeff * reward_loss
            + params.done_coeff * done_loss
        )

        # Observational Model
        r_o = self.decode(z[:, 1:], h)
        t_o = o[:, 1:].reshape(-1, *o.shape[2:])
        r_o = r_o.reshape(-1, *r_o.shape[2:])

        recon_loss, recon_loss_per_timestep = recon_loss_fn(t_o, r_o)

        obs_loss = (
            params.recon_coeff * recon_loss 
            + params.reg_coeff * reg_loss
        )

        dynamic_grad_norm = self.dynamic_model_opt.backward(dyn_loss, retain_graph=True)
        observation_grad_norm = self.observation_model_opt.backward(obs_loss, retain_graph=False)
        
        if no_update:
            self.dynamic_model_opt.zero_grad()
            self.observation_model_opt.zero_grad()
        else:
            self.dynamic_model_opt.update_parameters()
            self.observation_model_opt.update_parameters()

        losses = WorldModelLosses(
            recon_loss=recon_loss.detach().cpu().item(),
            reg_loss= reg_loss.detach().cpu().item(),
            dynamic_loss= dynamic_loss.detach().cpu().item(),
            reward_loss= reward_loss.detach().cpu().item(),
            done_loss= done_loss.detach().cpu().item(),
            dynamic_grad_norm=dynamic_grad_norm.cpu().item(),
            observation_grad_norm=observation_grad_norm.cpu().item(),
            recon_loss_per_timestep=recon_loss_per_timestep.reshape(b, t-1).detach().cpu(),
            dynamic_loss_per_timestep=dynamic_loss_per_timestep.reshape(b, t-1).detach().cpu(),
            reward_loss_per_timestep=reward_loss_per_timestep.reshape(b, t-1).detach().cpu(),
        )
        if return_init_states:
            return losses, self.flatten_batch_time(z=z, a=a, r=r, d=d)
        return losses

    def flatten_batch_time(self, z, a, r, d):
        b, t, *_ = z.shape
        z = z.detach().reshape(b * t, 1, -1)
        a = a.detach().reshape(b * t, 1, -1)
        r = r.detach().reshape(b * t, 1, -1)
        d = d.detach().reshape(b * t, 1, -1)
        return z, a, r, d

    def imagine_rollout(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            actor: Actor,
            num_timesteps: int=25,
            with_observations: bool=False,
            with_entropies: bool=False
        ):
        
        with FreezeParameters([self.dynamic_model, self.decoder]):
            if with_entropies:
                entropies = []  # Use a list to collect entropies
                # Calculate entropy for initial state
                action_dist = actor(z[:, -1, :].detach(), deterministic=False)
                entropies.append(action_dist.entropy()[:, None])

            h = None
            for i in range(num_timesteps):
                h, new_z, new_r, new_d = self \
                    .dynamic_model.rstep(z=z, a=a, r=r, d=d, h=h)
                if with_entropies:
                    action_dist = actor(
                        new_z[:, -1, :].detach(),
                        deterministic=False
                    )
                    action = action_dist.rsample()
                    entropies.append(action_dist.entropy()[:, None])
                else:
                    action = actor(
                        new_z[:, -1, :].detach(),
                        deterministic=True
                    )
                new_a = torch.cat((a, action[:, None, :]), dim=1)
                z, a, r, d = new_z, new_a, new_r, new_d
            to_return = [z, a, r, d]
            if with_entropies:
                # Stack the entropies along time dimension
                entropy = torch.stack(entropies, dim=1)  # [batch, time, 1]
                to_return.append(entropy)
            if with_observations:
                b, t, *_ = z.shape
                _z = z[:, 1:].reshape(b*(t-1), -1)
                _h = h.reshape(b*(t-1), -1)
                o = self.decode(_z, _h)
                o = o.reshape(b, t - 1, *o.shape[1:])
                to_return.append(o)
            return to_return