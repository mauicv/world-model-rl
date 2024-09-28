from typing import Optional
from dataclasses import dataclass
from reflect.components.observation_model import ObservationalModel
from reflect.components.observation_model.encoder import ConvEncoder
from reflect.components.observation_model.decoder import ConvDecoder
from itertools import chain
from reflect.components.base import Base

from pytfex.transformer.gpt import GPT
import torch
from reflect.utils import (
    recon_loss_fn,
    reg_loss_fn,
    cross_entropy_loss_fn,
    reward_loss_fn,
    AdamOptim,
    create_z_dist
)
import torch.distributions as D

done_loss_fn = torch.nn.BCELoss()


def get_causal_mask(l):
    mask = torch.tril(torch.ones(l, l))
    masked_indices = mask[None, None, :l, :l] == 0
    return masked_indices


@dataclass
class WorldModelTrainingParams:
    reg_coeff: float = 0.0
    recon_coeff: float = 1.0
    dynamic_coeff: float = 1.0
    consistency_coeff: float = 0.0
    reward_coeff: float = 10.0
    done_coeff: float = 1.0


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
            # observation_model: ObservationalModel,
            encoder: ConvEncoder,
            decoder: ConvDecoder,
            dynamic_model: GPT,
            num_ts: int,
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
        self.num_ts = num_ts
        self.num_cat = num_cat
        self.num_latent = num_latent
        self.mask = get_causal_mask(self.num_ts * 3)
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


    def _step(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
        ):
        z_dist, new_r, new_d = self.dynamic_model((
            z[:, -self.num_ts:],
            a[:, -self.num_ts:],
            r[:, -self.num_ts:]
        ))

        new_r = new_r[:, -1].reshape(-1, 1, 1)
        r = torch.cat([r, new_r], dim=1)

        new_d = new_d[:, -1].reshape(-1, 1, 1)
        d = torch.cat([d, new_d], dim=1)

        return z_dist, r, d

    def step(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
        ):
        z_dist, new_r, new_d = self._step(z, a, r, d)
        new_z = z_dist.sample()
        new_z = new_z[:, -1].reshape(-1, 1, self.num_cat * self.num_latent)
        new_z = torch.cat([z, new_z], dim=1)
        return new_z, new_r, new_d

    def rstep(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
        ):
        z_dist, new_r, new_d = self._step(z, a, r, d)
        new_z = z_dist.rsample()
        new_z = new_z[:, -1].reshape(-1, 1, self.num_cat * self.num_latent)
        new_z = torch.cat([z, new_z], dim=1)
        return new_z, new_r, new_d

    def encode(self, image):
        b, t, c, h, w = image.shape
        image = image.reshape(b * t, c, h, w)
        z = self.encoder(image)
        z_logits = z.reshape(-1, self.num_latent, self.num_cat)
        z_dist = create_z_dist(z_logits)
        z_sample = z_dist.rsample()
        return z_sample, z_logits

    def decode(self, z_sample):
        return self.decoder(z_sample)

    def update_observation_model(
            self,
            o: torch.Tensor,
            params: Optional[WorldModelTrainingParams] = None,
        ):
        if params is None:
            params = self.params

        self.mask = self.mask.to(o.device)
        z_sample, z_logits = self.encode(o)
        r_o = self.decode(z_sample)

        recon_loss = recon_loss_fn(o, r_o)
        reg_loss = reg_loss_fn(z_logits)
        loss = params.recon_coeff * recon_loss + params.reg_coeff * reg_loss
        self.observation_model_opt.backward(loss, retain_graph=False)
        self.observation_model_opt.update_parameters()

        return {
            'recon_loss': recon_loss.detach().cpu().item(),
            'reg_loss': reg_loss.detach().cpu().item(),
        }

    def update(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
            params: Optional[WorldModelTrainingParams] = None,
        ):
        if params is None:
            params = self.params
        b, t, *_  = o.shape

        self.mask = self.mask.to(o.device)
        z, z_logits = self.encode(o)
        r_o = self.decode(z)
        t_o = o.reshape(-1, *o.shape[2:])

        # Observational Model
        recon_loss = recon_loss_fn(t_o, r_o)
        reg_loss = reg_loss_fn(z_logits)

        # Dynamic Models
        _, num_z, num_c = z_logits.shape
        z = z.detach().reshape(b, t, -1)
        _, num_z, num_c = z_logits.shape
        z_logits = z_logits.reshape(b, t, num_z, num_c)
        r_targets = r[:, 1:].detach()
        d_targets = d[:, 1:].detach()
        z_logits = z_logits[:, 1:]
        next_z_dist = create_z_dist(z_logits.detach())
        c_z_dist = create_z_dist(z_logits)
        
        z_inputs, r_inputs, a_inputs = (
            z[:, :-1].detach(),
            r[:, :-1].detach(),
            a[:, :-1].detach()
        )

        z_pred, r_pred, d_pred = self.dynamic_model(
            (z_inputs, a_inputs, r_inputs),
            mask=self.mask
        )
        dynamic_loss = cross_entropy_loss_fn(z_pred, next_z_dist)
        reward_loss = reward_loss_fn(r_targets, r_pred)
        done_loss = done_loss_fn(d_pred, d_targets.float())

        # Update observation_model and dynamic_model
        dyn_loss = (
            params.dynamic_coeff * dynamic_loss 
            + params.reward_coeff * reward_loss
            + params.done_coeff * done_loss
        )
        consistency_loss = cross_entropy_loss_fn(c_z_dist, z_pred)
        # TODO: Consistency loss seems to penalize both the observation 
        # model and the dynamic model? Why?
        obs_loss = (
            params.recon_coeff * recon_loss 
            + params.reg_coeff * reg_loss 
            + params.consistency_coeff * consistency_loss
        )

        self.dynamic_model_opt.backward(dyn_loss, retain_graph=False)
        self.observation_model_opt.backward(obs_loss, retain_graph=False)
        self.dynamic_model_opt.update_parameters()
        self.observation_model_opt.update_parameters()

        return {
            'recon_loss': recon_loss.detach().cpu().item(),
            'reg_loss': reg_loss.detach().cpu().item(),
            'consistency_loss': consistency_loss.detach().cpu().item(),
            'dynamic_loss': dynamic_loss.detach().cpu().item(),
            'reward_loss': reward_loss.detach().cpu().item(),
            'done_loss': done_loss.detach().cpu().item(),
        }
