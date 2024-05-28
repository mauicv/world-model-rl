from typing import Optional
from dataclasses import dataclass

from reflect.models.world_model.observation_model import ObservationalModel
from pytfex.transformer.gpt import GPT
import torch
from reflect.models.td3_policy import EPS
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


class WorldModel(torch.nn.Module):
    def __init__(
            self,
            observation_model: ObservationalModel,
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
        self.observation_model = observation_model
        self.dynamic_model = dynamic_model
        self.num_ts = num_ts
        self.num_cat = num_cat
        self.num_latent = num_latent
        self.mask = get_causal_mask(self.num_ts)
        self.observation_model_opt = AdamOptim(
            self.observation_model.parameters(),
            lr=0.0001,
            eps=1e-5,
            weight_decay=1e-6,
            grad_clip=100
        )
        self.dynamic_model_opt = AdamOptim(
            self.dynamic_model.parameters(),
            lr=0.0001,
            eps=1e-5,
            weight_decay=1e-6,
            grad_clip=100
        )

    def forward(self, x):
        z_dist = self.obs_model.encode(x)
        z = z_dist.rsample()
        y = self.observation_model.decoder(z)
        y_hat = self.dynamic_model(y)
        return y_hat, z, z_dist

    def encode(self, image):
        b, t, c, h, w = image.shape
        image = image.reshape(b * t, c, h, w)
        z = self.observation_model.encode(image)
        return z.reshape(b, t, -1)

    def decode(self, z):
        b, t, _ = z.shape
        z = z.reshape(b * t, -1)
        image = self.observation_model.decode(z)
        return image.reshape(b, t, *image.shape[1:])

    def step(
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
        new_z = z_dist.sample()[:, -1].reshape(-1, 1, self.num_cat * self.num_latent)
        z = torch.cat([z, new_z], dim=1)

        new_r = new_r[:, -1].reshape(-1, 1, 1)
        r = torch.cat([r, new_r], dim=1)

        new_d = new_d[:, -1].reshape(-1, 1, 1)
        d = torch.cat([d, new_d], dim=1)

        return z, r, d

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

        self.mask.to(o.device)
        b, t, c, h, w  = o.shape
        o = o.reshape(b * t, c, h, w)

        # Observational Model
        r_o, z, z_logits = self.observation_model(o)
        recon_loss = recon_loss_fn(o, r_o)
        reg_loss = reg_loss_fn(z_logits)

        # Dynamic Models
        z = z.detach()
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
        consistency_loss = cross_entropy_loss_fn(c_z_dist, z_pred).detach()
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

    def load(
            self,
            path,
            name="world-model-checkpoint.pth",
            targets=None
        ):
        checkpoint = torch.load(f'{path}/{name}')
        if targets is None:
            targets = [
                'observation_model',
                'dynamic_model',
                'observation_model_opt',
                'dynamic_model_opt'
            ]
        
        for target in targets:
            print(f'Loading {target}...')
            getattr(self, target).load_state_dict(
                checkpoint[target]
            )


    def save(
            self,
            path,
            name="world-model-checkpoint.pth",
            targets=None
        ):
        if targets is None:
            targets = [
                'observation_model',
                'dynamic_model',
                'observation_model_opt',
                'dynamic_model_opt'
            ]
        
        checkpoint = {
            target: getattr(self, target).state_dict()
            for target in targets
        }
        torch.save(checkpoint, f'{path}/{name}')