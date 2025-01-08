from typing import Union
import copy
import torch

from reflect.components.models.actor import Actor
from reflect.components.models import ConvEncoder
from reflect.components.rssm_world_model.models import DenseModel
from reflect.components.transformer_world_model.world_model import WorldModel
from reflect.utils import FreezeParameters
from reflect.utils import create_z_dist

class EncoderActor(torch.nn.Module):
    def __init__(
            self,
            encoder: Union[DenseModel, ConvEncoder],
            actor: Actor,
            num_latent: int = 256,
            num_cat: int = 32,
        ):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.actor_copy = copy.deepcopy(actor)
        self.num_latent = num_latent
        self.num_cat = num_cat

    def perturb_actor(self, weight_perturbation_size: float = 0.01):
        self.reset_to_original_actor()
        for param_name, param in self.actor_copy.named_parameters():
            if "weight" in param_name:
                param.data = param.data + torch.randn_like(param.data) * weight_perturbation_size

    def reset(self):
        pass

    def reset_to_original_actor(self):
        self.actor_copy = copy.deepcopy(self.actor)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        s_{t} = encoder(o_{t})
        a_{t} = actor(s_{t})
        """
        device = next(self.actor.parameters()).device
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        obs = obs.to(device)

        with FreezeParameters([self.encoder, self.actor, self.actor_copy]):
            z = self.encoder(obs)
            z_logits = z.reshape(-1, self.num_latent, self.num_cat)
            z_dist = create_z_dist(z_logits)
            z_sample = z_dist.rsample()
            z_sample = z_sample.reshape(1, 1, -1)
            return self.actor_copy(z_sample, deterministic=True)


class TransformerWorldModelActor:
    def __init__(
            self,
            world_model: WorldModel,
            actor: Actor
        ):
        self.world_model = world_model
        self.actor = actor
        self.actor_copy = copy.deepcopy(actor)

    def perturb_actor(self, weight_perturbation_size: float = 0.01):
        self.reset_to_original_actor()
        for param_name, param in self.actor_copy.named_parameters():
            if "weight" in param_name:
                param.data = param.data + torch.randn_like(param.data) * weight_perturbation_size

    def reset(self):
        pass

    def reset_to_original_actor(self):
        self.actor_copy = copy.deepcopy(self.actor)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        s_{t} = encoder(o_{t})
        a_{t} = actor(s_{t})
        """
        device = next(self.actor.parameters()).device
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        obs = obs.to(device)

        with FreezeParameters([self.world_model, self.actor, self.actor_copy]):
            z, _ = self.world_model.encode(obs)
            z = z.reshape(1, 1, -1)
            return self.actor_copy(z, deterministic=True)
            