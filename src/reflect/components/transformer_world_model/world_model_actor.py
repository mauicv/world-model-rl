from typing import Union
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
            num_cat: int = 32
        ):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.num_latent = num_latent
        self.num_cat = num_cat

    def reset(self):
        pass

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        s_{t} = encoder(o_{t})
        a_{t} = actor(s_{t})
        """
        device = next(self.actor.parameters()).device
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        obs = obs.to(device)

        with FreezeParameters([self.encoder, self.actor]):    
            z = self.encoder(obs)
            z_logits = z.reshape(-1, self.num_latent, self.num_cat)
            z_dist = create_z_dist(z_logits)
            z_sample = z_dist.rsample()
            z_sample = z_sample.reshape(1, 1, -1)
            return self.actor(z_sample, deterministic=True)


class TransformerWorldModelActor:
    def __init__(
            self,
            world_model: WorldModel,
            actor: Actor
        ):
        self.world_model = world_model
        self.actor = actor

    def reset(self):
        pass

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        s_{t} = encoder(o_{t})
        a_{t} = actor(s_{t})
        """
        device = next(self.actor.parameters()).device
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        obs = obs.to(device)

        with FreezeParameters([self.world_model, self.actor]):    
            z, _ = self.world_model.encode(obs)
            z = z.reshape(1, 1, -1)
            return self.actor(z, deterministic=True)
            