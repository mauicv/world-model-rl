from typing import Union
import torch

from reflect.components.models.actor import Actor
from reflect.components.models import ConvEncoder
from reflect.components.rssm_world_model.models import DenseModel
from reflect.components.transformer_world_model.world_model import WorldModel
from reflect.utils import FreezeParameters


class EncoderActor(torch.nn.Module):
    def __init__(
            self,
            encoder: Union[DenseModel, ConvEncoder],
            actor: Actor
        ):
        super().__init__()
        self.encoder = encoder
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

        with FreezeParameters([self.encoder, self.actor]):    
            z, _ = self.encoder(obs)
            z = z.reshape(1, 1, -1)
            return self.actor(z, deterministic=True)


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
            