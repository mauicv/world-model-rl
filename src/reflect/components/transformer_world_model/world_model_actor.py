from typing import Union
from reflect.components.models.actor import Actor
from reflect.components.models.encoder import Encoder
from reflect.components.rssm_world_model.models import DenseModel
from reflect.utils import FreezeParameters
import torch


class EncoderActor(torch.nn.Module):
    def __init__(
            self,
            encoder: Union[DenseModel, Encoder],
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
            