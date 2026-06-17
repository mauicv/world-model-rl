from typing import Union
import copy
import torch

from reflect.components.models.actor import Actor
from reflect.components.models import ConvEncoder
from reflect.components.rssm_world_model.models import DenseModel
from reflect.components.transformer_world_model.world_model import WorldModel
from reflect.utils import FreezeParameters
from reflect.utils import create_z_dist

class WorldModelActor(torch.nn.Module):
    def __init__(
            self,
            actor: Actor,
        ):
        super().__init__()
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

        with FreezeParameters([self.actor, self.actor_copy]):
            return self.actor_copy(obs, deterministic=True)