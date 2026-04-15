from typing import Union
import copy
import torch

from reflect.components.latent_world_model.models.actor import MLPActor
from reflect.components.latent_world_model.models.encoder import MLPEncoder
from reflect.utils import FreezeParameters


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


class EncoderActor(torch.nn.Module):
    def __init__(
            self,
            latent_dim: int,
            encoder: MLPEncoder,
            actor: MLPActor,
        ):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.actor_copy = copy.deepcopy(actor)
        self.actor_copy.requires_grad_(False)
        self.latent_dim = latent_dim
        self._sync_actor_copy_device()

    def _sync_actor_copy_device(self):
        actor_device = _module_device(self.actor)
        actor_copy_device = _module_device(self.actor_copy)
        if actor_copy_device != actor_device:
            self.actor_copy = self.actor_copy.to(actor_device)

    def perturb_actor(self, weight_perturbation_size: float = 0.01):
        self.reset_to_original_actor()
        for param_name, param in self.actor_copy.named_parameters():
            if "weight" in param_name:
                param.data = param.data + torch.randn_like(param.data) * weight_perturbation_size

    def reset(self):
        pass

    def reset_to_original_actor(self):
        self.actor_copy = copy.deepcopy(self.actor)
        self._sync_actor_copy_device()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        s_{t} = encoder(o_{t})
        a_{t} = actor(s_{t})
        """
        encoder_device = _module_device(self.encoder)
        actor_device = _module_device(self.actor)
        if obs.dim() == 4 or obs.dim() == 2:
            obs = obs.unsqueeze(1)
        obs = obs.to(encoder_device)
        self._sync_actor_copy_device()

        with FreezeParameters([self.encoder, self.actor, self.actor_copy]):
            z = self.encoder(obs)
            return self.actor_copy(z, deterministic=True)
