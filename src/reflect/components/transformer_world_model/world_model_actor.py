from typing import Union
import copy
import torch

from reflect.components.models.actor import Actor
from reflect.components.models import ConvEncoder
from reflect.components.rssm_world_model.models import DenseModel
from reflect.components.transformer_world_model.world_model import WorldModel
from reflect.utils import FreezeParameters
from reflect.utils import create_z_dist


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


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
            z_logits = z.reshape(-1, self.num_latent, self.num_cat)
            z_dist = create_z_dist(z_logits)
            z_sample = z_dist.rsample()
            z_sample = z_sample.reshape(obs.shape[0], obs.shape[1], -1)
            z_sample = z_sample.to(actor_device)
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
        world_model_device = _module_device(self.world_model)
        actor_device = _module_device(self.actor)
        if obs.dim() == 4 or obs.dim() == 2:
            obs = obs.unsqueeze(1)
        obs = obs.to(world_model_device)
        self._sync_actor_copy_device()

        with FreezeParameters([self.world_model, self.actor, self.actor_copy]):
            z, _ = self.world_model.encode(obs)
            z = z.reshape(obs.shape[0], obs.shape[1], -1)
            z = z.to(actor_device)
            return self.actor_copy(z, deterministic=True)


class ObservationActor:
    def __init__(
            self,
            actor: Actor,
            flatten_observation: bool = False,
        ):
        self.actor = actor
        self.actor_copy = copy.deepcopy(actor)
        self.flatten_observation = flatten_observation
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
        actor_device = _module_device(self.actor)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        elif obs.dim() == 3 and self.flatten_observation:
            # Treat CHW tensors as a single observation when flattening.
            obs = obs.unsqueeze(0)
        obs = obs.to(actor_device)
        if self.flatten_observation and obs.dim() >= 4:
            if obs.dim() == 4:
                obs = obs.reshape(obs.shape[0], -1)
            else:
                obs = obs.reshape(obs.shape[0], obs.shape[1], -1)
        self._sync_actor_copy_device()
        with FreezeParameters([self.actor, self.actor_copy]):
            return self.actor_copy(obs, deterministic=True)


class TransformerObservationActor(ObservationActor):
    """Backward-compatible alias for a transformer policy wrapper.

    This wrapper bypasses world-model encoding and feeds observations directly
    into the actor.
    """
    pass