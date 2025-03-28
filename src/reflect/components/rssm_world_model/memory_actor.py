from reflect.components.models.actor import Actor
from reflect.components.rssm_world_model.world_model import WorldModel
from reflect.utils import FreezeParameters
import torch
import copy


class WorldModelActor:
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

    def reset_to_original_actor(self):
        self.actor_copy = copy.deepcopy(self.actor)

    def reset(self):
        self.state = self.world_model.dynamic_model.initial_state(1)
        self.action = torch.zeros(1, self.actor.output_dim)
        device = next(self.actor.parameters()).device
        self.state.to(device)
        self.action = self.action.to(device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        s_{t} = encoder(o_{t})
        p(s_{t} | h_{t}, o_{t})
        a_{t} = actor(s_{t}, h_{t})
        h_{t+1} = f(s_{t}, a_{t}, h_{t})
        """
        #################################
        # TODO: This code is a hack which adds in a time dim if it's missing for both
        # images and states. obs.dim() == 4 is for images, obs.dim() == 2 is
        # for states for when the time dimension is missing. i.e.
        # (b, c, h, w) and (b, s) Becuase we hard code these values we can't
        # handle the case where obs.dim() == 3.
        if obs.dim() == 4 or obs.dim() == 2:
            obs = obs.unsqueeze(1)
        #################################
        device = next(self.actor.parameters()).device
        obs = obs.to(device)
        with FreezeParameters([self.world_model, self.actor, self.actor_copy]):
            embed_obs = self.world_model.encoder(obs)
            next_state = self.world_model.dynamic_model.posterior(embed_obs[:, 0], self.state)
            self.action = self.actor(next_state.get_features(), deterministic=True)
            self.state = self.world_model.dynamic_model.prior(self.action, next_state)
            return self.action

