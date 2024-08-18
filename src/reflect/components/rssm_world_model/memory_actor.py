from reflect.components.actor import Actor
from reflect.components.rssm_world_model.world_model import WorldModel
from reflect.utils import FreezeParameters
import torch


class WorldModelActor:
    def __init__(
            self,
            world_model: WorldModel,
            actor: Actor
        ):
        self.world_model = world_model
        self.actor = actor
    
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
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        device = next(self.actor.parameters()).device
        obs = obs.to(device)
        with FreezeParameters([self.world_model, self.actor]):    
            embed_obs = self.world_model.encoder(obs)
            next_state = self.world_model.dynamic_model.posterior(embed_obs[:, 0], self.state)
            self.action = self.actor(next_state.get_features(), deterministic=True)
            self.state = self.world_model.dynamic_model.prior(self.action, next_state)
            return self.action

