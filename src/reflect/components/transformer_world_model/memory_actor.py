from reflect.components.actor import Actor
from reflect.components.transformer_world_model.world_model import TransformerWorldModel
from reflect.components.transformer_world_model.state import StateDistribution
from reflect.utils import FreezeParameters
import torch


class TransformerWorldModelActor:
    def __init__(
            self,
            world_model: TransformerWorldModel,
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
        if obs.dim() == 4:
            obs = obs.unsqueeze(1)
        device = next(self.actor.parameters()).device
        obs = obs.to(device)
        with FreezeParameters([self.world_model, self.actor]):    
            discrete_logits, continuous_mean, continuous_std \
                = self.world_model.embed_observation(obs)
            state_dist = StateDistribution.from_dist(
                continuous_mean=continuous_mean,
                continuous_std=continuous_std,
                discrete=discrete_logits
            )
            action = self.actor(
                state_dist.features[:, 0],
                deterministic=True
            )
            return action
