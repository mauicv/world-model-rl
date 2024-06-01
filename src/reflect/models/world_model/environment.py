from reflect.data.loader import EnvDataLoader
from reflect.models.world_model import WorldModel
import torch


class Environment():
    def __init__(
            self,
            world_model: WorldModel,
            data_loader: EnvDataLoader,
            batch_size: int,
            ignore_done: bool=False
        ) -> None:
        self.world_model = world_model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.ignore_done=ignore_done

    def reset(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        o, a, r, d = self.data_loader.sample(
            batch_size=batch_size,
            num_time_steps=1
        )
        self.states = self.world_model.encode(o)
        self.actions = torch.zeros(batch_size, 0, a.shape[-1])
        self.rewards = r
        self.dones = d
        return self.states, {}

    @property
    def not_done(self):
        if self.ignore_done:
            return torch.ones_like(self.dones[:, -1, 0]) > 0.5
        return self.dones[:, -1, 0] <= 0.5

    @property
    def done(self):
        return torch.all(~self.not_done)

    def step(self, action: torch.Tensor):
        """Batched Step function for the environment

        This function takes a batch of actions and returns the next state,
        reward, and done status for each environment. The function also
        updates the internal state of the environment.
        
        __Note__: The action tensor should have the same batch size as the
        environment's internal state. Sometimes states are done as a result
        of the last action, in which case the action tensor should not have
        an action for that state. The self.not_done property is used to
        filter out the states that are done.
        """
        self.world_model.eval()
        assert action.shape[0] == self.actions[self.not_done].shape[0], \
            "Some states are done, but actions are being passed for them."
        self.actions = torch.cat([self.actions[self.not_done], action], dim=1)
        z, r, d = self.world_model.step(
            z=self.states[self.not_done],
            a=self.actions,
            r=self.rewards[self.not_done],
            d=self.dones[self.not_done]
        )
        self.world_model.train()
        self.states = z
        self.rewards = r
        self.dones = d
        return (
            self.states[:, [-1]],
            self.rewards[:, [-1]],
            self.dones[:, [-1]]
        )