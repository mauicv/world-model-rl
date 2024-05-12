from reflect.models.td3_policy.actor import Actor
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.observation_model import ObservationalModel
from reflect.models.world_model.dynamic_model import DynamicsModel
import torch
from reflect.models.td3_policy import EPS


class WorldModel(torch.nn.Module):
    def __init__(
            self,
            observation_model: ObservationalModel,
            dynamic_model: DynamicsModel,
            num_ts: int,
        ):
        super().__init__()
        self.observation_model, = observation_model,
        self.dynamic_model = dynamic_model
        self.num_ts = num_ts

    def forward(self, x):
        z_dist = self.obs_model.encode(x)
        z = z_dist.rsample()
        y = self.observation_model.decoder(z)
        y_hat = self.dynamic_model(y)
        return y_hat, z, z_dist

    def encode(self, image):
        b, t, c, h, w = image.shape
        image = image.reshape(b * t, c, h, w)
        z = self.observation_model.encode(image)
        return z.reshape(b, t, -1)

    def step(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
        ):
        z_dist, new_r, new_d = self.dynamic_model((
            z[:, -self.num_ts:],
            a[:, -self.num_ts:],
            r[:, -self.num_ts:]
        ))

        new_z = z_dist.sample()[:, -1].reshape(-1, 1, 32 * 32)
        z = torch.cat([z, new_z], dim=1)

        new_r = new_r[:, -1].reshape(-1, 1, 1)
        r = torch.cat([r, new_r], dim=1)

        new_d = new_d[:, -1].reshape(-1, 1, 1)
        d = torch.cat([d, new_d], dim=1)

        return z, r, d


class Environment():
    def __init__(
            self,
            world_model: WorldModel,
            data_loader: EnvDataLoader,
            batch_size: int
        ) -> None:
        self.world_model = world_model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.states = None
        self.actions = None
        self.rewards = None
        self.dones = None

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
        return self.dones[:, -1, 0] <= 0.5

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