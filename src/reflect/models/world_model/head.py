import torch
import torch.distributions as D
from reflect.utils import create_z_dist

from pytfex.transformer.make_model import TransformerObjectRegistry


@TransformerObjectRegistry.register('Head')
class Head(torch.nn.Module):
    def __init__(
            self,
            latent_dim: int=None,
            hidden_dim: int=None,
        ):
        super(Head, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, latent_dim * 2)
        )
        self.reward = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, 1)
        )
        self.done = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, 1)
        )
        self.done_output_activation = torch.nn.Sigmoid()

    def forward(self, x):
        b, t, _ = x.shape
        r = self.reward(x)
        s = self.predictor(x)
        d = self.done_output_activation(self.done(x))
        s = s.reshape(b, t, -1)
        mean, std = s.chunk(2, dim=-1)
        z_dist = create_z_dist(mean, std)
        return z_dist, r, d