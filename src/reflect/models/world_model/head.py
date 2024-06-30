import torch
import torch.distributions as D

from pytfex.transformer.make_model import TransformerObjectRegistry


@TransformerObjectRegistry.register('Head')
class Head(torch.nn.Module):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
        ):
        super(Head, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_cat = num_cat
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim * 2, latent_dim * num_cat)
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

    @staticmethod
    def create_z_dist(logits, temperature=1):
        assert temperature > 0
        dist = D.OneHotCategoricalStraightThrough(
            logits=logits / temperature
        )
        return D.Independent(dist, 1)

    def forward(self, x):
        b, t, _ = x.shape
        reshaped_x = x.view(b, -1, 3, self.hidden_dim)
        s_emb, *_ = reshaped_x.unbind(dim=2)
        r = self.reward(s_emb)
        s = self.predictor(s_emb)
        d = self.done_output_activation(self.done(s_emb))
        s = s.reshape(b, int(t/3), self.latent_dim, self.num_cat)
        z_dist = self.create_z_dist(s)
        return z_dist, r, d