import torch
import torch.distributions as D


class BaseHead(torch.nn.Module):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
        ):
        super(BaseHead, self).__init__()
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


class StackHead(BaseHead):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
        ):
        super(StackHead, self).__init__(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hidden_dim
        )

    def forward(self, x):
        b, t, _ = x.shape
        reshaped_x = x.view(b, -1, 3, self.hidden_dim)
        s_emb, a_emb, r_emb = reshaped_x.unbind(dim=2)
        d = self.done_output_activation(self.done(s_emb))
        r = self.reward(r_emb)
        s = self.predictor(a_emb)
        s = s.reshape(b, int(t/3), self.latent_dim, self.num_cat)
        z_dist = self.create_z_dist(s)
        return z_dist, r, d


class AddHead(BaseHead):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
        ):
        super(AddHead, self).__init__(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hidden_dim
        )

    def forward(self, x):
        b, t, _ = x.shape
        d = self.done_output_activation(self.done(x))
        r = self.reward(x)
        s = self.predictor(x)
        s = s.reshape(b, t, self.latent_dim, self.num_cat)
        z_dist = self.create_z_dist(s)
        return z_dist, r, d


class ConcatHead(BaseHead):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
        ):
        super(ConcatHead, self).__init__(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hidden_dim
        )

    def forward(self, x):
        b, t, d = x.shape
        split_size = int(d/3)
        s_emb, a_emb, r_emb = torch.split(x, split_size, dim=-1)
        d = self.done_output_activation(self.done(s_emb))
        r = self.reward(r_emb)
        s = self.predictor(a_emb)
        s = s.reshape(b, t, self.latent_dim, self.num_cat)
        z_dist = self.create_z_dist(s)
        return z_dist, r, d


class StateActionStackHead(BaseHead):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
        ):
        super(StateActionStackHead, self).__init__(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hidden_dim
        )

    def forward(self, x):
        b, t, _ = x.shape
        reshaped_x = x.view(b, -1, 2, self.hidden_dim)
        s_emb, a_emb = reshaped_x.unbind(dim=2)
        d = self.done_output_activation(self.done(s_emb))
        r = self.reward(s_emb)
        s = self.predictor(a_emb)
        s = s.reshape(b, int(t/2), self.latent_dim, self.num_cat)
        z_dist = self.create_z_dist(s)
        return z_dist, r, d
    
    def compute_reward(self, x):
        b, t, _ = x.shape
        reshaped_x = x.view(b, -1, 2, self.hidden_dim)
        s_emb, a_emb = reshaped_x.unbind(dim=2)
        r = self.reward(s_emb)
        return r
