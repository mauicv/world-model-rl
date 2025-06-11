import torch
import torch.distributions as D
import random


class MLP(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float=0.0
        ):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
    

class EnsembleMLP(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            ensemble_size: int=1,
            dropout: float=0.0,
            sample_iterations: int=2,
            seed: int=0
        ):
        super(EnsembleMLP, self).__init__()
        self.ensemble_size = ensemble_size
        self.sample_iterations = sample_iterations

        # make model creation deterministic
        random.seed(seed)
        self.layers = torch.nn.ModuleList([
            MLP(
                input_dim,
                hidden_dim + random.randint(0, 16),
                output_dim,
                dropout=dropout
            ) for _ in range(ensemble_size)
        ])

    def forward(self, x):
        b, t, *_ = x.shape
        assert b % self.ensemble_size == 0, "Batch size must be divisible by ensemble size"
        int_b = int(b/self.ensemble_size)
        x = x.split(int_b, dim=0)
        y = torch.cat([layer(x[i]) for i, layer in enumerate(self.layers)], dim=0)
        return y

    def sample(self, x):
        was_training = self.training
        self.train()

        it = []
        for _ in range(self.sample_iterations):
            it.extend([layer(x) for layer in self.layers])
        y = torch.stack(it, dim=0)

        if not was_training:
            self.eval()

        mu = y.mean(dim=0)
        var = y.detach().std(dim=0)
        return mu, var


class BaseHead(torch.nn.Module):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
            b_r: float=None,
            b_u: float=None,
            ensemble_size: int=1,
            dropout: float=0.0
        ):
        super(BaseHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_cat = num_cat
        self.done = MLP(hidden_dim, hidden_dim * 2, 1)
        self.b_r = b_r
        self.b_u = b_u
        if ensemble_size == 1:
            self.reward = MLP(hidden_dim, hidden_dim * 2, 1, dropout=dropout)
            self.predictor = MLP(hidden_dim, hidden_dim * 2, latent_dim * num_cat)
            self.is_ensemble = False
        else:
            self.predictor = EnsembleMLP(
                hidden_dim,
                hidden_dim * 2,
                latent_dim * num_cat,
                ensemble_size,
                dropout=dropout
            )
            self.reward = EnsembleMLP(
                hidden_dim,
                hidden_dim * 2,
                1,
                ensemble_size,
                dropout=dropout
            )
            self.is_ensemble = True
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
            b_r: float=None,
            b_u: float=None,
            ensemble_size: int=1,
            dropout: float=0.0
        ):
        super(StackHead, self).__init__(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hidden_dim,
            ensemble_size=ensemble_size,
            b_r=b_r,
            b_u=b_u,
            dropout=dropout
        )

    def forward(self, x, train=True):
        b, t, _ = x.shape
        reshaped_x = x.view(b, -1, 3, self.hidden_dim)
        s_emb, a_emb, r_emb = reshaped_x.unbind(dim=2)
        if self.is_ensemble and not train:
            s, s_u = self.predictor.sample(a_emb)
            r, r_u = self.reward.sample(r_emb)
            s_u = s_u.mean(dim=-1, keepdim=True)
        else:
            s_u, r_u = None, None
            s = self.predictor(a_emb)
            r = self.reward(r_emb)
        s = s.reshape(b, int(t/3), self.latent_dim, self.num_cat)
        z_dist = self.create_z_dist(s)
        d = self.done_output_activation(self.done(s_emb))
        return (z_dist, s_u), (r, r_u), d


class AddHead(BaseHead):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
            b_r: float=None,
            b_u: float=None,
            ensemble_size: int=1,
            dropout: float=0.0
        ):
        super(AddHead, self).__init__(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hidden_dim,
            ensemble_size=ensemble_size,
            b_r=b_r,
            b_u=b_u,
            dropout=dropout
        )

    def forward(self, x, train=True):
        b, t, _ = x.shape
        if self.is_ensemble and not train:
            s, s_u = self.predictor.sample(x)
            r, r_u = self.reward.sample(x)
            s_u = s_u.mean(dim=-1, keepdim=True)
        else:
            s_u, r_u = None, None
            s = self.predictor(x)
            r = self.reward(x)
        d = self.done_output_activation(self.done(x))
        s = s.reshape(b, t, self.latent_dim, self.num_cat)
        z_dist = self.create_z_dist(s)
        return (z_dist, s_u), (r, r_u), d


class ConcatHead(BaseHead):
    def __init__(
            self,
            latent_dim: int=None,
            num_cat: int=None,
            hidden_dim: int=None,
            b_r: float=None,
            b_u: float=None,
            ensemble_size: int=1,
            dropout: float=0.0
        ):
        super(ConcatHead, self).__init__(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hidden_dim,
            ensemble_size=ensemble_size,
            b_r=b_r,
            b_u=b_u,
            dropout=dropout
        )

    def forward(self, x, train=True):
        b, t, d = x.shape
        split_size = int(d/3)
        s_emb, a_emb, r_emb = torch.split(x, split_size, dim=-1)
        if self.is_ensemble and not train:
            s, s_u = self.predictor.sample(a_emb)
            r, r_u = self.reward.sample(r_emb)
            s_u = s_u.mean(dim=-1, keepdim=True)
        else:
            s_u, r_u = None, None
            s = self.predictor(a_emb)
            r = self.reward(r_emb)
        s = s.reshape(b, t, self.latent_dim, self.num_cat)
        z_dist = self.create_z_dist(s)
        d = self.done_output_activation(self.done(s_emb))
        return (z_dist, s_u), (r, r_u), d