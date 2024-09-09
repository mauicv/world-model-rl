import torch


class Embedder(torch.nn.Module):
    def __init__(
            self,
            z_dim: int=None,
            a_size: int=None,
            hidden_dim: int=None,
        ):
        super(Embedder, self).__init__()
        self.hidden_dim = hidden_dim

        self.emb = torch.nn.Sequential(
            torch.nn.Linear(a_size + z_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        s, a, r = x
        z = torch.cat([a, s], dim=-1)
        return self.emb(z)