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

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(z_dim + a_size + 1, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        s, a, r = x
        s_emb = self.mlp(torch.cat([s, a, r], dim=-1))
        return s_emb
