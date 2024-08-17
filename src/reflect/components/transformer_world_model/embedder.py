from pytfex.transformer.make_model import TransformerObjectRegistry
import torch


@TransformerObjectRegistry.register('Embedder')
class Embedder(torch.nn.Module):
    def __init__(
            self,
            z_dim: int=None,
            a_size: int=None,
            hidden_dim: int=None,
        ):
        super(Embedder, self).__init__()
        self.hidden_dim = hidden_dim
        self.r_emb = torch.nn.Linear(1, hidden_dim)

        self.a_emb = torch.nn.Sequential(
            torch.nn.Linear(a_size, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

        self.z_emb = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        s, a, r = x
        b, *_ = s.shape
        a_emb = self.a_emb(a)
        r_emb = self.r_emb(r.type(torch.float))
        s_emb = self.z_emb(s)
        return (
            torch.stack([s_emb, a_emb, r_emb], dim=2)
            .view(b, -1, self.hidden_dim)
        )
