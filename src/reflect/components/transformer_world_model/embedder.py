import torch


class BaseEmbedder(torch.nn.Module):
    def __init__(
            self,
            z_dim: int=None,
            a_size: int=None,
            hidden_dim: int=None,
        ):
        super(BaseEmbedder, self).__init__()
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


class StackEmbedder(BaseEmbedder):
    def __init__(
            self,
            z_dim: int=None,
            a_size: int=None,
            hidden_dim: int=None,
        ):
        super(StackEmbedder, self).__init__(
            z_dim=z_dim,
            a_size=a_size,
            hidden_dim=hidden_dim,
        )

    def forward(self, x, *args, **kwargs):
        s, a, r = x
        b, *_ = s.shape
        a_emb = self.a_emb(a)
        r_emb = self.r_emb(r.type(torch.float))
        s_emb = self.z_emb(s)
        return (
            torch.stack([s_emb, a_emb, r_emb], dim=2)
            .view(b, -1, self.hidden_dim)
        )


class AddEmbedder(BaseEmbedder):
    def __init__(
            self,
            z_dim: int=None,
            a_size: int=None,
            hidden_dim: int=None,
        ):
        super(AddEmbedder, self).__init__(
            z_dim=z_dim,
            a_size=a_size,
            hidden_dim=hidden_dim,
        )

    def forward(self, x, *args, **kwargs):
        s, a, r = x
        b, *_ = s.shape
        a_emb = self.a_emb(a)
        r_emb = self.r_emb(r.type(torch.float))
        s_emb = self.z_emb(s)
        return s_emb + a_emb + r_emb


class ConcatEmbedder(BaseEmbedder):
    def __init__(
            self,
            z_dim: int=None,
            a_size: int=None,
            hidden_dim: int=None,
        ):
        super(ConcatEmbedder, self).__init__(
            z_dim=z_dim,
            a_size=a_size,
            hidden_dim=hidden_dim,
        )

    def forward(self, x, *args, **kwargs):
        s, a, r = x
        b, *_ = s.shape
        a_emb = self.a_emb(a)
        r_emb = self.r_emb(r.type(torch.float))
        s_emb = self.z_emb(s)
        return torch.concat([s_emb, a_emb, r_emb], dim=-1)
