import torch


class CrossAttention(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float=0.01,
        ) -> None:
        super(CrossAttention, self).__init__()
        assert hidden_dim % num_heads == 0, f"num_heads must divide hidden_dim, {hidden_dim=}, {num_heads=}"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = torch.tensor(
            dropout,
            dtype=torch.float32
        )
        self.attn_dropout = torch.nn.Dropout(self.dropout)
        self.resid_dropout = torch.nn.Dropout(self.dropout)

        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        self.linear = torch.nn.Linear(
            self.hidden_dim,
            self.hidden_dim
        )

        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(
            self,
            x_cond,
            x,
        ):
        b, lq, d = x.shape
        _, lc, _ = x_cond.shape

        q = self.q_proj(x).reshape(b, lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_cond).reshape(b, lc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_cond).reshape(b, lc, self.num_heads, self.head_dim).transpose(1, 2)

        a = q @ k.transpose(-2, -1) / self.scale

        a = torch.softmax(a, dim=-1)
        a = self.attn_dropout(a)
        output = (a @ v).transpose(1, 2).reshape(b, lq, d)
        output = self.linear(output)
        output = self.resid_dropout(output)
        return output