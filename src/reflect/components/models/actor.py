import torch
import torch.distributions as D
from reflect.utils import FreezeParameters


class Actor(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            bound,
            num_layers=3,
            hidden_dim=512,
            action_scale=5.0,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.count = 0
        self.action = None
        self.action_scale = action_scale
        self.bound = torch.tensor(bound, dtype=torch.float32)
        self.num_layers=num_layers
        self.hidden_dim=hidden_dim

        layers = []
        layers.extend([
            torch.nn.Linear(
                self.input_dim, hidden_dim
            ),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ELU()
        ])
        for _ in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ELU()
            ])

        self.layers = torch.nn.Sequential(*layers)
        self.mu = torch.nn.Linear(hidden_dim, self.output_dim)
        self.stddev = torch.nn.Linear(hidden_dim, self.output_dim)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.bound = self.bound.to(*args, **kwargs)
        return self

    def forward(self, x, deterministic=False):
        x = self.layers(x)
        mu = self.mu(x)
        mu = torch.tanh(mu / self.action_scale) * self.bound
        if deterministic:
            return mu

        min_std=0.01
        max_std=1.0
        std = self.stddev(x)
        std = max_std * torch.sigmoid(std) + min_std
        normal = D.normal.Normal(mu, std)
        return D.independent.Independent(normal, 1)

    def compute_action(self, state, deterministic=False):
        with FreezeParameters([self]):
            device = next(self.parameters()).device
            if len(state.shape) == 1: state=state[None, :]
            if not torch.is_tensor(state):
                state = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                )
            action = self(state, deterministic=deterministic)
            return action