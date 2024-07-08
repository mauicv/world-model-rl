import torch
import torch.nn.functional as F
import torch.distributions as D
WEIGHTS_FINAL_INIT = 3e-1
BIAS_FINAL_INIT = 3e-2


def tanh_normal(mu, std):
    # TanhTransform(Normal(5 tanh(x/5)))
    mean = 5 * torch.tanh(mu / 5)  # clip tanh arg to (-5, 5)
    std = F.softplus(std) + 0.1  # min_std = 0.1
    normal = D.normal.Normal(mean, std)
    normal = D.independent.Independent(normal, 1)
    tanh = D.TransformedDistribution(normal, [D.TanhTransform()])
    tanh.entropy = normal.entropy
    return tanh


class Actor(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            action_space,
            num_layers=3,
            hidden_dim=512,
            repeat=1
        ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = action_space.shape[0]
        self.repeat = repeat
        self.count = 0
        self.action = None

        self.bounds = (
            torch.tensor(action_space.low, dtype=torch.float32),
            torch.tensor(action_space.high, dtype=torch.float32)
        )
        self.num_layers=num_layers
        self.hidden_dim=hidden_dim

        layers = []
        layers.extend([
            torch.nn.Linear(
                self.input_dim, hidden_dim
            ),
            torch.nn.ELU()
        ])
        for _ in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ELU()
            ])

        self.layers = torch.nn.Sequential(*layers)
        self.mu = torch.nn.Linear(hidden_dim, self.output_dim)
        self.stddev = torch.nn.Linear(hidden_dim, self.output_dim)
        torch.nn.init.uniform_(
            self.stddev.weight,
            -WEIGHTS_FINAL_INIT,
            WEIGHTS_FINAL_INIT
        )
        torch.nn.init.uniform_(
            self.stddev.bias,
            -BIAS_FINAL_INIT,
            BIAS_FINAL_INIT
        )

        torch.nn.init.uniform_(
            self.mu.weight,
            -WEIGHTS_FINAL_INIT,
            WEIGHTS_FINAL_INIT
        )
        torch.nn.init.uniform_(
            self.mu.bias,
            -BIAS_FINAL_INIT,
            BIAS_FINAL_INIT
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.bounds = tuple(map(
            lambda x: x.to(*args, **kwargs),
            self.bounds
        ))
        return self

    def reset(self):
        self.count = 0
        self.action = None

    def _forward(self, x, deterministic=False):
        x = self.layers(x)
        mu = self.mu(x)
        if deterministic:
            action = torch.tanh(5 * torch.tanh(mu / 5))
            return action
        return tanh_normal(mu, self.stddev(x))

    def forward(self, x, deterministic=False):
        if self.repeat == 1:
            return self._forward(x, deterministic)

        if self.action == None:
            self.action = self._forward(x, deterministic)

        if self.count == self.repeat:
            self.count = 0
            self.action = self._forward(x, deterministic)

        self.count += 1
        return self.action

    def compute_action(self, state, deterministic=False):
        device = next(self.parameters()).device
        self.eval()
        if len(state.shape) == 1: state=state[None, :]
        if not torch.is_tensor(state):
            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            )
        action = self(state, deterministic=deterministic)
        self.train()
        return action