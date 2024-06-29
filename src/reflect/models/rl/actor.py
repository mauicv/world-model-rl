import torch
import torch.nn.functional as F
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


class Actor(torch.nn.Module):
    def __init__(self, input_dim, action_space, num_layers=3, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = action_space.shape[0]
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
            torch.nn.SiLU()
        ])
        for _ in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.SiLU()
            ])

        final_layer = torch.nn.Linear(hidden_dim, self.output_dim)
        layers.append(final_layer)
        self.layers = torch.nn.Sequential(*layers)

        torch.nn.init.uniform_(
            final_layer.weight,
            -WEIGHTS_FINAL_INIT,
            WEIGHTS_FINAL_INIT
        )
        torch.nn.init.uniform_(
            final_layer.bias,
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

    def forward(self, x, deterministic=True):
        x = self.layers(x)
        l, u = self.bounds
        mean = torch.sigmoid(x) * (u - l) + l
        if deterministic:
            return mean
        pass

    def compute_action(self, state):
        device = next(self.parameters()).device
        self.eval()
        if len(state.shape) == 1: state=state[None, :]
        if not torch.is_tensor(state):
            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            )
        action = self(state)
        action = action.to(state.device)
        self.train()
        return action.detach()
