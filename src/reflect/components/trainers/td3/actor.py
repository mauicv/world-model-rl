import torch


class TD3Actor(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_layers=3,
            hidden_dim=512,
            bound=1.0,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers=num_layers
        self.hidden_dim=hidden_dim
        self.bound = torch.tensor(bound)
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

        layers.extend([
            torch.nn.Linear(hidden_dim, self.output_dim),
            torch.nn.Tanh()
        ])
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, deterministic=True):
        x = self.layers(x)
        action = x * self.bound
        return action