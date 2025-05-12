import torch


class ValueCritic(torch.nn.Module):
    def __init__(self, state_dim, num_layers=3, hidden_dim=512):
        super().__init__()
        self.state_dim = state_dim
        self.num_layers=num_layers
        self.hidden_dim=hidden_dim

        layers = []
        layers.extend([
            torch.nn.Linear(self.state_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ELU()
        ])
        for _ in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ELU()
            ])

        final_layer = torch.nn.Linear(hidden_dim, 1)
        layers.append(final_layer)
        self.layers = torch.nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain('elu'))
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)
