import torch


class DenseModel(torch.nn.Module):
    def __init__(
                self,
                depth: int=4,
                input_dim: int=230,
                hidden_dim: int=256,
                output_dim: int=1,
                hidden_act: torch.nn.Module=torch.nn.ReLU,
                output_act: torch.nn.Module=torch.nn.Identity
            ):
        super().__init__()
        layers = [torch.nn.Linear(input_dim, hidden_dim), hidden_act()]
        for i in range(depth - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(hidden_act())
        self.fc = torch.nn.Sequential(
            *layers,
            torch.nn.Linear(hidden_dim, output_dim)
        )
        self.output_act = output_act()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.output_act(self.fc(z))
