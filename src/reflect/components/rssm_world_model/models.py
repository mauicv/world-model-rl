import torch

class DenseModel(torch.nn.Module):
    def __init__(
        self,
        depth: int = 3,  # number of hidden layers
        input_dim: int = 230,
        hidden_dim: int = 256,
        output_dim: int = 1,
        hidden_act: torch.nn.Module = torch.nn.SiLU,  # try GELU or SiLU
        output_act: torch.nn.Module = torch.nn.Identity,
        use_layernorm: bool = True
    ):
        super().__init__()
        layers = []
        
        # First hidden layer
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        if use_layernorm:
            layers.append(torch.nn.LayerNorm(hidden_dim))
        layers.append(hidden_act())
        
        # Additional hidden layers
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(hidden_act())
        
        # Final output layer (no normalization here)
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        
        self.fc = torch.nn.Sequential(*layers)
        self.output_act = output_act()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.output_act(self.fc(z))
