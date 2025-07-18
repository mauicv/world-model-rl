
import torch
import torch.nn.functional as F
from torch.distributions import Normal

WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


class TD3Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, noise_std=0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, output_dim)
        self.noise_std = noise_std

        torch.nn.init.uniform_(
            self.fc3.weight,
            -WEIGHTS_FINAL_INIT,
            WEIGHTS_FINAL_INIT
        )
        torch.nn.init.uniform_(
            self.fc3.bias,
            -BIAS_FINAL_INIT,
            BIAS_FINAL_INIT
        )

    def forward(self, x, deterministic=True):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        mean = torch.tanh(x)
        if deterministic:
            return mean
        else:
            return Normal(mean, self.noise_std)