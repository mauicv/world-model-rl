import torch
import torch.nn.functional as F
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_space):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_space.shape[0]
        self.fc1 = torch.nn.Linear(
            self.state_dim + self.action_dim, 400
        )
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, 1)

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

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
