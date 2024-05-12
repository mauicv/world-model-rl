import torch
import torch.nn.functional as F
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4


class Actor(torch.nn.Module):
    def __init__(self, input_dim, action_space):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = action_space.shape[0]
        self.bounds = (
            torch.tensor(
                action_space.low,
                dtype=torch.float32
            ),
            torch.tensor(
                action_space.high,
                dtype=torch.float32
            )
        )

        self.fc1 = torch.nn.Linear(self.input_dim, 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.fc3 = torch.nn.Linear(300, self.output_dim)

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

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        l, u = self.bounds
        return torch.sigmoid(x) * (u - l) + l

    def compute_action(self, state, eps=0):
        self.eval()
        if len(state.shape) == 1: state=state[None, :]
        if not torch.is_tensor(state): state = torch.tensor(state, dtype=torch.float32)
        action = self(state)
        nosie = torch.randn_like(action, device=state.device) * eps
        action = action + nosie
        action = torch.clip(action, *self.bounds)
        action = action.to(state.device)
        self.train()
        return action.detach()
