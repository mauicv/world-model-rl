from reflect.models.rl.actor import Actor
from reflect.utils import AdamOptim
import torch


class STGradAgent:
    def __init__(self,
            state_dim,
            action_space,
            actor_lr
        ):

        self.actor_lr = actor_lr
        self.actor = Actor(input_dim=state_dim, action_space=action_space)
        self.actor_optim = AdamOptim(self.actor.parameters(), lr=self.actor_lr)

    def update(
        self,
        reward_samples,
        done_samples
    ):
        loss = - (reward_samples * done_samples).sum()
        self.actor_optim.backward(loss)
        self.actor_optim.update_parameters()

    def to(self, device):
        self.actor.to(device)
        self.target_actor.to(device)

    def save(self, path):
        state_dict = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.optimizer.state_dict(),
        }
        torch.save(state_dict, f'{path}/agent.pth')

    def load(self, path):
        checkpoint = torch.load(f'{path}/agent.pth')
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_optim.optimizer \
            .load_state_dict(checkpoint['actor_optim'])
