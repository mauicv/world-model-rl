from reflect.models.rl.actor import Actor
from reflect.utils import AdamOptim
import torch


class STGradAgent:
    def __init__(self,
            actor,
            actor_lr=0.001,
            grad_clip=10,
            weight_decay=1e-4
        ):

        self.actor_lr = actor_lr
        self.actor = actor
        self.actor_optim = AdamOptim(
            self.actor.parameters(),
            lr=self.actor_lr,
            grad_clip=grad_clip,
            weight_decay=weight_decay
        )

    def update(
        self,
        reward_samples,
        done_samples
    ):
        batch_size = reward_samples.shape[0]
        loss = - ((1 - done_samples.detach()) * reward_samples).sum()/batch_size
        self.actor_optim.backward(loss)
        self.actor_optim.update_parameters()
        return {
            'actor_loss': loss.item()
        }

    def to(self, device):
        self.actor.to(device)

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
