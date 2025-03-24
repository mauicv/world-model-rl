from reflect.components.models.actor import Actor
from reflect.utils import AdamOptim
from dataclasses import dataclass
import torch

@dataclass
class RewardGradTrainerLosses:
    actor_loss: float
    entropy_loss: float
    actor_grad_norm: float


class RewardGradTrainer:
    def __init__(self,
            actor: Actor,
            lr: float=8e-05,
            grad_clip: float=1,
            eta: float=0.001
        ):
        self.eta = eta
        self.lr = lr
        self.grad_clip = grad_clip
        self.actor = actor
        self.actor_optim = AdamOptim(
            self.actor.parameters(),
            grad_clip=self.grad_clip,
            lr=self.lr
        )

    def update(
        self,
        reward_samples,
        done_samples,
        entropy=None
    ) -> RewardGradTrainerLosses:
        batch_size = reward_samples.shape[0]

        actor_loss = - ((1 - done_samples.detach()) * reward_samples).sum()/batch_size
        if entropy is not None:
            entropy = entropy.mean()
            total_loss = actor_loss - self.eta * entropy
            entropy_loss = entropy.item()
        else:
            total_loss = actor_loss
            entropy_loss = 0.0        
        
        grad_norm = self.actor_optim.backward(total_loss)
        self.actor_optim.update_parameters()
        return RewardGradTrainerLosses(
            actor_loss=actor_loss.item(),
            entropy_loss=entropy_loss,
            actor_grad_norm=grad_norm.item()
        )

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
