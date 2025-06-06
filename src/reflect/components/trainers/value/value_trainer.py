from reflect.utils import FreezeParameters
from reflect.utils import AdamOptim
from dataclasses import dataclass
import torch
import copy

def update_target_network(target_model, model, tau=5e-3):
    with FreezeParameters([target_model, model]):
        with torch.no_grad():
            for target_weights, weights in zip(
                    target_model.parameters(),
                    model.parameters()
                ):
                target_weights.data = (
                    tau * weights.data
                    + (1 - tau) * target_weights.data
                )


@dataclass
class ValueGradTrainerLosses:
    actor_loss: float
    actor_grad_norm: float
    value_loss: float
    value_grad_norm: float
    entropy_loss: float


class ValueGradTrainer:
    def __init__(self,
            actor,
            critic,
            actor_lr: float=0.001,
            critic_lr: float=0.001,
            grad_clip: float=100,
            gamma: float=0.99,
            lam: float=0.95,
            eta: float=0.001
        ):
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.actor = actor
        self.actor_optim = AdamOptim(
            self.actor.parameters(),
            lr=self.actor_lr,
            grad_clip=grad_clip
        )

        self.critic_lr = critic_lr
        self.critic = critic
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optim = AdamOptim(
            self.critic.parameters(),
            lr=self.critic_lr,
            grad_clip=grad_clip
        )
        self.lam = lam
        self.eta = eta
        self.gamma_rollout = None

    def compute_rollout_value(
            self,
            target_state_values,
            rewards,
            states,
            dones,
            k
        ):
        _, big_h, *_ = states.shape
        if self.gamma_rollout is None:
            self.gamma_rollout = torch.tensor([
                self.gamma**i for i in range(big_h)
            ]).to(states.device)
        h = min(k, big_h - 1)
        R = (
            self.gamma_rollout[:h][None, :, None]
            * rewards[:, :h]
            * (1 - dones[:, :h])
        )
        final_values = (
            self.gamma_rollout[h]
            * target_state_values[:, [h]]
            * (1 - dones[:, [h]])
        )
        return R.sum(1) + final_values[:, 0, :]

    def compute_value_target(
            self,
            target_state_values,
            rewards,
            states,
            dones
        ):
        val_sum = 0
        _, big_h, *_ = states.shape
        for k in range(1, big_h - 1):
            val = self.compute_rollout_value(
                target_state_values=target_state_values,
                rewards=rewards,
                states=states,
                dones=dones,
                k=k
            )
            val_sum = val_sum + self.lam**(k - 1) * val
        final_val = self.compute_rollout_value(
            target_state_values=target_state_values,
            rewards=rewards,
            states=states,
            dones=dones,
            k=big_h
        )
        return (1 - self.lam) * val_sum + self.lam**(big_h - 1) * final_val

    def value_loss(
            self,
            state_samples,
            reward_samples,
            done_samples,
        ):
        b, h, *l = state_samples.shape
        state_sample_values = self.critic(state_samples)
        with FreezeParameters([self.target_critic]):
            target_state_values = self.target_critic(state_samples)
            targets = []
            for i in range(h):
                rollout_targets = self.compute_value_target(
                    target_state_values=target_state_values[:, i:, :],
                    states=state_samples[:, i:, :],
                    rewards=reward_samples[:, i:, :],
                    dones=done_samples[:, i:, :],
                )
                targets.append(rollout_targets)
            target_values = torch.stack(targets, dim=1)
        loss = 0.5 * (target_values.detach() - state_sample_values)**2
        return loss.mean(), target_values

    def update(
            self,
            state_samples,
            reward_samples,
            done_samples,
            entropy=None
        ):
        value_loss, target_values = self.value_loss(
            state_samples=state_samples,
            reward_samples=reward_samples,
            done_samples=done_samples
        )
        value_gn = self.critic_optim.backward(
            value_loss, 
            retain_graph=True
        )
        self.critic_optim.update_parameters()

        if entropy is not None:
            entropy = entropy.mean()
            actor_loss = - target_values.mean() - self.eta * entropy
            entropy_loss = entropy.item()
        else:
            actor_loss = - target_values.mean()
            entropy_loss = 0.0
        actor_gn = self.actor_optim.backward(actor_loss)
        self.actor_optim.update_parameters()

        update_target_network(self.target_critic, self.critic)
        return ValueGradTrainerLosses(
            value_loss=value_loss.item(),
            value_grad_norm=value_gn.item(),
            actor_loss=actor_loss.item(),
            entropy_loss=entropy_loss,
            actor_grad_norm=actor_gn.item(),
        )

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.target_critic.to(device)

    def save(self, path):
        state_dict = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.optimizer.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optim': self.critic_optim.optimizer.state_dict(),
        }
        torch.save(state_dict, f'{path}/agent.pth')

    def load(self, path):
        device = next(self.actor.parameters()).device
        checkpoint = torch.load(f'{path}/agent.pth', map_location=torch.device(device))
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_optim.optimizer \
            .load_state_dict(checkpoint['actor_optim'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optim.optimizer \
            .load_state_dict(checkpoint['critic_optim'])
        self.target_critic = copy.deepcopy(self.critic)

