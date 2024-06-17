from reflect.models.rl.actor import Actor
from reflect.models.rl.critic import Critic
from reflect.utils import AdamOptim
import torch


class ValueGradTrainer:
    def __init__(self,
            actor: Actor,
            critic: Critic,
            actor_lr: float=0.001,
            critic_lr: float=0.001,
            grad_clip: float=10,
            weight_decay: float=1e-4,
            gamma: float=0.99,
            lam: float=0.95
        ):
        self.gamma = gamma

        self.actor_lr = actor_lr
        self.actor = actor
        self.actor_optim = AdamOptim(
            self.actor.parameters(),
            lr=self.actor_lr,
            grad_clip=grad_clip,
            weight_decay=weight_decay
        )

        self.critic_lr = critic_lr
        self.critic = critic
        self.critic_optim = AdamOptim(
            self.critic.parameters(),
            lr=self.critic_lr,
            grad_clip=grad_clip,
            weight_decay=weight_decay
        )
        self.lam = lam
        self.gamma_rollout = None

    def compute_rollout_value(self, rewards, states, dones, k):
        _, big_h, *_ = states.shape
        if self.gamma_rollout is None:
            self.gamma_rollout = torch.tensor([
                self.gamma**i for i in range(big_h)
            ])
        h = min(k, big_h - 1)
        R = (
            self.gamma_rollout[:h][None, :, None]
            * rewards[:, :h]
            * (1 - dones[:, :h])
        )
        final_values = (
            self.gamma_rollout[h]
            * self.critic(states[:, [h]]).detach()
            * (1 - dones[:, [h]])
        )
        return R.sum(1) + final_values[:, 0, :]

    def compute_value_target(self, rewards, states, dones):
        val_sum = 0
        _, big_h, *_ = states.shape
        for k in range(big_h - 1):
            val = self.compute_rollout_value(
                rewards=rewards,
                states=states,
                dones=dones,
                k=k
            )
            val_sum = (1 - self.lam) * self.lam**k * val
        final_val = self.compute_rollout_value(
            rewards=rewards,
            states=states,
            dones=dones,
            k=big_h
        )
        return val_sum + self.lam**(big_h - 1) * final_val

    def update(
            self,
            state_samples,
            reward_samples,
            done_samples  
        ):
        critic_update = self._update_critic(
            state_samples,
            reward_samples,
            done_samples
        )
        actor_update = self._update_actor(
            state_samples=state_samples
        )
        return {
            **critic_update,
            **actor_update
        }

    def _update_actor(self, state_samples):
        state_samples
        b, h, *l = state_samples.shape
        state_samples = state_samples.reshape(b*h, *l)
        loss = - self.critic(state_samples).mean()
        self.actor_optim.backward(loss)
        self.actor_optim.update_parameters()
        return {
            'actor_loss': loss.item()
        }

    def _update_critic(
            self,
            state_samples,
            reward_samples,
            done_samples
        ):
        state_values = self.critic(state_samples[:, 0])
        targets = self.compute_value_target(
            states=state_samples,
            rewards=reward_samples,
            dones=done_samples
        )
        loss = (targets.detach() - state_values)**2
        loss = loss.mean()
        self.critic_optim.backward(loss)
        self.critic_optim.update_parameters()
        return {
            'critic_loss': loss.item()
        }

    def to(self, device):
        self.actor.to(device)

    def save(self, path):
        state_dict = {
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.optimizer.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optim': self.critic_optim.optimizer.state_dict(),
        }
        torch.save(state_dict, f'{path}/agent.pth')

    def load(self, path):
        checkpoint = torch.load(f'{path}/agent.pth')
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_optim.optimizer \
            .load_state_dict(checkpoint['actor_optim'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optim.optimizer \
            .load_state_dict(checkpoint['critic_optim'])
