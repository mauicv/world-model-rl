from reflect.utils import FreezeParameters
from reflect.utils import AdamOptim
from dataclasses import dataclass
import torch
import copy
from typing import Optional

@dataclass
class PPOTrainerLosses:
    value_loss: float
    value_grad_norm: float
    actor_loss: Optional[float]
    actor_grad_norm: Optional[float]
    entropy_loss: Optional[float]


class PPOTrainer:
    def __init__(self,
            actor,
            critic,
            actor_lr: float=0.001,
            critic_lr: float=0.001,
            grad_clip: float=100,
            gamma: float=0.99,
            lam: float=0.95,
            eta: float=0.001,
            minibatch_size: int=32,
            clip_ratio: float=0.2
        ):
        self.gamma = gamma
        self.lam = lam
        self.eta = eta
        self.gamma_rollout = None
        self.minibatch_size = minibatch_size
        self.clip_ratio = clip_ratio

        self.actor = actor
        self.actor_lr = actor_lr
        self.actor_optim = AdamOptim(
            self.actor.parameters(),
            lr=self.actor_lr,
            grad_clip=grad_clip
        )

        self.critic = critic
        self.critic_lr = critic_lr
        self.critic_optim = AdamOptim(
            self.critic.parameters(),
            lr=self.critic_lr,
            grad_clip=grad_clip
        )

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
        targets = []
        for i in range(h):
            rollout_targets = self.compute_value_target(
                target_state_values=state_sample_values[:, i:, :].detach(),
                states=state_samples[:, i:, :],
                rewards=reward_samples[:, i:, :],
                dones=done_samples[:, i:, :],
            )
            targets.append(rollout_targets)
        target_values = torch.stack(targets, dim=1)
        advantages = target_values.detach() - state_sample_values
        loss = 0.5 * (advantages)**2
        return loss.mean(), advantages

    def actor_update(self, advantages, state_samples, action_samples):
        old_action_dist = self.actor(state_samples)
        old_action_log_probs = old_action_dist.log_prob(action_samples)[:, None].detach()

        b, h, *l = state_samples.shape
        inds = torch.randperm(b)
        for i in range(0, b, self.minibatch_size):
            sample_inds = inds[i:i+self.minibatch_size]
            state_sample = state_samples[sample_inds]
            action_sample = action_samples[sample_inds]
            advantage = advantages[sample_inds]
            action_dist = self.actor(state_sample)
            entropy_loss = action_dist.entropy().mean()
            action_log_probs = action_dist.log_prob(action_sample)
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            clipped_surrogate_objective = torch.min(ratio * advantage, clipped_ratio * advantage)
            actor_loss = - clipped_surrogate_objective.mean() - self.eta * entropy_loss
            actor_gn = self.actor_optim.backward(actor_loss)
            self.actor_optim.update_parameters()
        
        # return means ... 
        return actor_loss, actor_gn, entropy_loss

    def update(
            self,
            state_samples,
            reward_samples,
            done_samples,
            action_samples,
            critic_only=False
        ):
        value_loss, advantages = self.value_loss(
            state_samples=state_samples,
            reward_samples=reward_samples,
            done_samples=done_samples
        )
        value_gn = self.critic_optim.backward(
            value_loss, 
            retain_graph=not critic_only
        )
        self.critic_optim.update_parameters()

        actor_loss, actor_gn, entropy_loss = None, None, None

        if not critic_only:
            actor_loss, actor_gn, entropy_loss = self.actor_update(
                advantages=advantages.detach(),
                state_samples=state_samples.detach(),
                action_samples=action_samples.detach()
            )

        return PPOTrainerLosses(
            value_loss=value_loss.item(),
            value_grad_norm=value_gn.item(),
            actor_loss=actor_loss.item() if actor_loss is not None else None,
            entropy_loss=entropy_loss.item() if entropy_loss is not None else None,
            actor_grad_norm=actor_gn.item() if actor_loss is not None else None,
        )

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

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

