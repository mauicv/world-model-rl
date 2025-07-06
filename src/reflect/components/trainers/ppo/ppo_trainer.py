from reflect.utils import FreezeParameters
from reflect.utils import AdamOptim
from dataclasses import dataclass
import torch
import copy
from typing import Optional
import numpy as np

@dataclass
class PPOTrainerLosses:
    value_loss: float
    value_grad_norm: float
    actor_loss: Optional[float]
    actor_grad_norm: Optional[float]
    entropy_loss: Optional[float]
    clipfrac: Optional[float]
    approxkl: Optional[float]


class PPOTrainer:
    def __init__(self,
            actor,
            critic,
            actor_lr: float=1e-4,
            critic_lr: float=1e-4,
            grad_clip: float=0.5,
            gamma: float=0.99,
            lam: float=0.95,
            eta: float=0.001,
            minibatch_size: int=32,
            clip_ratio: float=0.1,
            target_kl: float=0.1
        ):
        self.gamma = gamma
        self.lam = lam
        self.eta = eta
        self.gamma_rollout = None
        self.minibatch_size = minibatch_size
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

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

    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        b, l, *_ = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(l)):
            if t == l - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
            advantages[:, t] = delta + gamma * lam * (1 - dones[:, t]) * last_advantage
            last_advantage = advantages[:, t]
        return advantages

    def compute_advantages(
            self,
            state_samples,
            reward_samples,
            done_samples,
        ):
        state_sample_values = self.critic(state_samples)
        advantages = self.compute_gae(
            rewards=reward_samples,
            values=state_sample_values,
            dones=done_samples
        )
        returns = state_sample_values + advantages
        return advantages.detach(), returns.detach()

    def actor_update(self, advantages, returns, state_samples, action_samples):
        b, h, *l = state_samples.shape

        old_action_dist = self.actor(state_samples)
        old_action_log_probs = old_action_dist \
            .log_prob(action_samples)[:, :, None] \
            .detach() \
            .sum(-1)

        actor_losses = []
        entropy_losses = []
        actor_gns = []
        clipfracs = []
        approxkls = []
        value_losses = []
        value_gns = []

        inds = torch.randperm(b)
        for i in range(0, b, self.minibatch_size):
            sample_inds = inds[i:i+self.minibatch_size]
            state_minibatch = state_samples[sample_inds]
            action_minibatch = action_samples[sample_inds]
            old_action_log_probs_minibatch = old_action_log_probs[sample_inds]
            advantage_minibatch = advantages[sample_inds]

            values_minibatch = self.critic(state_minibatch)
            value_loss = 0.5 * (values_minibatch - returns[sample_inds])**2
            value_loss = value_loss.mean()
            value_gn = self.critic_optim.backward(value_loss)
            self.critic_optim.update_parameters()
            value_losses.append(value_loss.item())
            value_gns.append(value_gn.item())

            action_dist = self.actor(state_minibatch)
            entropy_loss = action_dist.entropy().mean()
            action_log_probs_minibatch = action_dist \
                .log_prob(action_minibatch)[:, :, None] \
                .sum(-1)
            diff = action_log_probs_minibatch - old_action_log_probs_minibatch
            diff_clamped = torch.clamp(diff, min=-20, max=20)
            ratio = torch.exp(diff_clamped)

            with torch.no_grad():
                clipfrac = ((1 - ratio).abs() > self.clip_ratio).float().mean()
                approxkl = ((ratio - 1) - torch.log(ratio)).mean()
                clipfracs.append(clipfrac.item())
                approxkls.append(approxkl.item())
                if approxkl > self.target_kl:
                    print(f"Early stopping: KL too high ({approxkl.item()})")
                    break

            pg_loss_1 = - advantage_minibatch * ratio
            pg_loss_2 = - advantage_minibatch * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
            actor_loss = pg_loss - self.eta * entropy_loss
            actor_gn = self.actor_optim.backward(actor_loss)
            self.actor_optim.update_parameters()

            actor_losses.append(actor_loss.item())
            entropy_losses.append(entropy_loss.item())
            actor_gns.append(actor_gn.item())

        return (
            np.mean(actor_losses),
            np.mean(actor_gns),
            np.mean(entropy_losses),
            np.mean(clipfracs),
            np.mean(approxkls),
            np.mean(value_losses),
            np.mean(value_gns)
        )

    def update(
            self,
            state_samples,
            reward_samples,
            done_samples,
            action_samples,
        ):
        advantages, returns = self.compute_advantages(
            state_samples=state_samples,
            reward_samples=reward_samples,
            done_samples=done_samples
        )
        actor_loss, actor_gn, entropy_loss, clipfrac, approxkl, value_loss, value_gn = self.actor_update(
            advantages=advantages.detach(),
            returns=returns.detach(),
            state_samples=state_samples.detach(),
            action_samples=action_samples.detach(),
        )

        return PPOTrainerLosses(
            value_loss=value_loss.item(),
            value_grad_norm=value_gn.item(),
            actor_loss=actor_loss.item() if actor_loss is not None else None,
            entropy_loss=entropy_loss.item() if entropy_loss is not None else None,
            actor_grad_norm=actor_gn.item() if actor_loss is not None else None,
            clipfrac=clipfrac.item() if clipfrac is not None else None,
            approxkl=approxkl.item() if approxkl is not None else None,
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
