from reflect.utils import AdamOptim
from dataclasses import dataclass
import torch
from typing import Optional
import numpy as np
import torch.nn.functional as F
from itertools import chain

@dataclass
class PPOTrainerLosses:
    value_loss: float
    actor_loss: Optional[float]
    grad_norm: Optional[float]
    entropy_loss: Optional[float]
    clipfrac: Optional[float]
    approxkl: Optional[float]
    num_epochs: int
    lr: Optional[float] = None


class PPOTrainer:
    def __init__(self,
            actor,
            critic,
            lr: float=1e-4,
            grad_clip: float=0.5,
            gamma: float=0.99,
            lam: float=0.95,
            eta: float=0.001,
            clip_ratio: float=0.05,
            target_kl: float=0.2,
            num_minibatch: int=16,
            update_epochs: int=3,
            vf_coef: float=0.5,
            adaptive_lr: bool=False,
            adaptive_lr_min: float=1e-5,
            adaptive_lr_max: float=1e-2,
        ):
        self.gamma = gamma
        self.lam = lam
        self.eta = eta
        self.num_minibatch = num_minibatch
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.vf_coef = vf_coef
        self.adaptive_lr = adaptive_lr
        self.adaptive_lr_min = adaptive_lr_min
        self.adaptive_lr_max = adaptive_lr_max
        self.actor = actor
        self.critic = critic
        self.lr = lr
        self.optim = AdamOptim(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=self.lr,
            grad_clip=grad_clip,
            eps=1e-5
        )

    def compute_gae(
            self,
            critic_states,
            rewards,
            dones
        ):
        _, l, *_ = rewards.shape
        values = self.critic(critic_states)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(l-1)):
            next_value = values[:, t + 1]
            nextnonterminal = 1.0 - dones[:, t + 1]
            delta = rewards[:, t] + self.gamma * next_value * nextnonterminal - values[:, t]
            advantages[:, t] = delta + self.gamma * self.lam * nextnonterminal * last_advantage
            last_advantage = advantages[:, t]
        returns = advantages + values
        # drop the last advantage and return because we don't have the next_done or next_state.
        advantages = advantages[:, :-1, :].squeeze(-1)
        returns = returns[:, :-1, :].squeeze(-1)
        return advantages, returns

    def actor_update(
            self,
            advantages,
            returns,
            actor_state_samples,
            critic_state_samples,
            action_samples,
            num_minibatch: Optional[int]=None,
            update_epochs: Optional[int]=None,
        ):
        if num_minibatch is None:
            num_minibatch = self.num_minibatch
        if update_epochs is None:
            update_epochs = self.update_epochs

        with torch.no_grad():
            old_action_dist = self.actor(actor_state_samples)
            old_action_log_probs = old_action_dist \
                .log_prob(action_samples) \
                .detach() \
                .sum(-1)

        value_losses = []
        actor_losses = []
        entropy_losses = []
        grad_norms = []
        clipfracs = []
        approxkls = []

        actor_state_samples = actor_state_samples.reshape(-1, *actor_state_samples.shape[2:])
        critic_state_samples = critic_state_samples.reshape(-1, *critic_state_samples.shape[2:])
        action_samples = action_samples.reshape(-1, *action_samples.shape[2:])
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        returns = returns.reshape(-1, *returns.shape[2:])
        old_action_log_probs = old_action_log_probs.reshape(-1, *old_action_log_probs.shape[2:])
        b, *_ = actor_state_samples.shape
        minibatch_size = int(b / self.num_minibatch)

        for epoch in range(self.update_epochs):
            inds = torch.randperm(b, device=advantages.device)
            for i in range(0, b, minibatch_size):
                sample_inds = inds[i:i+minibatch_size]
                actor_state_minibatch = actor_state_samples[sample_inds]
                critic_state_minibatch = critic_state_samples[sample_inds]
                action_minibatch = action_samples[sample_inds]
                old_action_log_probs_minibatch = old_action_log_probs[sample_inds]
                advantage_minibatch = advantages[sample_inds]

                action_dist = self.actor(actor_state_minibatch)
                entropy_loss = -action_dist.entropy().sum(-1).mean()
                action_log_probs_minibatch = action_dist \
                    .log_prob(action_minibatch) \
                    .sum(-1)
                diff = action_log_probs_minibatch - old_action_log_probs_minibatch
                ratio = torch.exp(diff)

                with torch.no_grad():
                    clipfrac = ((1 - ratio).abs() > self.clip_ratio).float().mean()
                    approxkl = ((ratio - 1) - torch.log(ratio)).mean()
                    clipfracs.append(clipfrac.item())
                    approxkls.append(approxkl.item())
                    if not self.adaptive_lr and approxkl > self.target_kl:
                        break

                advantage_minibatch = (advantage_minibatch - advantage_minibatch.mean()) / (advantage_minibatch.std() + 1e-8)
                assert advantage_minibatch.shape == ratio.shape
                pg_loss_1 = - advantage_minibatch * ratio
                pg_loss_2 = - advantage_minibatch * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                values_minibatch = self.critic(critic_state_minibatch).view(-1)
                value_loss = F.mse_loss(returns[sample_inds], values_minibatch)
                
                loss = pg_loss + self.eta * entropy_loss + self.vf_coef * value_loss
                gn = self.optim.backward(loss)
                self.optim.update_parameters()

                value_losses.append(value_loss.item())
                actor_losses.append(pg_loss.item())
                entropy_losses.append(entropy_loss.item())
                grad_norms.append(gn.item())
                clipfracs.append(clipfrac.item())
                approxkls.append(approxkl.item())

            if self.adaptive_lr:
                if approxkl > 2 * self.target_kl:
                    self.lr *= 0.5
                elif approxkl < 0.5 * self.target_kl:
                    self.lr *= 1.5
                self.lr = min(max(self.lr, self.adaptive_lr_min), self.adaptive_lr_max)
                self.optim.update_lr(self.lr)
            elif approxkl > self.target_kl:
                break

        return (
            np.mean(value_losses),
            np.mean(actor_losses),
            np.mean(entropy_losses),
            np.mean(grad_norms),
            np.mean(clipfracs),
            np.mean(approxkls),
            epoch,
            self.lr
        )

    def update(
            self,
            actor_state_samples,
            critic_state_samples,
            reward_samples,
            done_samples,
            action_samples,
            num_minibatch: Optional[int]=None,
            update_epochs: Optional[int]=None,
        ):
        advantages, returns = self.compute_gae(
            critic_states=critic_state_samples,
            rewards=reward_samples,
            dones=done_samples
        )
        value_loss, actor_loss, entropy_loss, grad_norm, clipfrac, approxkl, num_epochs, lr = self.actor_update(
            advantages=advantages.detach(),
            returns=returns.detach(),
            actor_state_samples=actor_state_samples[:, :-1].detach(),
            critic_state_samples=critic_state_samples[:, :-1].detach(),
            action_samples=action_samples[:,:-1].detach(),
            num_minibatch=num_minibatch,
            update_epochs=update_epochs
        )

        return PPOTrainerLosses(
            value_loss=value_loss.item(),
            actor_loss=actor_loss.item() if actor_loss is not None else None,
            entropy_loss=entropy_loss.item() if entropy_loss is not None else None,
            grad_norm=grad_norm.item() if grad_norm is not None else None,
            clipfrac=clipfrac.item() if clipfrac is not None else None,
            approxkl=approxkl.item() if approxkl is not None else None,
            num_epochs=num_epochs,
            lr=lr
        )

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def save(self, path):
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optim': self.optim.optimizer.state_dict(),
        }
        torch.save(state_dict, f'{path}/agent.pth')

    def load(self, path):
        checkpoint = torch.load(f'{path}/agent.pth')
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optim.optimizer \
            .load_state_dict(checkpoint['optim'])
