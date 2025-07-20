import copy
from reflect.utils import AdamOptim
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class TD3TrainerLosses:
    value_losses: list[float]
    value_grad_norms: list[float]
    actor_loss: float
    actor_grad_norm: float
    pd_loss: float


class TD3Trainer:
    def __init__(self,
            actor,
            critics,
            actor_lr: float=1e-5,
            critic_lr: float=1e-4,
            grad_clip: float=1,
            gamma: float=0.98,
            actor_update_frequency: int=1,
            alpha: float=0.0,
            num_actor_updates: int=3,
            tau: float=5e-3,
            action_reg_sig: float=0.2,
            action_reg_clip: float=0.5,
        ):
        self.tau = tau
        self.gamma = gamma
        self.action_reg_sig = action_reg_sig
        self.action_reg_clip = action_reg_clip
        self.num_critics = len(critics)
        self.actor_update_frequency = actor_update_frequency
        self.num_actor_updates = num_actor_updates
        self.actor_lr = actor_lr
        self.actor = actor
        self.alpha = alpha

        actor_optim = AdamOptim(
            self.actor.parameters(),
            lr=self.actor_lr,
            grad_clip=grad_clip,
        )
        self.actor_optim = actor_optim
        self.target_actor = copy.deepcopy(actor)
        self.target_actor.requires_grad_(False)

        self.critics = critics
        self.target_critics = []
        self.critic_optimizers = []
        for critic in critics:
            target_critic = copy.deepcopy(critic)
            target_critic.requires_grad_(False)
            self.target_critics.append(target_critic)
            critic_optim = AdamOptim(
                critic.parameters(),
                lr=critic_lr,
                grad_clip=grad_clip,
            )
            self.critic_optimizers.append(critic_optim)

    def compute_TD_target(
            self,
            next_states,
            rewards,
            dones,
            target_critic
        ):
        with torch.no_grad():
            next_state_actions = self.target_actor(
                next_states
            )
            perturbed_actions = self.perturb_actions(next_state_actions) \
                .clamp(-1, 1) \
                .detach()
            next_state_action_values = target_critic(
                next_states,
                perturbed_actions
            ).squeeze(-1)
            targets = rewards + self.gamma * (1 - dones) * next_state_action_values
        return targets.detach()

    def update_critics(
            self,
            current_states,
            next_states,
            current_actions,
            rewards,
            dones
        ):

        b_targets = []
        for i in range(self.num_critics):
            targets = self.compute_TD_target(
                next_states,
                rewards,
                dones,
                self.target_critics[i]
            )
            b_targets.append(targets.view(-1, 1))

        b_current_states = current_states.reshape(-1, current_states.shape[-1])
        b_current_actions = current_actions.reshape(-1, current_actions.shape[-1])

        cat_targets = torch.cat(b_targets, dim=-1)
        targets = torch.min(
            cat_targets,
            dim=-1
        ).values
        losses = []
        value_gns = []
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            current_state_action_value = critic(
                b_current_states,
                b_current_actions
            ).squeeze(-1)
            loss = (targets.detach() - current_state_action_value)**2
            loss = loss.mean()
            value_gn = optimizer.backward(loss)
            optimizer.update_parameters()
            losses.append(loss.item())
            value_gns.append(value_gn.item())
        return losses, value_gns

    def update_actor(self, states):
        policy_diff = None
        if self.num_actor_updates > 1:
            with torch.no_grad():
                old_actions = self.actor(states)
                lambda_Q = self.alpha / self.critics[0](states, old_actions).abs().mean()

        actor_losses = []
        actor_gns = []
        for i in range(self.num_actor_updates):
            actions = self.actor(states)
            action_values = - self.critics[0](states, actions)
            actor_loss = action_values.mean()
            policy_diff = (actions - old_actions).abs().mean()
            actor_loss = lambda_Q * actor_loss + policy_diff
            actor_gn = self.actor_optim.backward(actor_loss)
            self.actor_optim.update_parameters()
            actor_losses.append(actor_loss.item())
            actor_gns.append(actor_gn.item())
            policy_diff = policy_diff.item()

        return np.mean(actor_losses), np.mean(actor_gns), policy_diff

    def update_target_network(self, target_model, model):
        with torch.no_grad():
            for target_weights, weights in zip(
                    target_model.parameters(),
                    model.parameters()
                ):
                target_weights.data = (
                    self.tau * weights.data
                    + (1 - self.tau) * target_weights.data
                )

    def perturb_actions(self, action_samples):
        noise = torch.randn_like(action_samples) * self.action_reg_sig
        return action_samples + torch.clamp(
            noise,
            -self.action_reg_clip,
            self.action_reg_clip
        )

    def update(
            self,
            state_samples,
            reward_samples,
            done_samples,
            action_samples,
        ):
        for i in range(self.actor_update_frequency):
            value_losses, value_gns = self.update_critics(
                current_states=state_samples[:, :-1],
                next_states=state_samples[:, 1:],
                current_actions=action_samples[:, :-1],
                rewards=reward_samples[:, :-1].squeeze(-1),
                dones=done_samples[:, :-1].squeeze(-1),
            )

        actor_loss, actor_gn, pd_loss = self.update_actor(
            states=state_samples.reshape(-1, state_samples.shape[-1])
        )
        self.update_target_network(self.target_actor, self.actor)
        for target_critic, critic in zip(self.target_critics, self.critics):
            self.update_target_network(target_critic, critic)

        return TD3TrainerLosses(
            value_losses=value_losses,
            value_grad_norms=value_gns,
            actor_loss=actor_loss,
            actor_grad_norm=actor_gn,
            pd_loss=pd_loss,
        )

    def save(self, path):
        state_dict = {
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'actor_optim': self.actor_optim.optimizer.state_dict(),
            'critics': [critic.state_dict() for critic in self.critics],
            'target_critics': [critic.state_dict() for critic in self.target_critics],
            'critic_optimizers': [optimizer.optimizer.state_dict() for optimizer in self.critic_optimizers],
        }
        torch.save(state_dict, f'{path}/agent.pth')

    def load(self, path):
        checkpoint = torch.load(f'{path}/agent.pth')
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_optim.optimizer \
            .load_state_dict(checkpoint['actor_optim'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        for i, (critic, target_critic, critic_optim) in enumerate(zip(
                self.critics,
                self.target_critics,
                self.critic_optimizers
            )):
            critic.load_state_dict(checkpoint['critics'][i])
            target_critic.load_state_dict(checkpoint['target_critics'][i])
            critic_optim.optimizer.load_state_dict(checkpoint['critic_optimizers'][i])
