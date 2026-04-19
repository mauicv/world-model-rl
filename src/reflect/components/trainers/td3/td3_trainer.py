import copy
from reflect.utils import AdamOptim
from dataclasses import dataclass
import torch


@dataclass
class TD3CriticLosses:
    value_losses: list[float]
    value_grad_norms: list[float]


@dataclass
class TD3ActorLosses:
    actor_loss: float
    actor_grad_norm: float


class TD3Trainer:
    def __init__(self,
            actor,
            critics,
            actor_lr=3e-4,
            critic_lr=3e-4,
            grad_clip: float=1,
            gamma: float=0.98,
            tau: float=5e-3,
            action_reg_sig: float=0.2,
            action_reg_clip: float=0.5,
            n_steps: int=1,
        ):
        if len(critics) < 2:
            raise ValueError("TD3Trainer requires at least 2 critics.")
        self.tau = tau
        self.gamma = gamma
        self.n_steps = n_steps
        self.action_reg_sig = action_reg_sig
        self.action_reg_clip = action_reg_clip
        self.num_critics = len(critics)
        self.actor = actor

        actor_optim = AdamOptim(
            self.actor.parameters(),
            lr=actor_lr,
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
            n_step_rewards,
            not_done_mask,
            perturbed_actions,
            target_critic,
            gamma_n,
        ):
        with torch.no_grad():
            next_state_action_values = target_critic(
                next_states,
                perturbed_actions
            ).squeeze(-1)
            targets = n_step_rewards + gamma_n * not_done_mask * next_state_action_values
        return targets.detach()

    def update_critics(self, state_samples, reward_samples, done_samples, action_samples):
        return self._update_critics(
            state_samples=state_samples,
            reward_samples=reward_samples.squeeze(-1),
            done_samples=done_samples.squeeze(-1),
            action_samples=action_samples,
        )

    def _update_critics(
            self,
            state_samples,
            reward_samples,
            done_samples,
            action_samples,
        ):
        n = self.n_steps
        b, T_plus_1, _ = state_samples.shape
        T = T_plus_1 - 1
        num_valid = T - n + 1  # valid starting positions for n-step targets

        # Accumulate n-step discounted rewards with done masking
        n_step_rewards = torch.zeros(b, num_valid, device=state_samples.device)
        not_done_mask = torch.ones(b, num_valid, device=state_samples.device)
        for k in range(n):
            n_step_rewards += (self.gamma ** k) * not_done_mask * reward_samples[:, k:k + num_valid]
            not_done_mask = not_done_mask * (1 - done_samples[:, k:k + num_valid])

        current_states = state_samples[:, :num_valid]
        next_states = state_samples[:, n:n + num_valid]
        current_actions = action_samples[:, :num_valid]
        gamma_n = self.gamma ** n

        with torch.no_grad():
            next_state_actions = self.target_actor(next_states)
            perturbed_actions = self.perturb_actions(next_state_actions).clamp(-1, 1)

        sampled_indices = self._sample_critics(k=2)
        b_targets = []
        for i in sampled_indices:
            targets = self.compute_TD_target(
                next_states,
                n_step_rewards,
                not_done_mask,
                perturbed_actions,
                self.target_critics[i],
                gamma_n,
            )
            b_targets.append(targets.view(-1, 1))

        b_current_states = current_states.reshape(-1, current_states.shape[-1])
        b_current_actions = current_actions.reshape(-1, current_actions.shape[-1])

        cat_targets = torch.cat(b_targets, dim=-1)
        targets = torch.min(cat_targets, dim=-1).values
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
        return TD3CriticLosses(value_losses=losses, value_grad_norms=value_gns)

    def update_actor(self, state_samples):
        return self._update_actor(
            states=state_samples.reshape(-1, state_samples.shape[-1])
        )

    def _sample_critics(self, k=2):
        indices = torch.randperm(self.num_critics)[:k].tolist()
        return indices

    def _update_actor(self, states):
        sampled_indices = self._sample_critics(k=2)
        actions = self.actor(states)
        q_values = torch.stack(
            [self.critics[i](states, actions) for i in sampled_indices], dim=0
        ).mean(0)
        actor_loss = -q_values.mean()
        actor_gn = self.actor_optim.backward(actor_loss)
        self.actor_optim.update_parameters()
        self._update_target_network(self.target_actor, self.actor)
        for target_critic, critic in zip(self.target_critics, self.critics):
            self._update_target_network(target_critic, critic)
        return TD3ActorLosses(actor_loss=actor_loss.item(), actor_grad_norm=actor_gn.item())

    def _update_target_network(self, target_model, model):
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
