import copy
from reflect.utils import AdamOptim
from dataclasses import dataclass
import torch


@dataclass
class TD3TrainerLosses:
    value_losses: list[float]
    value_grad_norms: list[float]
    actor_loss: float
    actor_grad_norm: float


class TD3Trainer:
    def __init__(self,
            actor,
            critic,
            actor_lr: float=1e-4,
            critic_lr: float=1e-4,
            num_critics: int=2,
            grad_clip: float=0.5,
            gamma: float=0.99,
            lam: float=0.95,
            eta: float=0.001,
            actor_udpate_frequency: int=2,
            tau: float=5e-3,
            action_reg_sig: float=0.05,
            action_reg_clip: float=0.2,
        ):
        self.gamma = gamma
        self.lam = lam
        self.eta = eta
        self.tau = tau
        self.action_reg_sig = action_reg_sig
        self.action_reg_clip = action_reg_clip
        self.num_critics = num_critics
        self.actor_udpate_frequency = actor_udpate_frequency

        self.actor = actor
        self.actor_lr = actor_lr
        self.actor_optim = AdamOptim(
            self.actor.parameters(),
            lr=self.actor_lr,
            grad_clip=grad_clip,
            eps=1e-5
        )
        target_actor = copy.deepcopy(actor)
        target_actor.requires_grad_(False)
        self.target_actor = target_actor

        self.critics = []
        self.target_critics = []
        self.critic_optimizers = []
        for i in range(num_critics):
            new_critic = copy.deepcopy(critic)
            new_critic = self.randomize_critic(new_critic)
            self.critics.append(new_critic)
            target_critic = copy.deepcopy(critic)
            target_critic.requires_grad_(False)
            self.target_critics.append(target_critic)
            critic_optim = AdamOptim(
                new_critic.parameters(),
                lr=critic_lr,
                grad_clip=grad_clip,
                eps=1e-5
            )
            self.critic_optimizers.append(critic_optim)

    def randomize_critic(self, critic):
        # reinitialize the critic by adding noise to the parameters
        for param in critic.parameters():
            param.data = torch.randn_like(param.data) * 0.1
        return critic

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
            next_state_actions = torch \
                .clamp(next_state_actions, -1, 1) \
                .detach()
            next_state_action_values = target_critic(
                next_states,
                next_state_actions
            )
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
            b_targets.append(targets.view(-1, targets.shape[-1]))

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
            loss = 0.5*(targets.detach() - current_state_action_value)**2
            loss = loss.mean()
            value_gn = optimizer.backward(loss)
            optimizer.update_parameters()
            losses.append(loss.item())
            value_gns.append(value_gn.item())
        return losses, value_gns
    
    def update_actor(self, states):
        states = torch.tensor(states)
        actions = self.actor(states)
        action_values = - self.critics[0](states, actions)
        actor_loss = action_values.mean()
        actor_gn = self.actor_optim.backward(actor_loss)
        self.actor_optim.update_parameters()
        return actor_loss.item(), actor_gn.item()
    
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
            for i in range(self.actor_udpate_frequency):
                perturbed_actions = self.perturb_actions(action_samples[:, :-1])
                value_losses, value_gns = self.update_critics(
                    current_states=state_samples[:, :-1],
                    next_states=state_samples[:, 1:],
                    current_actions=perturbed_actions,
                    rewards=reward_samples[:, :-1],
                    dones=done_samples[:, :-1],
                )

            actor_loss, actor_gn = self.update_actor(
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
            )

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

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
