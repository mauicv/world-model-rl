import copy
import torch
from reflect.models.td3_policy import TAU, GAMMA
from reflect.models.td3_policy.actor import Actor
from reflect.models.td3_policy.critic import Critic
from reflect.utils import AdamOptim


def to_tensor(t):
    if isinstance(t, torch.Tensor):
        return t
    return torch.tensor(t)



def update_target_network(target_model, model, tau=TAU):
    with torch.no_grad():
        for target_weights, weights in zip(
                target_model.parameters(),
                model.parameters()
            ):
            target_weights.data = (
                tau * weights.data
                + (1 - tau) * target_weights.data
            )


def compute_TD_target(
        next_states,
        rewards,
        dones,
        target_actor,
        target_critic,
        gamma=GAMMA
    ):
    next_state_actions = target_actor.compute_action(
        next_states,
        eps=0
    )
    next_state_action_values = target_critic(
        next_states,
        next_state_actions
    )
    targets = rewards + gamma * (1 - dones) * next_state_action_values[:, 0]
    return targets.detach()


class Agent:
    def __init__(self, state_dim, action_space, actor_lr, critic_lr, tau=TAU):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        # Init Actor
        self.actor = Actor(input_dim=state_dim, action_space=action_space)
        self.target_actor = Actor(input_dim=state_dim, action_space=action_space)
        self.target_actor.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        self.actor_optim = AdamOptim(self.actor.parameters(), lr=self.actor_lr)

        # Init Critic 1
        self.critic_1 = Critic(state_dim=state_dim, action_space=action_space)
        self.target_critic_1 = Critic(state_dim=state_dim, action_space=action_space)
        self.target_critic_1.load_state_dict(copy.deepcopy(self.critic_1.state_dict()))
        self.critic_1_optim = AdamOptim(self.critic_1.parameters(), lr=self.critic_lr)

        # Init Critic 2
        self.critic_2 = Critic(state_dim=state_dim, action_space=action_space)
        self.target_critic_2 = Critic(state_dim=state_dim, action_space=action_space)
        self.target_critic_2.load_state_dict(copy.deepcopy(self.critic_2.state_dict()))
        self.critic_2_optim = AdamOptim(self.critic_2.parameters(), lr=self.critic_lr)

    def update_critic(
            self,
            current_states,
            next_states,
            current_actions,
            rewards,
            dones,
            gamma=GAMMA
        ):
        targets_1 = compute_TD_target(
            next_states,
            rewards,
            dones,
            self.target_actor,
            self.target_critic_1,
            gamma=gamma
        )

        targets_2 = compute_TD_target(
            next_states,
            rewards,
            dones,
            self.target_actor,
            self.target_critic_2,
            gamma=gamma
        )

        cat_targets = torch.cat(
            (targets_1[:, None], targets_2[:, None]),
            dim=-1
        )
        targets = torch.min(cat_targets, dim=-1).values
        current_state_action_values_1 = self.critic_1(current_states, current_actions)
        current_state_action_values_2 = self.critic_2(current_states, current_actions)
        loss_1 = (targets.detach() - current_state_action_values_1[:, 0])**2
        loss_1 = loss_1.mean()
        loss_2 = (targets.detach() - current_state_action_values_2[:, 0])**2
        loss_2 = loss_2.mean()
        self.critic_1_optim.step(loss_1)
        self.critic_2_optim.step(loss_2)
        return loss_1.detach().item(), loss_2.detach().item()

    def update_actor(self, states):
        actions = self.actor(states)
        action_values = - self.critic_1(states, actions)
        actor_loss=action_values.mean()
        self.actor_optim.step(actor_loss)
        return actor_loss.detach().item()

    def update_critic_target_network(self):
        update_target_network(self.target_critic_1, self.critic_1, tau=self.tau)
        update_target_network(self.target_critic_2, self.critic_2, tau=self.tau)

    def update_actor_target_network(self):
        update_target_network(self.target_actor, self.actor, tau=self.tau)

    def to(self, device):
        self.actor.to(device)
        self.target_actor.to(device)
        self.critic_1.to(device)
        self.target_critic_1.to(device)
        self.critic_2.to(device)
        self.target_critic_2.to(device)

    def save(self, path):
        state_dict = {
            'actor': self.actor.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'actor_optim': self.actor_optim.optimizer.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'critic_1_optim': self.critic_1_optim.optimizer.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'critic_2_optim': self.critic_2_optim.optimizer.state_dict(),
        }
        torch.save(state_dict, f'{path}/agent.pth')

    def load(self, path):
        checkpoint = torch.load(f'{path}/agent.pth')
        self.actor.load_state_dict(checkpoint['actor'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.actor_optim.optimizer \
            .load_state_dict(checkpoint['actor_optim'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.critic_1_optim.optimizer \
            .load_state_dict(checkpoint['critic_1_optim'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])
        self.critic_2_optim.optimizer \
            .load_state_dict(checkpoint['critic_2_optim'])
