import copy
import torch
from reflect.models.agent import TAU, GAMMA
from reflect.models.agent.actor import Actor
from reflect.models.agent.critic import Critic
from reflect.utils import AdamOptim


ACTION_REG_SIG=0.05
ACTION_REG_CLIP = 0.2
ACTOR_UPDATE_FREQ = 2
TAU=5e-3
ACTOR_LR=1e-3
CRITIC_LR=1e-3
GAMMA=0.99
EPS=0.5


def to_tensor(t):
    if isinstance(t, torch.Tensor):
        return t.detach()
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
    # TODO: Add noise to the next actions
    next_state_actions = target_actor.compute_action(
        next_states,
        deterministic=True
    )
    next_state_action_values = target_critic(
        next_states,
        next_state_actions
    )
    targets = rewards + gamma * (1 - dones) * next_state_action_values[:, 0]
    return targets.detach()


# TODO: Add noise generator arg, same pattern as loader...
class TD3Agent:
    def __init__(
            self,
            state_dim,
            action_space,
            actor_lr,
            critic_lr,
            grad_clip=10,
            weight_decay=1e-4,
            tau=TAU,
            entropy_weight: float=1e-3,
        ):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.entropy_weight = entropy_weight
        # Init Actor
        self.actor = Actor(
            input_dim=state_dim,
            output_dim=action_space.shape[0],
            bound=action_space.high[0],
        )
        self.target_actor = Actor(
            input_dim=state_dim,
            output_dim=action_space.shape[0],
            bound=action_space.high[0],
        )
        self.target_actor.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        self.actor_optim = AdamOptim(
            self.actor.parameters(),
            lr=self.actor_lr,
            grad_clip=grad_clip
        )

        # Init Critic 1
        self.critic_1 = Critic(state_dim=state_dim, action_space=action_space)
        self.target_critic_1 = Critic(state_dim=state_dim, action_space=action_space)
        self.target_critic_1.load_state_dict(copy.deepcopy(self.critic_1.state_dict()))
        self.critic_1_optim = AdamOptim(
            self.critic_1.parameters(),
            lr=self.critic_lr,
            grad_clip=grad_clip
        )

        # Init Critic 2
        self.critic_2 = Critic(state_dim=state_dim, action_space=action_space)
        self.target_critic_2 = Critic(state_dim=state_dim, action_space=action_space)
        self.target_critic_2.load_state_dict(copy.deepcopy(self.critic_2.state_dict()))
        self.critic_2_optim = AdamOptim(
            self.critic_2.parameters(),
            lr=self.critic_lr,
            grad_clip=grad_clip
        )

    def update(
        self,
        state_samples,
        next_state_samples,
        action_samples,
        reward_samples,
        done_samples
    ):
        actor_loss = None
        noise = torch.randn_like(action_samples) * ACTION_REG_SIG
        action_samples = action_samples + torch.clamp(
            noise,
            -ACTION_REG_CLIP,
            ACTION_REG_CLIP
        )

        q_loss_1, q_loss_2 = self.update_critic(
            state_samples,
            next_state_samples,
            action_samples,
            reward_samples,
            done_samples,
            gamma=GAMMA
        )
        self.update_critic_target_network()

        actor_loss = self.update_actor(state_samples)
        self.update_actor_target_network()

        return {
            "critic_1_loss": q_loss_1,
            "critic_2_loss": q_loss_2,
            "actor_loss": actor_loss
        }

    def update_critic(
            self,
            current_states,
            next_states,
            current_actions,
            rewards,
            dones,
            gamma=GAMMA,
        ):
        current_actions = to_tensor(current_actions)
        current_states = to_tensor(current_states)
        next_states = to_tensor(next_states)
        rewards = to_tensor(rewards)
        dones = to_tensor(dones)

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
        self.critic_1_optim.backward(loss_1, retain_graph=True)
        self.critic_1_optim.update_parameters()
        self.critic_2_optim.backward(loss_2)
        self.critic_2_optim.update_parameters()
        return loss_1.item(), loss_2.item()

    def update_actor(self, states):
        states = states.detach()
        action_dist = self.actor(states)
        action = action_dist.rsample()
        entropy = action_dist.entropy().mean()
        action_values = - self.critic_1(states, action)
        actor_loss=action_values.mean() + \
            self.entropy_weight * entropy
        self.actor_optim.backward(actor_loss)
        self.actor_optim.update_parameters()
        return actor_loss.item()

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
