from reflect.models.rl.actor import Actor
from reflect.models.rl.critic import Critic
from reflect.models.world_model.environment import Environment
from reflect.utils import AdamOptim
import torch
import copy


def update_target_network(target_model, model, tau=5e-3):
    with torch.no_grad():
        for target_weights, weights in zip(
                target_model.parameters(),
                model.parameters()
            ):
            target_weights.data = (
                tau * weights.data
                + (1 - tau) * target_weights.data
            )


class ValueGradTrainer:
    def __init__(self,
            actor: Actor,
            critic: Critic,
            env: Environment,
            actor_lr: float=0.001,
            critic_lr: float=0.001,
            grad_clip: float=10,
            weight_decay: float=1e-4,
            gamma: float=0.99,
            lam: float=0.95,
            entropy_weight: float=1e-3,
        ):
        self.gamma = gamma
        self.env = env
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
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optim = AdamOptim(
            self.critic.parameters(),
            lr=self.critic_lr,
            grad_clip=grad_clip,
            weight_decay=weight_decay
        )
        self.lam = lam
        self.gamma_rollout = None
        self.entropy_weight = entropy_weight

    def compute_rollout_value(self, rewards, states, dones, k):
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
            * self.target_critic(states[:, [h]]).detach()
            * (1 - dones[:, [h]])
        )
        return R.sum(1) + final_values[:, 0, :]

    def compute_value_target(self, rewards, states, dones):
        val_sum = 0
        _, big_h, *_ = states.shape
        for k in range(1, big_h - 1):
            val = self.compute_rollout_value(
                rewards=rewards,
                states=states,
                dones=dones,
                k=k
            )
            val_sum = val_sum + self.lam**(k - 1) * val
        final_val = self.compute_rollout_value(
            rewards=rewards,
            states=states,
            dones=dones,
            k=big_h
        )
        return (1 - self.lam) * val_sum + self.lam**(big_h - 1) * final_val

    def update(self, horizon=20):
        # train agent
        current_state, _ = self.env.reset(batch_size=12)
        entropy_loss = 0
        for _ in range(horizon):
            action_dist = self.actor(current_state)
            entropy_loss = entropy_loss + action_dist.entropy()
            action = action_dist.rsample()
            next_state, *_ = self.env.step(action)
            current_state = next_state

        s, _, r, d = self.env.get_rollouts()
        critic_loss = self.critic_loss(s,r,d)
        policy_loss = self.policy_loss(state_samples=s)
        actor_loss = policy_loss + self.entropy_weight * entropy_loss.mean()
        self.critic_optim.backward(critic_loss, retain_graph=True)
        self.actor_optim.backward(actor_loss, retain_graph=False)
        self.critic_optim.update_parameters()
        self.actor_optim.update_parameters()
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy_loss": entropy_loss.mean().item(),
        }

    def policy_loss(self, state_samples):
        b, h, *l = state_samples.shape
        state_samples = state_samples.reshape(b*h, *l)
        return - self.critic(state_samples).mean()

    def critic_loss(
            self,
            state_samples,
            reward_samples,
            done_samples,
        ):
        state_values = self.critic(state_samples[:, 0])
        targets = self.compute_value_target(
            states=state_samples,
            rewards=reward_samples,
            dones=done_samples
        )
        loss = 0.5 * (targets.detach() - state_values)**2
        return loss.mean()

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
        checkpoint = torch.load(f'{path}/agent.pth')
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_optim.optimizer \
            .load_state_dict(checkpoint['actor_optim'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optim.optimizer \
            .load_state_dict(checkpoint['critic_optim'])
