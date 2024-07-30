from reflect.models.rl.actor import Actor
from reflect.models.rl.critic import Critic
from reflect.models.world_model.environment import Environment
from reflect.utils import FreezeParameters
from reflect.utils import AdamOptim
import torch
import copy

def update_target_network(target_model, model, tau=5e-3):
    with FreezeParameters([target_model, model]):
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
            actor,
            critic,
            env=None,
            actor_lr: float=0.001,
            critic_lr: float=0.001,
            grad_clip: float=0.5,
            weight_decay: float=1e-4,
            gamma: float=0.99,
            lam: float=0.95,
            entropy_weight: float=1e-4,
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
            * target_state_values[:, [h]].detach()
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

    def policy_loss(
            self,
            target_values
        ):
        return - target_values.mean()
        
    def critic_loss(
            self,
            state_samples,
            target_values,
        ):
        state_sample_values = self.critic(state_samples)
        loss = 0.5 * (target_values.detach() - state_sample_values)**2
        return loss.mean()

    def compute_values(
            self,
            state_samples,
            reward_samples,
            done_samples,
        ):
        b, h, *l = state_samples.shape
        target_state_values = self.target_critic(state_samples).detach()
        targets = []
        for i in range(h):
            rollout_targets = self.compute_value_target(
                target_state_values=target_state_values[:, i:, :],
                states=state_samples[:, i:, :],
                rewards=reward_samples[:, i:, :],
                dones=done_samples[:, i:, :],
            )
            targets.append(rollout_targets)
        target_values = torch.stack(targets, dim=1)
        return target_values

    def update(self, horizon=20, batch_size=12):
        current_state, _ = self.env.reset(batch_size=batch_size)
        self.actor.reset()
        entropy = 0
        for _ in range(horizon):
            action_dist = self.actor(current_state)
            action = action_dist.rsample()
            with FreezeParameters([self.env.world_model]):
                next_state, *_ = self.env.step(action)
            entropy = entropy + action_dist.entropy().mean()
            current_state = next_state

        states, _, rewards, dones = self.env.get_rollouts()
        target_values = self.compute_values(
            state_samples=states,
            reward_samples=rewards,
            done_samples=dones
        )

        policy_loss = self.policy_loss(target_values)
        actor_loss = policy_loss - self.entropy_weight * entropy
        self.actor_optim.backward(actor_loss, retain_graph=True)

        critic_loss = self.critic_loss(
            state_samples=states,
            target_values=target_values
        )
        self.critic_optim.backward(critic_loss)
        self.actor_optim.update_parameters()
        self.critic_optim.update_parameters()

        update_target_network(self.target_critic, self.critic)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "entropy_loss": entropy.item(),
        }

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
        self.target_critic = copy.deepcopy(self.critic)

