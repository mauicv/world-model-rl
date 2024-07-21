from reflect.data.differentiable_pendulum import DiffPendulumEnv
from reflect.models.rl.value_trainer import ValueGradTrainer, update_target_network
from reflect.models.rl.actor import Actor
from reflect.models.rl.value_critic import ValueCritic
import torch
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


def test_actor(actor, env, batch_size=10):
    with torch.no_grad():
        current_state, _ = env.reset(batch_size=batch_size)
        rewards = []
        for _ in range(100):
            action = actor(current_state).sample().squeeze(1)
            next_state, reward, _, _ = env.step(action)
            current_state = next_state
            rewards.append(reward.mean())
        return sum(rewards)


if __name__ == '__main__':
    batch_size = 32
    env = DiffPendulumEnv()
    env.set_seed(0)
    actor = Actor(
        input_dim=3,
        output_dim=1,
        hidden_dim=64,
        num_layers=3,
        bound=2
    )

    critic = ValueCritic(
        state_dim=3,
        hidden_dim=64,
        num_layers=3
    )

    trainer = ValueGradTrainer(
        actor=actor,
        critic=critic,
        actor_lr=0.0005,
        critic_lr=0.0001,
        env=None
    )
    logs = defaultdict(list)
    pbar = tqdm.tqdm(range(60_000 // batch_size))
    for _ in pbar:
        rewards = []
        dones = []
        states = []
        entropy = 0
        current_state, _ = env.reset(batch_size=batch_size)
        for _ in range(40):
            action_dist = trainer.actor(current_state)
            action = action_dist.rsample()
            entropy = entropy + action_dist.entropy()
            action = action.squeeze(1)
            states.append(current_state)
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            rewards.append(reward)
            dones.append(done)

        rewards = torch.stack(rewards, dim=1)
        states = torch.stack(states, dim=1)
        dones = torch.stack(dones, dim=1).to(torch.float32)

        action_loss = trainer.policy_loss(state_samples=states)
        actor_loss = action_loss - trainer.entropy_weight * entropy.mean()
        trainer.actor_optim.backward(actor_loss)
        trainer.actor_optim.update_parameters()

        critic_loss = trainer.critic_loss(
            states.detach(),
            rewards.detach(),
            dones.detach()
        )
        trainer.critic_optim.backward(critic_loss)
        trainer.critic_optim.update_parameters()
        update_target_network(trainer.target_critic, trainer.critic)

        reward = test_actor(actor, env, batch_size=10)

        logs["return"].append(actor_loss.detach().item())
        logs["reward"].append(reward)
        logs["last_reward"].append(rewards[:, -1].detach().mean().item())
        logs["action_loss"].append(action_loss.detach().mean().item())
        logs["critic_loss"].append(critic_loss.detach().mean().item())
        logs["entropy"].append(entropy.detach().mean().item())

        pbar.set_description(
            f"reward: {reward: 4.4f}, "
            f"last reward: {rewards[:, -1].mean(): 4.4f}"
        )

    fig, axs = plt.subplots(ncols=2, nrows=2)
    axs[0, 0].plot(logs["reward"])
    axs[0, 1].plot(logs["last_reward"])
    axs[1, 0].plot(logs["critic_loss"])
    axs[1, 1].plot(logs["action_loss"])
    axs[1, 1].legend(["action_loss"])
    axs[1, 0].legend(["critic_loss"])
    axs[0, 0].legend(["reward"])
    axs[0, 1].legend(["last_reward"])
    plt.show()