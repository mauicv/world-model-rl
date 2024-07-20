from reflect.data.differentiable_pendulum import DiffPendulumEnv
from reflect.models.rl.reward_trainer import RewardGradTrainer
import torch.nn as nn
import torch
import tqdm
from collections import defaultdict


class Policy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(64),
            nn.Tanh(),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    batch_size = 32
    policy = Policy()
    env = DiffPendulumEnv()
    trainer = RewardGradTrainer(actor=policy)
    env.set_seed(0)
    pbar = tqdm.tqdm(range(20_000 // batch_size))
    logs = defaultdict(list)
    for _ in pbar:
        current_state, _ = env.reset(batch_size=batch_size)
        rewards = []
        dones = []
        for _ in range(100):
            action = policy(current_state).squeeze(1)
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            rewards.append(reward)
            dones.append(done)

        rewards = torch.cat(rewards, dim=1)
        dones = torch.cat(dones, dim=1).to(torch.float32)

        actor_loss = trainer.update(rewards, dones)['actor_loss']
        
        pbar.set_description(
            f"reward: {actor_loss: 4.4f}, "
            f"last reward: {rewards[:, -1].mean(): 4.4f}"
        )
        logs["return"].append(actor_loss)
        logs["last_reward"].append(rewards[:, -1].mean().item())
        # scheduler.step()

    def plot():
        import matplotlib
        from matplotlib import pyplot as plt

        is_ipython = "inline" in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        with plt.ion():
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(logs["return"])
            plt.title("returns")
            plt.xlabel("iteration")
            plt.subplot(1, 2, 2)
            plt.plot(logs["last_reward"])
            plt.title("last reward")
            plt.xlabel("iteration")
            if is_ipython:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            plt.show()

    plot()