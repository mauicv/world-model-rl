from reflect.data.differentiable_pendulum import DiffPendulumEnv
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
    env.set_seed(0)
    optim = torch.optim.Adam(policy.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20_000)
    pbar = tqdm.tqdm(range(20_000 // batch_size))
    logs = defaultdict(list)
    for _ in pbar:
        current_state, _ = env.reset(batch_size=batch_size)
        rewards = []
        for _ in range(100):
            action = policy(current_state).squeeze(1)
            next_state, reward, done, _ = env.step(action)
            current_state = next_state
            rewards.append(reward)

        rewards = torch.cat(rewards, dim=1)
        traj_return = (-rewards).mean()
        traj_return.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        pbar.set_description(
            f"reward: {traj_return: 4.4f}, "
            f"last reward: {rewards[:, -1].mean(): 4.4f}, gradient norm: {gn: 4.4}"
        )
        logs["return"].append(traj_return.item())
        logs["last_reward"].append(rewards[:, -1].mean().item())
        scheduler.step()

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
