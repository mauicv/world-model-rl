import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from reflect.models.nes_policy.model import NESPolicy, rank_transform


class GymEnvs:
    def __init__(self, count=10) -> None:
        self.count = count
        self.envs = [gym.make('CartPole-v1', render_mode='rgb_array') for _ in range(count)]
        self.terminated = [False for _ in range(count)]
        self.rewards = [0 for _ in range(count)]

    def reset(self):
        self.terminated = [False for _ in range(self.count)]
        self.rewards = [0 for _ in range(self.count)]

        int_obs = []
        for env in self.envs:
            o, _ = env.reset()
            int_obs.append(torch.tensor(o, dtype=torch.float32).unsqueeze(0))
        return torch.cat(int_obs, dim=0)

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = map(lambda x: int(x), actions.tolist())
        return torch.cat([
            self._step_env(ind, env, action)
            for ind, (env, action) in enumerate(zip(self.envs, actions))
        ], dim=0)

    def _step_env(self, env_ind, env, action):
        o, r, d, *_ = env.step(action)
        self.rewards[env_ind] += r
        self.terminated[env_ind] = d
        return torch.tensor(o, dtype=torch.float32) \
            .unsqueeze(0)

    def sample_actions(self):
        return torch.tensor([
            env.action_space.sample()
            for env in self.envs
        ], dtype=torch.float32)


def compute_rewards(envs: GymEnvs, policy: NESPolicy):
    states = envs.reset()
    for _ in range(100):
        actions = policy(states).argmax(dim=1)
        states = envs.step(actions)
    return torch.tensor(envs.rewards)


def train_step(envs: GymEnvs, policy: NESPolicy):
    policy.perturb(eps=0.05)
    rewards = compute_rewards(envs, policy)
    avg_reward = rewards.mean()
    scores = rank_transform(rewards)
    grads = policy.compute_grads(scores, eps=0.05)
    policy.update(grads)
    return avg_reward


def play(policy: NESPolicy):
    env = gym.make('CartPole-v1', render_mode='human')
    state_fn = lambda o: torch.tensor(o, dtype=torch.float32).unsqueeze(0)
    action_fn = lambda s: policy(s, training=False).argmax(dim=1).item()
    o, *_ = env.reset()
    state = state_fn(o)

    for _ in range(250):
        action = action_fn(state)
        o, *_ = env.step(action)
        state = state_fn(o)
        env.render()
    env.close()


if __name__ == '__main__':
    envs = GymEnvs(count=100)
    policy = NESPolicy(
        input_dim=4,
        output_dim=2,
        hidden_dims=[32, 32],
        population_size=100,
    )
    hist = []
    for i in range(100):
        avg_reward = train_step(envs, policy)
        print('run=',i,'average reward=',avg_reward.item())
        hist.append(avg_reward.item())

    plt.plot(hist)
    plt.show()

    play(policy)
