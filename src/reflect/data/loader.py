"""Data loader for the environment data.

See also: https://colab.research.google.com/drive/10-QQlnSFZeWBC7JCm0mPraGBPLVU2SnS
"""

import gymnasium as gym
import torch

num_time_steps=48


def to_tensor(t):
    if isinstance(t, torch.Tensor):
        return t
    return torch.tensor(t)


class EnvDataLoader:
    def __init__(
            self,
            num_time_steps,
            batch_size=64,
            num_runs=64,
            rollout_length=100,
            transforms=None,
            img_shape=(256, 256),
            policy=None,
            observation_model=None,
            env=gym.make(
                "InvertedPendulum-v4",
                render_mode="rgb_array"
            )
        ):
        self.env = env
        _ = self.env.reset()
        self.action_dim = self.env.action_space.shape[0]
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.rollout_length = rollout_length
        self.img_shape = img_shape
        self.observation_model = observation_model
        self.policy = policy

        self.rollout_ind = 0
        self.img_buffer = torch.zeros(
            (self.num_runs, self.rollout_length, *self.img_shape),
            dtype=torch.float32
        )

        self.action_buffer = torch.zeros(
            (self.num_runs, self.rollout_length, self.action_dim),
            dtype=torch.float32
        )

        self.reward_buffer = torch.zeros(
            (self.num_runs, self.rollout_length),
            dtype=torch.float32
        )

        self.done_buffer = torch.ones(
            (self.num_runs, self.rollout_length),
            dtype=torch.int
        )

        self.end_index = torch.zeros(
            self.num_runs,
            dtype=torch.int
        )

        self.current_index = 0
        self.transforms = transforms

    def perform_rollout(self, noise=0.5):
        _ = self.env.reset()
        img = self.env.render()
        img = self._preprocess(img)
        for index in range(self.rollout_length):
            action = self.compute_action(
                img[None, :],
                noise=noise
            )
            _, reward, done, *_ \
                = self.env.step(action.cpu().numpy())
            img = self.env.render()
            img = self._preprocess(img)
            run_index = self.rollout_ind % self.num_runs
            self.img_buffer[run_index, index] = img
            self.action_buffer[run_index, index] = to_tensor(action)
            # weird issue with pendulum environment always returns 1 reward
            if self.env.unwrapped.spec.id == "InvertedPendulum-v4":
                reward = -10 if done else 1
            self.reward_buffer[run_index, index] = to_tensor(reward)
            self.done_buffer[run_index, index] = to_tensor(done)
            if done and index > self.num_time_steps:
                break
        self.end_index[run_index] = index
        self.rollout_ind += 1

    def compute_action(self, observation, noise=0.5):
        device = next(self.observation_model.parameters()).device
        if self.policy:
            observation = observation.to(device)
            z = self.observation_model.encode(observation)
            z = z.view(1, -1)
            action = (
                self.policy
                    .compute_action(z, eps=noise)
                    .squeeze(0)
                    .detach()
                )
        else:
            action = self.env.action_space.sample()
            action = torch.tensor(action, device=device)
        return action

    def close(self):
        self.env.close()

    def _preprocess(self, x):
        x = to_tensor(x.copy())
        x = x.permute(2, 0, 1)
        x = self.transforms(x)
        x = x / 256
        return x

    def sample(self, batch_size=None, num_time_steps=None):
        if not batch_size:
            batch_size = self.batch_size
        if not num_time_steps:
            num_time_steps = self.num_time_steps

        max_index = min(self.rollout_ind, self.num_runs)
        b_inds = torch.randint(0, max_index, (batch_size, 1))
        end_inds = self.end_index[b_inds]
        t_inds = []
        for end_ind in end_inds:
            t_ind = torch.randint(0, (end_ind - self.num_time_steps), (1, ))
            t_inds.append(t_ind)
        t_inds = torch.cat(t_inds, dim=0)
        t_inds = t_inds[:, None] + torch.arange(0, num_time_steps)
        return (
            self.img_buffer[b_inds, t_inds],
            self.action_buffer[b_inds, t_inds],
            self.reward_buffer[b_inds, t_inds].unsqueeze(-1),
            self.done_buffer[b_inds, t_inds].unsqueeze(-1),
        )