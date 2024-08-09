"""Data loader for the environment data.

See also: https://colab.research.google.com/drive/10-QQlnSFZeWBC7JCm0mPraGBPLVU2SnS
"""

import gymnasium as gym
from reflect.data.noise import NoNoise
from reflect.utils import FreezeParameters
import torch

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
            env=gym.make(
                "InvertedPendulum-v4",
                render_mode="rgb_array"
            ),
            noise_generator=None
        ):
        self.env = env
        _ = self.env.reset()
        self.action_dim = self.env.action_space.shape[0]
        self.bounds = (
            torch.tensor(
                self.env.action_space.low,
                dtype=torch.float32
            ),
            torch.tensor(
                self.env.action_space.high,
                dtype=torch.float32
            )
        )
        self.num_time_steps = num_time_steps
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.rollout_length = rollout_length
        self.img_shape = img_shape
        self.policy = policy
        self.noise_generator = noise_generator

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

        if noise_generator is None:
            self.noise_generator = NoNoise(dim=self.action_dim)

    def perform_rollout(self):
        """Performs a rollout of the environment.

        Iterate rollouts and store the images, actions, rewards, and done
        signals in the corresponding buffers.

        Note: we store (s_t, a_t, r_t, d_t) in the buffers. where _t denotes
        the time step. So a_t is the action taken at time step t not the action
        that generated s_t.
        """
        _ = self.env.reset()
        if self.policy is not None:
            self.policy.reset()
        img = self.env.render()
        img = self._preprocess(img)
        done = False
        reward = 0
        run_index = self.rollout_ind % self.num_runs
        for index in range(self.rollout_length):
            action = self.compute_action(img[None, :])
            self.img_buffer[run_index, index] = img
            self.action_buffer[run_index, index] = to_tensor(action)
            # weird issue with pendulum environment always returns 1 reward
            if hasattr(self.env, 'unwrapped') and \
                    self.env.unwrapped.spec.id == "InvertedPendulum-v4":
                reward = -10 if done else 1
            self.reward_buffer[run_index, index] = to_tensor(reward)
            self.done_buffer[run_index, index] = to_tensor(done)
            _, reward, done, *_ \
                = self.env.step(action.cpu().numpy())
            img = self.env.render()
            img = self._preprocess(img)
            if done and index > self.num_time_steps:
                break
        self.end_index[run_index] = index
        self.rollout_ind += 1

    def compute_action(self, observation):
        if self.policy:
            action = self.policy(observation)
            action = action + torch.normal(torch.zeros_like(action), 0.3)
            action = action.squeeze(0)
        else:
            action = self.noise_generator()
            action = torch.tensor(action, device=observation.device)
        l, u = self.bounds
        l, u = l.to(action.device), u.to(action.device)
        action = torch.clamp(action, l, u)
        return action

    def close(self):
        self.env.close()

    def _preprocess(self, x):
        x = to_tensor(x.copy())
        x = x.permute(2, 0, 1)
        x = self.transforms(x)
        x = x / 256 - 0.5
        return x

    def postprocess(self, x):
        x = x.permute(1, 2, 0)
        x = (x + 0.5) * 256
        return x

    def sample(
            self,
            batch_size=None,
            num_time_steps=None,
            from_start=False
        ):
        """Sample a batch of data from the buffer.

        args:
            batch_size: int, optional
                The number of samples to return.
            num_time_steps: int, optional
                The number of time steps to sample.
            from_start: bool, optional
                If True, sample from the start of the rollout.
        """
        if not batch_size:
            batch_size = self.batch_size
        if not num_time_steps:
            num_time_steps = self.num_time_steps

        max_index = min(self.rollout_ind, self.num_runs)
        b_inds = torch.randint(0, max_index, (batch_size, 1))
        end_inds = self.end_index[b_inds]
        t_inds = []
        if from_start:
            t_inds = torch.zeros(batch_size, dtype=torch.int)
        else:
            for end_ind in end_inds:
                t_ind = torch.randint(0, (end_ind - num_time_steps), (1, ))
                t_inds.append(t_ind)
            t_inds = torch.cat(t_inds, dim=0)
        t_inds = t_inds[:, None] + torch.arange(0, num_time_steps)
        return (
            self.img_buffer[b_inds, t_inds].detach(),
            self.action_buffer[b_inds, t_inds].detach(),
            self.reward_buffer[b_inds, t_inds].unsqueeze(-1).detach(),
            self.done_buffer[b_inds, t_inds].unsqueeze(-1).detach(),
        )