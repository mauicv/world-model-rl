"""Data loader for the environment data.

See also: https://colab.research.google.com/drive/10-QQlnSFZeWBC7JCm0mPraGBPLVU2SnS
"""

import gymnasium as gym
from reflect.data.noise import NoNoise
from reflect.utils import FreezeParameters
from torchvision.transforms import Resize, Compose
import torch
import numpy as np
import math


def to_tensor(t):
    if isinstance(t, torch.Tensor):
        return t
    if isinstance(t, np.ndarray):
        return torch.tensor(t.copy(), dtype=torch.float32)
    return torch.tensor(t, dtype=torch.float32)


class Processing:
    def __init__(self, transforms):
        self.transforms = transforms

    def preprocess(self, x):
        raise NotImplementedError

    def postprocess(self, x):
        raise NotImplementedError


class GymRenderImgProcessing(Processing):
    def __init__(
            self,
            transforms=None
        ):
        if transforms is None:
            transforms = Compose([Resize((64, 64))])
        self.transforms = transforms

    def preprocess(self, x):
        x = x.permute(2, 0, 1)
        x = self.transforms(x)
        x = x / 256 - 0.5
        return x

    def postprocess(self, x):
        x = x.permute(1, 2, 0)
        x = (x + 0.5) * 256
        return x


class GymStateProcessing(Processing):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x


class SteeredEnvDataLoader:
    def __init__(
            self,
            num_time_steps,
            batch_size=64,
            num_runs=64,
            rollout_length=100,
            processing=None,
            state_shape=(3, 256, 256),
            policy=None,
            env=gym.make(
                "InvertedPendulum-v4",
                render_mode="rgb_array"
            ),
            noise_generator=None,
            seed=None,
            noise_size=0.05,
            weight_perturbation_size=0.01,
            use_imgs_as_states=True,
            steering_ratio=0.5,
            num_steering=32,
            max_steering_std=25
        ):

        if processing is None:
            if use_imgs_as_states:
                processing = GymRenderImgProcessing(transforms=None)
            else:
                processing = GymStateProcessing(transforms=None)

        self.processing = processing
        self.env = env
        self.seed = seed
        self.noise_size = noise_size
        self.use_imgs_as_states = use_imgs_as_states
        _ = self.env.reset()
        self.action_dim = self.env.action_space.shape[0]
        self.weight_perturbation_size = weight_perturbation_size
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
        self.state_shape = state_shape
        self.policy = policy
        self.noise_generator = noise_generator
        self.steering_ratio = steering_ratio

        self.rollout_ind = 0
        self.state_buffer = torch.zeros(
            (self.num_runs, self.rollout_length, *self.state_shape),
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

        self.reward_sums = torch.zeros(
            self.num_runs,
            dtype=torch.float32
        )

        self.current_index = 0

        self.num_steering = num_steering
        self.steering_labels = []
        self.steering_means = []
        self.dead_rollouts = set()
        self.max_steering_std = max_steering_std

        if noise_generator is None:
            self.noise_generator = NoNoise(dim=self.action_dim)

    def step(self, action):
        state, reward, done, *_ \
            = self.env.step(action.cpu().numpy())
        if self.use_imgs_as_states:
            state = self.env.render()
        state = to_tensor(state)
        state = self.processing.preprocess(state)
        return state, reward, done

    def reset(self):
        state, *_ = self.env.reset(seed=self.seed)
        if self.policy is not None:
            self.policy.reset()
            self.policy.perturb_actor(
                weight_perturbation_size=self.weight_perturbation_size
            )
        if self.use_imgs_as_states:
            state = self.env.render()
        state = to_tensor(state)
        state = self.processing.preprocess(state)
        return state

    def perform_rollout(self):
        """Performs a rollout of the environment.

        Iterate rollouts and store the images, actions, rewards, and done
        signals in the corresponding buffers.

        Note: we store (s_t, a_t, r_t, d_t) in the buffers. where _t denotes
        the time step. So a_t is the action taken at time step t not the action
        that generated s_t.
        """
        state = self.reset()
        done = False
        reward = 0
        run_index = self.rollout_ind % self.num_runs
        for index in range(self.rollout_length):
            action = self.compute_action(state[None, :])
            self.state_buffer[run_index, index] = state
            self.action_buffer[run_index, index] = to_tensor(action)
            # weird issue with pendulum environment always returns 1 reward
            if hasattr(self.env, 'unwrapped') and \
                    self.env.unwrapped.spec.id == "InvertedPendulum-v4":
                reward = -10 if done else 1
            self.reward_buffer[run_index, index] = to_tensor(reward)
            self.done_buffer[run_index, index] = to_tensor(done)
            state, reward, done = self.step(action)
            if done and index > self.num_time_steps:
                break
        self.reward_sums[run_index] = self.reward_buffer[run_index].sum()
        self.end_index[run_index] = index

        self._assign_steering_label(run_index, self.reward_sums[run_index])

        if run_index in self.dead_rollouts:
            self.dead_rollouts.remove(run_index)

        self.rollout_ind += 1

    def _assign_steering_label(self, run_index, reward_sum):
        # NOTE this is code for experiementation and in reality reduces information. In practice this
        # labeling should be done by a human. The key idea here is just that good rewards are selected
        # for the steering set.

        if len(self.steering_means) < 2:
            self.steering_means.append(reward_sum)
            self.steering_labels.append(run_index)
            return
        
        mean, std = self.get_steering_stats()
        std = min(std, self.max_steering_std)

        if reward_sum > mean - 0.5 * std:
            self.steering_labels.append(run_index)
            self.steering_means.append(reward_sum)

        if len(self.steering_labels) == self.num_steering:
            dead_index = self.steering_labels.pop(0)
            self.steering_means.pop(0)
            self.dead_rollouts.add(dead_index)

    def get_steering_stats(self):
        mean = sum(self.steering_means)/len(self.steering_means)
        std = math.sqrt(sum((x - mean)**2 for x in self.steering_means)/len(self.steering_means))
        return mean, std

    def compute_action(self, observation):
        if self.policy:
            action = self.policy(observation)
            action = action + torch.normal(torch.zeros_like(action), self.noise_size)
            action = action.squeeze()
        else:
            action = self.noise_generator()
            action = torch.tensor(action, device=observation.device)
        l, u = self.bounds
        l, u = l.to(action.device), u.to(action.device)
        action = torch.clamp(action, l, u)
        return action

    def close(self):
        self.env.close()

    def postprocess(self, x):
        return self.processing.postprocess(x)
    
    @property
    def steering_set_full(self):
        return len(self.steering_labels) >= self.num_steering
    
    def _sample_indices(self, batch_size, steering_ratio, max_index=None):
        n_steering = int(batch_size * steering_ratio)
        n_regular = batch_size - n_steering

        indices = []
        labels = []

        if n_steering > 0 and self.steering_labels:
            steering_indices = torch.tensor(list(set(self.steering_labels) - self.dead_rollouts))
            steering_samples = torch.randint(0, len(steering_indices), (n_steering,))
            indices.extend(steering_indices[steering_samples])
            labels.extend(torch.ones(n_steering))

        if n_regular > 0:
            available_indices = list(set(range(max_index)) - set(self.steering_labels) - self.dead_rollouts)
            if available_indices:
                regular_indices = torch.tensor(available_indices)
                regular_samples = torch.randint(0, len(regular_indices), (n_regular,))
                indices.extend(regular_indices[regular_samples])
                labels.extend(torch.zeros(n_regular))
        
        indices = torch.tensor(indices)
        labels = torch.tensor(labels)
        return indices.unsqueeze(1), labels.unsqueeze(1)

    def sample(
            self,
            batch_size=None,
            num_time_steps=None,
            from_start=False,
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
        b_inds, steering_labels = self._sample_indices(
            batch_size,
            steering_ratio=self.steering_ratio,
            max_index=max_index
        )

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
            self.state_buffer[b_inds, t_inds].detach(),
            self.action_buffer[b_inds, t_inds].detach(),
            steering_labels,
            self.done_buffer[b_inds, t_inds].unsqueeze(-1).detach(),
        )