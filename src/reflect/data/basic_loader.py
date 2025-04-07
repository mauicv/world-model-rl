"""Data loader for the environment data.

See also: https://colab.research.google.com/drive/10-QQlnSFZeWBC7JCm0mPraGBPLVU2SnS
"""

import gymnasium as gym
from reflect.data.noise import NoNoise
import torch
from reflect.data.processing import GymRenderImgProcessing, GymStateProcessing, to_tensor

class EnvDataLoader:
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
            priority_sampling_temperature=None
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
        self.priority_sampling_temperature = priority_sampling_temperature

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
        self.rollout_ind += 1

    def compute_action(self, observation):
        if self.policy:
            action = self.policy(observation)
            action = action + torch.normal(torch.zeros_like(action), self.noise_size)
            # action = action.squeeze(0)
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
    
    def _sample_indices(self, batch_size, temperature=None, max_index=None):
        rewards_tensor = torch.tensor(self.reward_sums, dtype=torch.float32)
        if temperature is None:
            indices = torch.randint(0, max_index, (batch_size, 1))
        else:
            probs = torch.softmax(rewards_tensor[:max_index] / temperature, dim=0)
            indices = torch.multinomial(probs, batch_size, replacement=True)
            indices = indices.unsqueeze(1)
        return indices

    def sample(
            self,
            batch_size=None,
            num_time_steps=None,
            from_start=False,
            use_priority_sampling=None
        ):
        """Sample a batch of data from the buffer.

        args:
            batch_size: int, optional
                The number of samples to return.
            num_time_steps: int, optional
                The number of time steps to sample.
            from_start: bool, optional
                If True, sample from the start of the rollout.
            use_priority_sampling: bool, optional
                Can be used to disable priority sampling if
                priority_sampling_temperature in __init__ is
                not None. Otherwise has no effect.
        """
        if not batch_size:
            batch_size = self.batch_size
        if not num_time_steps:
            num_time_steps = self.num_time_steps

        if use_priority_sampling is None:
            use_priority_sampling = self.priority_sampling_temperature is not None

        max_index = min(self.rollout_ind, self.num_runs)
        priority_sampling_temperature = self.priority_sampling_temperature \
            if use_priority_sampling else None
        b_inds = self._sample_indices(
            batch_size,
            temperature=priority_sampling_temperature,
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
            self.reward_buffer[b_inds, t_inds].unsqueeze(-1).detach(),
            self.done_buffer[b_inds, t_inds].unsqueeze(-1).detach(),
        )