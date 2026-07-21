"""Data loader for the environment data.

See also: https://colab.research.google.com/drive/10-QQlnSFZeWBC7JCm0mPraGBPLVU2SnS
"""

import gymnasium as gym
from reflect.data.noise import NormalNoise
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
            weight_perturbation_size=0.01,
            use_imgs_as_states=True,
            priority_sampling_temperature=None,
            use_custom_priorities=False
        ):

        if processing is None:
            if use_imgs_as_states:
                processing = GymRenderImgProcessing(transforms=None)
            else:
                processing = GymStateProcessing(transforms=None)

        self.processing = processing
        self.env = env
        self.seed = seed
        self.use_imgs_as_states = use_imgs_as_states
        _ = self.env.reset(seed=seed)
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
        self.use_custom_priorities = use_custom_priorities

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

        self.priorities = torch.ones(
            self.num_runs,
            self.rollout_length,
            dtype=torch.float32
        )

        self.current_index = 0

        if noise_generator is None:
            self.noise_generator = NormalNoise(
                dim=self.action_dim,
                sigma=0.2,
                dt=1e-2,
                repeat=1
            )

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items() if k != 'env' and k != 'policy'
        }
        
    def __setstate__(self, state):
        self.__dict__.update(state)

    def step(self, action):
        state, reward, done, *_ \
            = self.env.step(action.cpu().numpy())
        if self.use_imgs_as_states:
            state = self.env.render()
        state = to_tensor(state)
        state = self.processing.preprocess(state)
        return state, reward, done

    def reset(self, seed=None):
        state, *_ = self.env.reset(seed=seed)
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
    
    def update_priorities(self, b_inds, t_inds, values):
        self.priorities[b_inds, t_inds] = values

    def _step_rollout(self, run_index, index, state, reward, done):
        action = self.compute_action(state[None, :])
        self.state_buffer[run_index, index] = state
        self.action_buffer[run_index, index] = to_tensor(action)
        self.reward_buffer[run_index, index] = to_tensor(reward)
        self.done_buffer[run_index, index] = to_tensor(done)
        state, reward, done = self.step(action)
        return state, reward, done

    def perform_rollout(self, seed=None):
        """Performs a rollout of the environment.

        Iterate rollouts and store the images, actions, rewards, and done
        signals in the corresponding buffers.

        Note: we store (s_t, a_t, r_t, d_t) in the buffers. where _t denotes
        the time step. So a_t is the action taken at time step t not the action
        that generated s_t.
        """
        state = self.reset(seed=seed)
        run_index = self.rollout_ind % self.num_runs

        # take first step of rollout as it is not a valid step
        action = self.compute_action(state[None, :])
        state, reward, done = self.step(action)
        
        # start recording rollout after first step
        for index in range(self.rollout_length):
            state, reward, done = self._step_rollout(run_index, index, state, reward, done)
            if done and index > self.num_time_steps:
                if index + 1 < self.rollout_length:
                    state, reward, done = self._step_rollout(run_index, index + 1, state, reward, done)
                    index += 1
                break
        self.reward_sums[run_index] = self.reward_buffer[run_index].sum()
        self.priorities[run_index] = torch.ones(self.rollout_length)
        self.priorities[run_index, index-self.num_time_steps:] = 0
        self.end_index[run_index] = index
        self.rollout_ind += 1

    def compute_action(self, observation):
        if self.policy:
            action = self.policy(observation)
            noise = self.noise_generator()
            action = action + torch.tensor(noise, device=action.device)
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
    
    def _sample_batch_indices(
            self,
            batch_size,
            temperature=None,
            max_index=None,
        ):

        if temperature is None:
            indices = torch.randint(0, max_index, (batch_size, 1))
        else:
            if self.use_custom_priorities:
                priorities = torch.tensor(self.priorities, dtype=torch.float32).sum(dim=1)
            else:
                priorities = torch.tensor(self.reward_sums, dtype=torch.float32)
            probs = torch.softmax(priorities[:max_index] / temperature, dim=0)
            indices = torch.multinomial(probs, batch_size, replacement=True)
            indices = indices.unsqueeze(1)
        return indices
    
    def _sample_step_indices(
            self,
            b_inds,
            num_time_steps,
            temperature=None,
            sample_offset=0,
        ):
        assert sample_offset < num_time_steps, f'{sample_offset=}, {num_time_steps=}'
        end_inds = self.end_index[b_inds] + 2 # +2 to account for the first step of the rollout
        t_inds = []
        if self.use_custom_priorities:
            priorities = torch.tensor(self.priorities[b_inds[:, 0]], dtype=torch.float32)
            for i, end_ind in enumerate(end_inds):
                probs = torch.softmax(priorities[i, sample_offset:end_ind - (num_time_steps - sample_offset)] / temperature, dim=0)
                t_ind = torch.multinomial(probs, 1, replacement=True)
                t_inds.append(t_ind - sample_offset)
        else:
            for end_ind in end_inds:
                t_ind = torch.randint(0, (end_ind - num_time_steps), (1, ))
                t_inds.append(t_ind)
        t_inds = torch.cat(t_inds, dim=0)
        return t_inds

    def sample(
            self,
            batch_size=None,
            num_time_steps=None,
            use_priority_sampling=None,
            sample_offset=0,
        ):
        """Sample a batch of data from the buffer.

        args:
            batch_size: int, optional
                The number of samples to return.
            num_time_steps: int, optional
                The number of time steps to sample.
            use_priority_sampling: bool, optional
                Can be used to disable priority sampling if
                priority_sampling_temperature in __init__ is
                not None. Otherwise has no effect.
            sample_offset: int, optional
                The number of time steps to offset the sample 
                by. For use with priority sampling to choose
                portions of the rollout that contain the most
                information.
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
        b_inds = self._sample_batch_indices(
            batch_size,
            temperature=priority_sampling_temperature,
            max_index=max_index
        )
        t_inds = self._sample_step_indices(
            b_inds,
            num_time_steps,
            temperature=priority_sampling_temperature,
            sample_offset=sample_offset,
        )
        t_inds = t_inds[:, None] + torch.arange(0, num_time_steps)
        return (
            b_inds, t_inds,
            self.state_buffer[b_inds, t_inds].detach(),
            self.action_buffer[b_inds, t_inds].detach(),
            self.reward_buffer[b_inds, t_inds].unsqueeze(-1).detach(),
            self.done_buffer[b_inds, t_inds].unsqueeze(-1).detach(),
        )

    def sample_pairs(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample consecutive state pairs (s_i, s_{i+1}) from the buffer.

        Returns:
            Tuple of (s_i, s_next), each of shape (batch_size, *state_shape).
        """
        max_index = min(self.rollout_ind, self.num_runs)
        b_inds = torch.randint(0, max_index, (batch_size,))
        t_inds = torch.randint(0, self.rollout_length - 1, (batch_size,))
        s_i = self.state_buffer[b_inds, t_inds].detach()
        s_next = self.state_buffer[b_inds, t_inds + 1].detach()
        return s_i, s_next