
"""ReplayBuffer class

Uses circular memory storage to store states, actions, rewards and dones.
Implements sample that returns a batch of training samples for learning.
"""

import numpy as np
import random
REPLAY_BUFFER_SIZE=1000000
BATCH_SIZE=256


class ReplayBuffer:
    def __init__(
            self,
            action_space_dim,
            state_space_dim,
            size=REPLAY_BUFFER_SIZE,
            sample_size=BATCH_SIZE):
        self.states = np.zeros((size, state_space_dim), dtype='float32')
        self.next_states = np.zeros((size, state_space_dim), dtype='float32')
        self.actions = np.zeros((size, action_space_dim), dtype='float32')
        self.rewards = np.zeros(size, dtype='float32')
        self.dones = np.zeros(size, dtype='float32')
        self._end_index = 0
        self.sample_size = sample_size

    def push(self, step_data):
        (state, next_state, action, reward, done) = step_data
        index = self._end_index % len(self.states)
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self._end_index += 1

    @property
    def ready(self):
        return self._end_index >= self.sample_size

    @property
    def full(self):
        return self._end_index >= len(self.states)

    def sample(self):
        length = min(self._end_index, len(self.states))
        sample_size = min(self.sample_size, self._end_index + 1)
        random_integers = random.sample(range(length), sample_size)
        index_array = np.array(random_integers)
        state_samples = self.states[index_array]
        next_state_samples = self.next_states[index_array]
        action_samples = self.actions[index_array]
        reward_samples = self.rewards[index_array]
        done_samples = self.dones[index_array]
        return state_samples, next_state_samples, action_samples, \
            reward_samples, done_samples