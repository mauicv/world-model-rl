import pytest
import gymnasium as gym
from shimmy.registration import DM_CONTROL_SUITE_ENVS
import random
import numpy as np
import torch
from reflect.components.trainers.td3.replay_buffer import ReplayBuffer
from reflect.components.trainers.td3.actor import TD3Actor
from reflect.components.trainers.td3.critic import TD3Critic
from reflect.components.trainers.td3.td3_trainer import TD3Trainer


seed=2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

@pytest.fixture
def env():
    env = gym.make(
        'dm_control/walker-walk-v0',
        render_mode="rgb_array",
        render_kwargs={'camera_id': 0}
    )
    env.np_random.seed(seed)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)
    return env

@pytest.fixture
def replay_buffer(env):
    return ReplayBuffer(
        action_space_dim=env.action_space.shape[0],
        state_space_dim=env.observation_space.shape[0],
        size=5000,
        sample_size=10,
    )

@pytest.fixture
def trainer(env):
    actor = TD3Actor(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.shape[0],
        )
    critic_1 = TD3Critic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    critic_2 = TD3Critic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )
    return TD3Trainer(
        actor=actor,
        critics=[critic_1, critic_2],
    )
