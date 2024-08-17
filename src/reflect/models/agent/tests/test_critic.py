import pytest
from reflect.models.agent.critic import Critic
import gymnasium as gym
import torch


@pytest.mark.parametrize("env_name,batch_size", [
    ("InvertedPendulum-v4", 1),
    ("InvertedPendulum-v4", 5),
    ("Ant-v4", 1),
    ("Ant-v4", 5),
])
def test_actor(env_name, batch_size):
    gym_env = gym.make(env_name)
    actor = Critic(
        state_dim=32*32,
        action_space=gym_env.action_space
    )
    state_input = torch.rand((batch_size, 32*32))
    action_input = torch.rand((batch_size, gym_env.action_space.shape[0]))
    q_value = actor(state_input, action_input)
    assert q_value.shape == (batch_size, 1)