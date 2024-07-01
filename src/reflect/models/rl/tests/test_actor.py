import pytest
from reflect.models.rl.actor import Actor
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
    actor = Actor(
        input_dim=32*32,
        action_space=gym_env.action_space
    )
    input = torch.rand((batch_size, 32*32))
    action = actor(input)

    assert action.shape == (batch_size, gym_env.action_space.shape[0])
    assert torch.all(action >= torch.tensor(gym_env.action_space.low))
    assert torch.all(action <= torch.tensor(gym_env.action_space.high))


@pytest.mark.parametrize("env_name,batch_size", [
    ("InvertedPendulum-v4", 1),
    ("InvertedPendulum-v4", 5),
    ("Ant-v4", 1),
    ("Ant-v4", 5),
])
def test_actor_compute_action(env_name, batch_size):
    gym_env = gym.make(env_name)
    actor = Actor(
        input_dim=32*32,
        action_space=gym_env.action_space
    )
    input = torch.rand((batch_size, 32*32))
    action = actor.compute_action(input)

    assert action.shape == (batch_size, gym_env.action_space.shape[0])
    assert torch.all(action >= torch.tensor(gym_env.action_space.low))
    assert torch.all(action <= torch.tensor(gym_env.action_space.high))


@pytest.mark.parametrize("env_name,batch_size", [
    ("InvertedPendulum-v4", 1),
    ("InvertedPendulum-v4", 5),
    ("Ant-v4", 1),
    ("Ant-v4", 5),
])
def test_actor_stochastic(env_name, batch_size):
    gym_env = gym.make(env_name)
    actor = Actor(
        input_dim=32*32,
        action_space=gym_env.action_space,
        stochastic=True
    )
    input = torch.rand((batch_size, 32*32))
    a_dist, action = actor(input)
    assert a_dist.mean.shape == (batch_size, gym_env.action_space.shape[0])
    assert action.shape == (batch_size, gym_env.action_space.shape[0])
    assert torch.all(action >= torch.tensor(gym_env.action_space.low))
    assert torch.all(action <= torch.tensor(gym_env.action_space.high))