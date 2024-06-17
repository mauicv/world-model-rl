import gymnasium as gym
from reflect.models.rl.value_trainer import ValueGradTrainer
from reflect.models.rl.actor import Actor
from reflect.models.rl.value_critic import ValueCritic
import torch
import pytest

@pytest.mark.parametrize("env_name,k", [
    ("InvertedPendulum-v4", 3),
    ("InvertedPendulum-v4", 14),
    ("Ant-v4", 3),
    ("Ant-v4", 14),
])
def test_compute_rollout_value(env_name, k):
    gym_env = gym.make(env_name)
    actor = Actor(
        input_dim=32*32,
        action_space=gym_env.action_space
    )
    critic = ValueCritic(
        state_dim=32*32,
    )
    trainer = ValueGradTrainer(
        actor=actor,
        critic=critic
    )

    states = torch.rand((3, 12, 32*32))
    rewards = torch.rand((3, 12, 1))
    dones = torch.zeros((3, 12, 1))

    rollout_value = trainer.compute_rollout_value(
        states=states,
        rewards=rewards,
        dones=dones,
        k=k
    )

    assert rollout_value.shape == (3, 1)


@pytest.mark.parametrize("env_name", [
    "InvertedPendulum-v4",
    "Ant-v4",
])
def test_compute_value_target(env_name):
    gym_env = gym.make(env_name)
    actor = Actor(
        input_dim=32*32,
        action_space=gym_env.action_space
    )
    critic = ValueCritic(
        state_dim=32*32,
    )
    trainer = ValueGradTrainer(
        actor=actor,
        critic=critic,
    )

    states = torch.rand((3, 12, 32*32))
    rewards = torch.rand((3, 12, 1))
    dones = torch.zeros((3, 12, 1))

    rollout_value = trainer.compute_value_target(
        states=states,
        rewards=rewards,
        dones=dones,
    )

    assert rollout_value.shape == (3, 1)


def test_update():
    gym_env = gym.make(
        'InvertedPendulum-v4'
    )
    actor = Actor(
        input_dim=32*32,
        action_space=gym_env.action_space
    )
    critic = ValueCritic(
        state_dim=32*32,
    )
    trainer = ValueGradTrainer(
        actor=actor,
        critic=critic,
    )

    states = torch.rand((3, 12, 32*32))
    rewards = torch.rand((3, 12, 1))
    dones = torch.zeros((3, 12, 1))

    history = trainer.update(
        state_samples=states,
        reward_samples=rewards,
        done_samples=dones,
    )

    assert 'critic_loss' in history
    assert 'actor_loss' in history