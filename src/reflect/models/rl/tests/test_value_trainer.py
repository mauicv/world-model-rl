import gymnasium as gym
from reflect.models.rl.value_trainer import ValueGradTrainer
from reflect.models.rl.actor import Actor
from reflect.models.rl.value_critic import ValueCritic
from conftest import make_dynamic_model
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model import WorldModel
from reflect.models.world_model.environment import Environment
from torchvision.transforms import Resize, Compose
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
        critic=critic,
        env=None
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
        env=None
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


@pytest.mark.parametrize("env_name", [
    "InvertedPendulum-v4",
    "Ant-v4",
])
def test_update(env_name, observation_model):
    batch_size=10
    real_env = gym.make(env_name, render_mode="rgb_array")
    action_size = real_env.action_space.shape[0]

    dm = make_dynamic_model(a_size=action_size)

    wm = WorldModel(
        observation_model=observation_model,
        dynamic_model=dm,
        num_ts=16,
    )

    dl = EnvDataLoader(
        num_time_steps=17,
        img_shape=(3, 64, 64),
        transforms=Compose([
            Resize((64, 64))
        ]),
        observation_model=observation_model,
        env=real_env
    )

    dl.perform_rollout()

    env = Environment(
        world_model=wm,
        data_loader=dl,
        batch_size=batch_size
    )
    actor = Actor(
        input_dim=32*32,
        action_space=real_env.action_space
    )
    critic = ValueCritic(
        state_dim=32*32,
    )
    trainer = ValueGradTrainer(
        actor=actor,
        critic=critic,
        env=env
    )

    history = trainer.update()
    assert 'critic_loss' in history
    assert 'actor_loss' in history