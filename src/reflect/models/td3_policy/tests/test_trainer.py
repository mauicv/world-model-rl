import gymnasium as gym
from reflect.models.td3_policy.trainer import AgentTrainer
import torch
import pytest


@pytest.mark.parametrize("environment", [
    "Pendulum-v1",
    "BipedalWalker-v3",
])
def test_agent_trainer_update_actor(environment):
    env = gym.make(environment)
    trainer = AgentTrainer(
        state_dim=32,
        action_space=env.action_space,
        actor_lr=0.001,
        critic_lr=0.001
    )
    trainer.update_actor(
        states=torch.tensor([[1.0]*32]),
    )


@pytest.mark.parametrize("environment", [
    "Pendulum-v1",
    "BipedalWalker-v3",
])
def test_agent_trainer_update_critic(environment):
    env = gym.make(environment)
    trainer = AgentTrainer(
        state_dim=32,
        action_space=env.action_space,
        actor_lr=0.001,
        critic_lr=0.001
    )
    trainer.update_critic(
        current_states=torch.tensor([[1.0]*32]),
        next_states=torch.tensor([[1.0]*32]),
        current_actions=torch.tensor([[1.0]*env.action_space.shape[0]]),
        rewards=torch.tensor([1.0]*32),
        dones=torch.tensor([1.0]*32),
    )


@pytest.mark.parametrize("environment", [
    "Pendulum-v1",
    "BipedalWalker-v3",
])
def test_agent_trainer_update_critic_target_network(environment):
    # TODO: should check correct difference between target and model
    env = gym.make(environment)
    trainer = AgentTrainer(
        state_dim=32,
        action_space=env.action_space,
        actor_lr=0.001,
        critic_lr=0.001
    )
    trainer.update_critic_target_network()


@pytest.mark.parametrize("environment", [
    "Pendulum-v1",
    "BipedalWalker-v3",
])
def test_agent_trainer_update_actor_target_network(environment):
    # TODO: should check correct difference between target and model
    env = gym.make(environment)
    trainer = AgentTrainer(
        state_dim=32,
        action_space=env.action_space,
        actor_lr=0.001,
        critic_lr=0.001
    )
    trainer.update_actor_target_network()