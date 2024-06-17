import pytest
from reflect.models.rl.value_critic import ValueCritic
import gymnasium as gym
import torch


@pytest.mark.parametrize("batch_size", [1, 5])
def test_value_critic(batch_size):
    actor = ValueCritic(
        state_dim=32*32,
    )
    state_input = torch.rand((batch_size, 32*32))
    q_value = actor(state_input)
    assert q_value.shape == (batch_size, 1)