# tests each of the training routine updates are independent of each other.
import torch
import random
import numpy as np
import os

seed = 42

# Set seeds
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Ensure deterministic algorithms
torch.use_deterministic_algorithms(True)

# Control backend settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(seed)

# Optional: Warn about non-deterministic operations
torch.set_deterministic_debug_mode('warn')

import gymnasium as gym
from reflect.models.rl.value_trainer import ValueGradTrainer
from reflect.models.rl.actor import Actor
from reflect.models.rl.value_critic import ValueCritic
from conftest import make_dynamic_model
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model import WorldModel
from reflect.models.world_model.environment import Environment
from torchvision.transforms import Resize, Compose
import pytest


import torch.nn as nn

def initialize_weights_large(module):
    """
    Initialize the weights of a given module with large values.
    Args:
        module (nn.Module): The module to initialize.
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=1.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@pytest.mark.parametrize("env_name", [
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

    trainer.actor_optim.optimizer.zero_grad()

    initial_state, _ = trainer.env.reset(batch_size=batch_size)
    current_state = initial_state.detach().clone()
    trainer.actor.reset()

    for _ in range(3):
        action = trainer.actor(current_state, deterministic=True)
        env.world_model.requires_grad_(False)
        next_state, *_ = trainer.env.step(action)
        env.world_model.requires_grad_(True)
        current_state = next_state

    s, _, r, d = trainer.env.get_rollouts()
    actor_loss = trainer.policy_loss(state_samples=s)
    actor_loss.backward()

    for name, param in trainer.actor.named_parameters():
        print(name, None if param.grad is None else param.grad.norm().item())
