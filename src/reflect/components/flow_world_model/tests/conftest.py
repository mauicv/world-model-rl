import pytest
import gymnasium as gym
from reflect.components.flow_world_model.dynamic_model import DynamicFlowModel, DynamicAttentionalFlowModel
from reflect.components.flow_world_model.world_model_actor import WorldModelActor
from reflect.components.flow_world_model.world_model import WorldModel
from reflect.data.loader import EnvDataLoader
from reflect.components.models.actor import Actor

import torch
from torchvision.transforms import Resize, Compose

@pytest.fixture
def env():
    return gym.make("InvertedPendulum-v4", render_mode="rgb_array")

@pytest.fixture
def actor(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return Actor(
        input_dim=state_dim,
        output_dim=action_dim,
        bound=env.action_space.high,
        num_layers=3,
        hidden_dim=256,
    )

@pytest.fixture
def world_model_actor(actor):
    return WorldModelActor(
        actor=actor,
    )


@pytest.fixture
def env_data_loader(world_model_actor, env):
    state_dim = env.observation_space.shape[0]
    return EnvDataLoader(
        num_time_steps=10,
        state_shape=(state_dim,),
        policy=world_model_actor,
        env=env,
        use_imgs_as_states=False,
    )


@pytest.fixture
def dynamic_model(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return DynamicFlowModel(
        input_dim=state_dim,
        conditioning_dim=3*(state_dim + action_dim),
        output_dim=state_dim,
        time_embed_dim=16,
        hidden_dim=128,
        depth=4,
        use_layer_norm=True,
        num_positions=3,
    )

@pytest.fixture
def dynamic_attentional_model(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return DynamicAttentionalFlowModel(
        input_dim=state_dim,
        conditioning_dim=state_dim + action_dim,
        output_dim=state_dim,
        hidden_dim=128,
        num_heads=2,
        num_positions=3,
        depth=4,
        use_layer_norm=True,
    )


@pytest.fixture
def world_model(dynamic_model, env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return WorldModel(
        dynamic_model=dynamic_model,
        observation_dim=state_dim,
        action_dim=action_dim,
        environment_action_bound=env.action_space.high,
    )
