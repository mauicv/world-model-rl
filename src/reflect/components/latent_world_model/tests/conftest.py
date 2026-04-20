import pytest
import gymnasium as gym
from reflect.data.loader import EnvDataLoader, GymRenderImgProcessing

from reflect.components.latent_world_model.models import MLPDynamicModel, MLPEncoder
from reflect.components.latent_world_model.models.encoder_actor import EncoderActor
from reflect.components.latent_world_model.models.actor import MLPActor

import torch
from torchvision.transforms import Resize, Compose


LATENT_DIM = 512


@pytest.fixture
def encoder(env_bipedal_walker):
    input_dim = env_bipedal_walker.observation_space.shape[0]
    return MLPEncoder(
        input_dim=input_dim,
        output_dim=LATENT_DIM,
        num_layers=3,
        hidden_dim=512,
    )

@pytest.fixture
def discrete_encoder(env_bipedal_walker):
    input_dim = env_bipedal_walker.observation_space.shape[0]
    return MLPEncoder(
        input_dim=input_dim,
        output_dim=512,
        num_layers=3,
        hidden_dim=512,
        output_activation=torch.nn.Tanh,
    )

@pytest.fixture
def dynamic_model(env_bipedal_walker):
    action_dim = env_bipedal_walker.action_space.shape[0]
    return MLPDynamicModel(
        latent_dim=LATENT_DIM,
        action_dim=action_dim,
        num_layers=3,
        hidden_dim=512,
    )

@pytest.fixture
def discrete_dynamic_model(env_bipedal_walker):
    action_dim = env_bipedal_walker.action_space.shape[0]
    return MLPDynamicModel(
        latent_dim=LATENT_DIM,
        action_dim=action_dim,
        num_layers=3,
        hidden_dim=512,
        output_dim=256*64,
    )

@pytest.fixture
def actor(env_bipedal_walker):
    action_dim = env_bipedal_walker.action_space.shape[0]
    return MLPActor(
        latent_dim=LATENT_DIM,
        action_dim=action_dim,
        num_layers=3,
        hidden_dim=512,
    )

@pytest.fixture
def encoder_actor(encoder, actor):
    return EncoderActor(
        encoder=encoder,
        actor=actor,
        latent_dim=LATENT_DIM,
    )


@pytest.fixture
def env_bipedal_walker():
    return gym.make("BipedalWalker-v3", render_mode="rgb_array")

@pytest.fixture
def env_data_loader(env_bipedal_walker, encoder_actor):
    return EnvDataLoader(
        num_time_steps=10,
        state_shape=(env_bipedal_walker.observation_space.shape),
        processing=None,
        policy=encoder_actor,
        env=env_bipedal_walker,
        use_imgs_as_states=False,
    )
