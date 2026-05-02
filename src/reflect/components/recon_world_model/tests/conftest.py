import pytest
import gymnasium as gym

from reflect.components.latent_world_model.models import MLPDynamicModel, MLPEncoder
from reflect.components.latent_world_model.models.actor import MLPActor
from reflect.components.latent_world_model.models.encoder_actor import EncoderActor
from reflect.data.loader import EnvDataLoader

LATENT_DIM = 512


@pytest.fixture
def env():
    return gym.make("BipedalWalker-v3", render_mode="rgb_array")


@pytest.fixture
def encoder(env):
    return MLPEncoder(
        input_dim=env.observation_space.shape[0],
        output_dim=LATENT_DIM,
        num_layers=3,
        hidden_dim=512,
    )


@pytest.fixture
def decoder(env):
    return MLPEncoder(
        input_dim=LATENT_DIM,
        output_dim=env.observation_space.shape[0],
        num_layers=3,
        hidden_dim=512,
    )


@pytest.fixture
def dynamic_model(env):
    return MLPDynamicModel(
        latent_dim=LATENT_DIM,
        action_dim=env.action_space.shape[0],
        num_layers=3,
        hidden_dim=512,
    )


@pytest.fixture
def actor(env):
    return MLPActor(
        latent_dim=LATENT_DIM,
        action_dim=env.action_space.shape[0],
        num_layers=3,
        hidden_dim=512,
    )


@pytest.fixture
def encoder_actor(encoder, actor):
    return EncoderActor(encoder=encoder, actor=actor, latent_dim=LATENT_DIM)


@pytest.fixture
def env_data_loader(env, encoder_actor):
    return EnvDataLoader(
        num_time_steps=10,
        state_shape=env.observation_space.shape,
        processing=None,
        policy=encoder_actor,
        env=env,
        use_imgs_as_states=False,
    )
