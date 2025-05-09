import pytest
import gymnasium as gym
from reflect.data.basic_loader import EnvDataLoader, GymRenderImgProcessing

from reflect.components.models import ConvEncoder, ConvDecoder
from reflect.components.rssm_world_model.models import DenseModel
from reflect.components.rssm_world_model.rssm import ContinuousRSSM, DiscreteRSSM
from reflect.components.rssm_world_model.world_model import WorldModel
from reflect.components.rssm_world_model.memory_actor import WorldModelActor
from reflect.components.models.actor import Actor
from reflect.components.trainers.reward.reward_trainer import RewardGradTrainer
from reflect.components.trainers.value.value_trainer import ValueGradTrainer
from reflect.components.trainers.value.critic import ValueCritic

import torch
from torchvision.transforms import Resize, Compose


@pytest.fixture
def encoder():
    return ConvEncoder(
        input_shape=(3, 64, 64),
        embed_size=1024,
        activation=torch.nn.ReLU(),
        depth=32
    )

@pytest.fixture
def state_encoder():
    return DenseModel(
        depth=1,
        input_dim=4,
        hidden_dim=256,
        output_dim=1024,
    )

@pytest.fixture
def decoder():
    return ConvDecoder(
        output_shape=(3, 64, 64),
        input_size=64,
        activation=torch.nn.ReLU(),
        depth=32
    )

@pytest.fixture
def state_decoder():
    return DenseModel(
        depth=1,
        input_dim=64,
        hidden_dim=256,
        output_dim=4,
    )

@pytest.fixture
def continuous_rssm():
    return ContinuousRSSM(
        hidden_size=32,
        deter_size=32,
        stoch_size=32,
        obs_embed_size=1024,
        action_size=1,
    )

@pytest.fixture
def discrete_rssm():
    return DiscreteRSSM(
        hidden_size=32,
        deter_size=32,
        stoch_size=8,
        num_categories=4,
        obs_embed_size=1024,
        action_size=1,
    )

@pytest.fixture
def actor():
    return Actor(
        input_dim=64,
        output_dim=1,
        bound=1,
        num_layers=3,
        hidden_dim=512,
    )

@pytest.fixture
def state_world_model(continuous_rssm, state_encoder, state_decoder, done_model, reward_model):
    return WorldModel(
        encoder=state_encoder,
        decoder=state_decoder,
        dynamic_model=continuous_rssm,
        done_model=done_model,
        reward_model=reward_model
    )

@pytest.fixture
def world_model(continuous_rssm, encoder, decoder, done_model, reward_model):
    return WorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=continuous_rssm,
        done_model=done_model,
        reward_model=reward_model
    )

@pytest.fixture
def discrete_world_model(discrete_rssm, encoder, decoder, done_model, reward_model):
    return WorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=discrete_rssm,
        done_model=done_model,
        reward_model=reward_model
    )

@pytest.fixture
def discrete_state_world_model(discrete_rssm, state_encoder, state_decoder, done_model, reward_model):
    return WorldModel(
        encoder=state_encoder,
        decoder=state_decoder,
        dynamic_model=discrete_rssm,
        done_model=done_model,
        reward_model=reward_model
    )

@pytest.fixture
def env_data_loader(world_model_actor):
    env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
    return EnvDataLoader(
        num_time_steps=10,
        state_shape=(3, 64, 64),
        processing=GymRenderImgProcessing(
            transforms=Compose([
                Resize((64, 64))
            ])
        ),
        policy=world_model_actor,
        env=env
    )

@pytest.fixture
def env_state_data_loader(state_world_model_actor):
    env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
    return EnvDataLoader(
        num_time_steps=10,
        state_shape=(4,),
        policy=state_world_model_actor,
        env=env,
        use_imgs_as_states=False,
    )

@pytest.fixture
def reward_model():
    return DenseModel(
        input_dim=64,
        hidden_dim=256,
        output_dim=1,
    )

@pytest.fixture
def done_model():
    return DenseModel(
        input_dim=64,
        hidden_dim=256,
        output_dim=1,
        output_act=torch.nn.Sigmoid
    )

@pytest.fixture
def reward_grad_trainer(actor):
    return RewardGradTrainer(
        actor=actor,
        lr=0.001,
        grad_clip=1.0
    )

@pytest.fixture
def value_model():
    return ValueCritic(64, 4, 400)

@pytest.fixture
def value_grad_trainer(actor, value_model):
    return ValueGradTrainer(
        actor=actor,
        actor_lr=0.001,
        critic=value_model,
        critic_lr=0.001,
        grad_clip=1.0
    )

@pytest.fixture
def world_model_actor(world_model, actor):
    return WorldModelActor(
        actor=actor,
        world_model=world_model
    )

@pytest.fixture
def state_world_model_actor(state_world_model, actor):
    return WorldModelActor(
        actor=actor,
        world_model=state_world_model
    )
