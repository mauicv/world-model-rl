import pytest
import gymnasium as gym
from reflect.components.transformer_world_model.transformer import PytfexTransformer
from reflect.data.loader import EnvDataLoader, GymRenderImgProcessing

from reflect.components.models import ConvEncoder, ConvDecoder
from reflect.components.rssm_world_model.models import DenseModel
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
def decoder():
    return ConvDecoder(
        output_shape=(3, 64, 64),
        input_size=1024,
        activation=torch.nn.ReLU(),
        depth=32
    )

@pytest.fixture
def actor():
    return Actor(
        input_dim=1024,
        output_dim=8,
        bound=1,
        num_layers=3,
        hidden_dim=512,
    )

@pytest.fixture
def env_data_loader(world_model_actor):
    env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
    return EnvDataLoader(
        num_time_steps=10,
        img_shape=(3, 64, 64),
        processing=GymRenderImgProcessing(
            transforms=Compose([
                Resize((64, 64))
            ])
        ),
        policy=world_model_actor,
        env=env
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
def dynamic_model_1d_action():
    return make_dynamic_model(1)


@pytest.fixture
def dynamic_model_8d_action():
    return make_dynamic_model(8)

def make_dynamic_model(a_size):
    hdn_dim=32
    num_heads=8
    latent_dim=32
    num_cat=32
    t_dim=16
    layers=2
    dropout=0.05

    dynamic_model = PytfexTransformer(
        dropout=dropout,
        hdn_dim=hdn_dim,
        num_heads=num_heads,
        num_ts=t_dim,
        num_cat=num_cat,
        latent_dim=latent_dim,
        action_size=a_size,
        num_layers=layers
    )
    return dynamic_model