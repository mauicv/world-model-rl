import pytest
import gymnasium as gym
from reflect.components.observation_model.observation_model import ObservationalModel
from reflect.components.observation_model.latent_spaces import DiscreteLatentSpace
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.components.transformer_world_model.head import Head
from reflect.components.transformer_world_model.embedder import Embedder
from reflect.data.loader import EnvDataLoader

from reflect.components.observation_model.encoder import ConvEncoder
from reflect.components.observation_model.decoder import ConvDecoder
from reflect.components.rssm_world_model.models import DenseModel
from reflect.components.rssm_world_model.rssm import ContinuousRSSM, DiscreteRSSM
from reflect.components.rssm_world_model.world_model import WorldModel
from reflect.components.rssm_world_model.memory_actor import WorldModelActor
from reflect.components.actor import Actor
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
        input_size=64,
        activation=torch.nn.ReLU(),
        depth=32
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
def env_data_loader(world_model_actor):
    env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
    return EnvDataLoader(
        num_time_steps=10,
        img_shape=(3, 64, 64),
        transforms=Compose([Resize((64, 64))]),
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
    t_dim=48
    input_dim=1024
    layers=2
    dropout=0.05

    dynamic_model = GPT(
        dropout=dropout,
        hidden_dim=hdn_dim,
        num_heads=num_heads,
        embedder=Embedder(
            z_dim=input_dim,
            a_size=a_size,
            hidden_dim=hdn_dim
        ),
        head=Head(
            latent_dim=latent_dim,
            num_cat=num_cat,
            hidden_dim=hdn_dim
        ),
        layers=[
            TransformerLayer(
                hidden_dim=hdn_dim,
                attn=RelativeAttention(
                    hidden_dim=hdn_dim,
                    num_heads=num_heads,
                    num_positions=t_dim,
                    dropout=dropout
                ),
                mlp=MLP(
                    hidden_dim=hdn_dim,
                    dropout=dropout
                )
            ) for _ in range(layers)
        ]
    )
    return dynamic_model