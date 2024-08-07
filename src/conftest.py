import pytest
import gymnasium as gym
from reflect.models.observation_model.observation_model import ObservationalModel
from reflect.models.observation_model.latent_spaces import DiscreteLatentSpace
from reflect.models.observation_model.encoder import ConvEncoder
from reflect.models.observation_model.decoder import ConvDecoder
from reflect.models.rssm_world_model.models import DenseModel
from reflect.models.rssm_world_model.rssm import RSSM
from reflect.models.rssm_world_model.world_model import WorldModel
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.models.agent.actor import Actor
from reflect.models.transformer_world_model.head import Head
from reflect.models.transformer_world_model.embedder import Embedder
from reflect.data.loader import EnvDataLoader
from reflect.models.agent.reward_trainer import RewardGradTrainer

import torch
from torchvision.transforms import Resize, Compose

@pytest.fixture
def observation_model():
    encoder = ConvEncoder(
        input_shape=(3, 64, 64),
        embed_size=1024,
        activation=torch.nn.ReLU(),
        depth=32
    )

    decoder = ConvDecoder(
        output_shape=(3, 64, 64),
        input_size=1024,
        activation=torch.nn.ReLU(),
        depth=32
    )

    latent_space = DiscreteLatentSpace(
        num_classes=32,
        num_latent=32,
        input_shape=(1024, 4, 4),
    )

    return ObservationalModel(
        encoder=encoder,
        decoder=decoder,
        latent_space=latent_space,
    )

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
        input_size=230,
        activation=torch.nn.ReLU(),
        depth=32
    )

@pytest.fixture
def rssm():
    return RSSM(
        hidden_size=200,
        deter_size=200,
        stoch_size=30,
        obs_embed_size=1024,
        action_size=1,
    )

@pytest.fixture
def actor():
    return Actor(
        230,
        1,
        1,
        num_layers=3,
        hidden_dim=512,
        repeat=1
    )

@pytest.fixture
def world_model(rssm, encoder, decoder, done_model, reward_model):
    return WorldModel(
        encoder=encoder,
        decoder=decoder,
        dynamic_model=rssm,
        done_model=done_model,
        reward_model=reward_model
    )

@pytest.fixture
def env_data_loader(encoder):
    env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
    return EnvDataLoader(
        num_time_steps=10,
        img_shape=(3, 64, 64),
        transforms=Compose([Resize((64, 64))]),
        observation_model=encoder,
        env=env
    )

@pytest.fixture
def reward_model():
    return DenseModel(
        input_dim=230,
        hidden_dim=256,
        output_dim=1,
    )

@pytest.fixture
def done_model():
    return DenseModel(
        input_dim=230,
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