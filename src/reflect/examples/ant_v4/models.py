from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.environment import Environment
from reflect.models.world_model import WorldModel
from torchvision.transforms import Resize, Compose
from reflect.models.world_model.observation_model import ObservationalModel, LatentSpace
from pytfex.convolutional.decoder import DecoderLayer, Decoder
from pytfex.convolutional.encoder import EncoderLayer, Encoder
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.models.world_model.head import Head
from reflect.models.world_model.embedder import Embedder

import gymnasium as gym
import torch

hdn_dim=512
num_heads=8
latent_dim=256
num_cat=32
t_dim=65
input_dim=32*256
num_layers=10
# num_layers=2
dropout=0.0
a_size=8

def make_models():
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
            ) for _ in range(num_layers)
        ]
    )

    encoder_layers = [
        EncoderLayer(
            in_channels=64,
            out_channels=128,
            num_residual=0,
        ),
        EncoderLayer(
            in_channels=128,
            out_channels=256,
            num_residual=0,
        ),
        EncoderLayer(
            in_channels=256,
            out_channels=512,
            num_residual=0,
        ),
        EncoderLayer(
            in_channels=512,
            out_channels=1024,
            num_residual=0,
        ),
    ]

    encoder = Encoder(
        nc=3,
        ndf=64,
        layers=encoder_layers,
    )

    decoder_layers = [
        DecoderLayer(
            in_filters=1024,
            out_filters=512,
            num_residual=0,
        ),
        DecoderLayer(
            in_filters=512,
            out_filters=256,
            num_residual=0,
        ),
        DecoderLayer(
            in_filters=256,
            out_filters=128,
            num_residual=0,
        ),
        DecoderLayer(
            in_filters=128,
            out_filters=64,
            num_residual=0,
        )
    ]

    decoder = Decoder(
        nc=3,
        ndf=64,
        layers=decoder_layers,
        output_activation=torch.nn.Sigmoid(),
    )

    latent_space = LatentSpace(
        num_latent=latent_dim,
        num_classes=num_cat,
        input_shape=(1024, 4, 4),
    )

    observation_model = ObservationalModel(
        encoder=encoder,
        decoder=decoder,
        latent_space=latent_space,
    )

    world_model = WorldModel(
        dynamic_model=dynamic_model,
        observation_model=observation_model,
        num_ts=t_dim-1,
        num_cat=num_cat,
        num_latent=latent_dim,
    )

    world_model.load(
        "./experiments/ant-v4/",
        name="world-model-checkpoint-2.pth"
    )

    env = gym.make(
        "Ant-v4",
        render_mode="rgb_array"
    )
    env.reset()

    loader = EnvDataLoader(
        num_time_steps=t_dim,
        batch_size=32,
        num_runs=1,
        rollout_length=100,
        transforms=Compose([Resize((64, 64))]),
        img_shape=(3, 64, 64),
        env=env,
        observation_model=world_model.observation_model,
    )
    loader.perform_rollout()

    wm_env = Environment(
        world_model=world_model,
        data_loader=loader,
        batch_size=1,
        ignore_done=True
    )

    return wm_env