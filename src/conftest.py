import pytest
from reflect.models.world_model.observation_model import ObservationalModel, LatentSpace
from pytfex.convolutional.decoder import DecoderLayer, Decoder
from pytfex.convolutional.encoder import EncoderLayer, Encoder
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.models.world_model.head import Head
from reflect.models.world_model.embedder import Embedder
import torch


@pytest.fixture
def observation_model():
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
        )
    ]

    encoder = Encoder(
        nc=3,
        ndf=64,
        layers=encoder_layers,
    )
    
    layers = [
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
        ),
    ]

    decoder = Decoder(
        nc=3,
        ndf=64,
        layers=layers,
        output_activation=torch.nn.Sigmoid(),
    )

    latent_space = LatentSpace(
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
def dynamic_model_1d_action():
    return make_dynamic_model(1)


@pytest.fixture
def dynamic_model_8d_action():
    return make_dynamic_model(8)


def make_dynamic_model(a_size):
    hdn_dim=256
    num_heads=8
    latent_dim=32
    num_cat=32
    t_dim=48
    input_dim=1024
    layers=10
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