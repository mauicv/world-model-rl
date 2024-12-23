from pytfex.convolutional.decoder import DecoderLayer, Decoder
from pytfex.convolutional.encoder import EncoderLayer, Encoder
from pytfex.transformer.gpt import GPT
from pytfex.transformer.layer import TransformerLayer
from pytfex.transformer.mlp import MLP
from pytfex.transformer.attention import RelativeAttention
from reflect.components.transformer_world_model.head import StackHead
from reflect.components.transformer_world_model.embedder import StackEmbedder
import torch

hdn_dim=256
num_heads=4
latent_dim=8
num_cat=8
t_dim=3
input_dim=8*8
num_layers=4
dropout=0.05
a_size=2


def make_models():
    dynamic_model = GPT(
        dropout=dropout,
        hidden_dim=hdn_dim,
        num_heads=num_heads,
        embedder=StackEmbedder(
            z_dim=input_dim,
            a_size=a_size,
            hidden_dim=hdn_dim
        ),
        head=StackHead(
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
    ]

    encoder = Encoder(
        nc=3,
        ndf=64,
        layers=encoder_layers,
    )
    
    decoder_layers = [
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

    # latent_space = DiscreteLatentSpace(
    #     num_latent=latent_dim,
    #     num_classes=num_cat,
    #     input_shape=(128, 2, 2),
    # )

    # observation_model = ObservationalModel(
    #     encoder=encoder,
    #     decoder=decoder,
    #     latent_space=latent_space,
    # )

    return dynamic_model