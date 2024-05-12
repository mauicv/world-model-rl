import torch
from pytfex.convolutional.decoder import DecoderLayer, Decoder
from pytfex.convolutional.encoder import EncoderLayer, Encoder
import torch.distributions as D


class ObservationalModel(torch.nn.Module):
    def __init__(self, num_classes=32, num_latent=32):
        super().__init__()

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

        self.encoder = Encoder(
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

        self.decoder = Decoder(
            nc=3,
            ndf=64,
            layers=layers,
            output_activation=torch.nn.Sigmoid(),
        )

        self.latent_space = LatentSpace(
            num_classes=num_classes,
            num_latent=num_latent
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        y_enc, z, z_dist = self.latent_space(x_enc)
        return self.decoder(y_enc), z, z_dist

    def encode(self, x):
        x_enc = self.encoder(x)
        _, z, _ = self.latent_space(x_enc)
        return z


class LatentSpace(torch.nn.Module):
    def __init__(self, num_classes=32, num_latent=32):
        super().__init__()

        self.num_classes = num_classes
        self.num_latent = num_latent

        self.enc_mlp = torch.nn.Sequential(
            torch.nn.Linear(1024*4*4, 768),
            torch.nn.Dropout(0.1),
            torch.nn.ELU(),
            torch.nn.Linear(768, 768),
            torch.nn.Dropout(0.1),
            torch.nn.ELU(),
            torch.nn.Linear(768, num_classes*num_latent)
        )

        self.dec_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_classes*num_latent, 768),
            torch.nn.ELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(768, 768),
            torch.nn.Dropout(0.1),
            torch.nn.ELU(),
            torch.nn.Linear(768, 1024*4*4),
        )

    @staticmethod
    def create_z_dist(logits, temperature=1):
        assert temperature > 0
        dist = D.OneHotCategoricalStraightThrough(logits=logits / temperature)
        return D.Independent(dist, 1)

    def encode(self, x):
        x = x.flatten(1)
        logits = self.enc_mlp(x)
        logits = logits.unflatten(-1, (self.num_latent, self.num_classes))
        return self.create_z_dist(logits)

    def forward(self, x):
        z_dist = self.encode(x)
        z_sample = z_dist.rsample()
        return self.decode(z_sample), z_sample, z_dist

    def decode(self, z):
        z = z.flatten(1)
        x = self.dec_mlp(z)
        x = x.unflatten(-1, (1024, 4, 4))
        return x