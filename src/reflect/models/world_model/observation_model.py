import torch
from pytfex.convolutional.decoder import DecoderLayer, Decoder
from pytfex.convolutional.encoder import EncoderLayer, Encoder
import torch.distributions as D
from functools import reduce
import operator


class ObservationalModel(torch.nn.Module):
    def __init__(
            self,
            encoder=None,
            decoder=None,
            latent_space=None,
        ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_space = latent_space

    def forward(self, x):
        x_enc = self.encoder(x)
        y_enc, z, z_dist = self.latent_space(x_enc)
        return self.decoder(y_enc), z, z_dist

    def encode(self, x):
        x_enc = self.encoder(x)
        _, z, _ = self.latent_space(x_enc)
        return z

    def decode(self, z):
        y_enc = self.latent_space.decode(z)
        return self.decoder(y_enc)


class LatentSpace(torch.nn.Module):
    def __init__(
            self,
            input_shape=(1024, 4, 4),
            num_classes=32,
            num_latent=32,
        ):
        super().__init__()

        self.num_classes = num_classes
        self.num_latent = num_latent
        self.input_shape = input_shape
        self.flat_size = reduce(operator.mul, input_shape , 1)

        self.enc_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.flat_size, num_classes*num_latent)
        )

        self.dec_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_classes*num_latent, self.flat_size)
        )

    @staticmethod
    def create_z_dist(logits, temperature=1):
        assert temperature > 0
        dist = D.OneHotCategoricalStraightThrough(logits=logits / temperature)
        return D.Independent(dist, 1)

    def encode(self, x):
        b, *_ = x.shape
        x = x.reshape(b, -1)
        logits = self.enc_mlp(x)
        logits = logits.reshape(-1, self.num_latent, self.num_classes)
        return logits

    def forward(self, x):
        z_logits = self.encode(x)
        z_dist = self.create_z_dist(z_logits)
        z_sample = z_dist.rsample()
        return self.decode(z_sample), z_sample, z_logits

    def decode(self, z):
        z = z.flatten(1)
        x = self.dec_mlp(z)
        x = x.unflatten(-1, self.input_shape)
        return x