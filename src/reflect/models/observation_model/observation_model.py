import torch
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
