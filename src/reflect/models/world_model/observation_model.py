import torch
import torch.nn.functional as F
import torch.distributions as D
from functools import reduce
import operator
from reflect.utils import create_z_dist


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
        _, z, *_ = self.latent_space(x_enc)
        return z

    def decode(self, z):
        y_enc = self.latent_space.decode(z)
        return self.decoder(y_enc)

    def loss(self, x):
        x_enc = self.encoder(x)
        y_enc, _, z_dist = self.latent_space(x_enc)
        obs_dist = self.decoder(y_enc)
        obs_loss = -torch.mean(obs_dist.log_prob(x))
        return obs_loss, {
            'std': z_dist.std.mean().item(),
        }

class LatentSpace(torch.nn.Module):
    def __init__(
            self,
            input_shape=(1024, 4, 4),
            latent_dim=1024
        ):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.flat_size = reduce(operator.mul, input_shape , 1)
        self.fc_1 = torch.nn.Linear(self.flat_size, 2*latent_dim)
        self.dec_mlp = torch.nn.Linear(latent_dim, self.flat_size)

    def encode(self, x):
        x = self.fc_1(x.flatten(1))
        mean, _ = x.chunk(2, dim=-1)
        return mean

    def forward(self, x):
        x = self.fc_1(x.flatten(1))
        mean, std = x.chunk(2, dim=-1)
        z_dist = create_z_dist(mean, std)
        z_sample = z_dist.rsample()
        return self.decode(z_sample), z_sample, z_dist

    def decode(self, z):
        z = z.flatten(1)
        x = self.dec_mlp(z)
        x = x.unflatten(-1, self.input_shape)
        return x