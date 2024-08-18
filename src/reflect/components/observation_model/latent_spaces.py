import torch
import torch.distributions as D
from functools import reduce
import operator


class DiscreteLatentSpace(torch.nn.Module):
    def __init__(
            self,
            input_shape=(1024, 4, 4),
            num_classes=32,
            num_latent=32,
        ):
        """DiscreteLatentSpace

        Maps flattened input to 32 probability vectors of 32 classes.

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1024, 4, 4).
            num_classes (int, optional): Number of classes. Defaults to 32.
            num_latent (int, optional): Number of latent variables. Defaults to 32.
        """
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
