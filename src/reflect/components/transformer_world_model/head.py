import torch
import torch.distributions as D
from reflect.components.general import DenseModel


class Head(torch.nn.Module):
    def __init__(
            self,
            predictor: DenseModel,
            latent_dim: int,
            num_cat: int,
            hidden_dim: int,
        ):
        super(Head, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_cat = num_cat
        self.predictor = predictor

    def forward(self, x):
        b, t, _ = x.shape
        s = self.predictor(x)
        s_logits = s.reshape(b, t, self.latent_dim, self.num_cat)
        return s_logits, x