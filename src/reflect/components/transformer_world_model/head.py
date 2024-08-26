import torch
import torch.distributions as D
from reflect.components.general import DenseModel


class Head(torch.nn.Module):
    def __init__(
            self,
            predictor: DenseModel,
            reward_model: DenseModel,
            done_model: DenseModel,
            latent_dim: int,
            num_cat: int,
            hidden_dim: int,
        ):
        super(Head, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_cat = num_cat
        self.predictor = predictor
        self.reward_model = reward_model
        self.done_model = done_model

    def forward(self, x):
        b, t, _ = x.shape
        reshaped_x = x.view(b, -1, 3, self.hidden_dim)
        s_emb, a_emb, r_emb = reshaped_x.unbind(dim=2)
        d_mean = self.done_model(s_emb)
        r_mean = self.reward_model(r_emb)
        s = self.predictor(a_emb)
        s_logits = s.reshape(b, int(t/3), self.latent_dim, self.num_cat)
        return s_logits, r_mean, d_mean