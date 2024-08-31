from typing import Optional
import torch
from reflect.components.general import DenseModel
from reflect.components.transformer_world_model.distribution import create_z_dist, create_norm_dist
from reflect.components.transformer_world_model.state import StateDistribution


class Head(torch.nn.Module):
    def __init__(
            self,
            reward_model: DenseModel,
            done_model: DenseModel,
            discrete_latent_dim: int,
            continuous_latent_dim: int,
            num_cat: int,
            hidden_dim: int,
            predictor: Optional[DenseModel]=None,
        ):
        super(Head, self).__init__()
        self.hidden_dim = hidden_dim
        self.discrete_latent_dim = discrete_latent_dim
        self.continuous_latent_dim = continuous_latent_dim
        self.num_cat = num_cat
        self.predictor = predictor
        self.reward_model = reward_model
        self.done_model = done_model

    def forward(self, x):
        b, *_ = x.shape
        reshaped_x = x.view(b, -1, 3, self.hidden_dim)
        s_emb, a_emb, r_emb = reshaped_x.unbind(dim=2)
        d_mean = self.done_model(s_emb)
        r_mean = self.reward_model(r_emb)
        state = self.predictor(a_emb)
        sizes = (
            self.discrete_latent_dim*self.num_cat,
            self.continuous_latent_dim,
            self.continuous_latent_dim,
        )
        discrete_state_logits, continuous_state_mean, continuous_state_std = \
            torch.split(state, sizes, dim=-1)
        discrete_state_logits = discrete_state_logits \
            .reshape(b, -1,  self.discrete_latent_dim, self.num_cat)

        dist = StateDistribution.from_sard(
            continuous_mean=continuous_state_mean,
            continuous_std=continuous_state_std,
            discrete=discrete_state_logits,
            reward=r_mean,
            done=d_mean
        )
        return dist