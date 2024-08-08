from typing import Optional
from dataclasses import dataclass
from reflect.models.agent import EPS
from reflect.models.rssm_world_model.rssm import RSSM, InternalState, InternalStateSequence
from reflect.models.rssm_world_model.models import DenseModel
from reflect.models.agent.actor import Actor
from reflect.models.observation_model import ConvEncoder, ConvDecoder
import torch.distributions as D
import torch.nn.functional as F
import torch
from reflect.utils import AdamOptim


@dataclass
class WorldModelTrainingParams:
    recon_coeff: float = 1.0
    dynamic_coeff: float = 1.0
    reward_coeff: float = 10.0
    done_coeff: float = 1.0
    rho: float = 3.0
    lr: float = 1e-3
    grad_clip: float = 1.0


@dataclass
class WorldModelLosses:
    recon_loss: float
    dynamic_model_kl_loss: float
    reward_loss: float
    done_loss: float
    loss: float


@dataclass
class ImaginedRollouts:
    rewards: torch.Tensor
    dones: torch.Tensor
    observations: Optional[torch.Tensor] = None


def get_norm_dist(means, stds=None):
    if stds is None:
        stds = torch.ones_like(means)
    normal = D.Normal(means, stds)
    return D.independent.Independent(normal, 1)


class WorldModel(torch.nn.Module):
    def __init__(
            self,
            encoder: ConvEncoder,
            decoder: ConvDecoder,
            done_model: DenseModel,
            reward_model: DenseModel,
            dynamic_model: RSSM,
            params: Optional[WorldModelTrainingParams] = None,
        ):
        super().__init__()

        if params is None:
            params = WorldModelTrainingParams()

        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.dynamic_model = dynamic_model
        self.reward_model = reward_model
        self.done_model = done_model

        self.opt = AdamOptim(
            self.parameters(),
            lr=params.lr,
            grad_clip=params.grad_clip
        )

    def observe_rollout(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
        ):
        o_emb = self.encoder(obs)
        prior_state_sequence, posterior_state_sequence = \
            self.dynamic_model.observe_rollout(o_emb, action)
        return o_emb, prior_state_sequence, posterior_state_sequence

    def update(
            self,
            prior_state_sequence: InternalStateSequence,
            posterior_state_sequence: InternalStateSequence,
            obs: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
        ) -> WorldModelLosses:
        prior_rewards = self.reward_model(
            prior_state_sequence.get_features()
        )
        prior_dones = self.done_model(
            prior_state_sequence.get_features()
        )
        prior_dist = prior_state_sequence.get_dist()
        posterior_dist = posterior_state_sequence.get_dist()
        reward_dist = get_norm_dist(prior_rewards)
        dynamic_model_kl_loss = D.kl_divergence(
            prior_dist,
            posterior_dist
        )
        # dynamic_model_kl_loss = torch.max(
        #     dynamic_model_kl_loss,
        #     torch.ones_like(dynamic_model_kl_loss) * self.params.rho
        # )
        dynamic_model_kl_loss = dynamic_model_kl_loss.mean()
        reward_loss = -reward_dist.log_prob(reward).mean()
        done_loss = F.binary_cross_entropy_with_logits(prior_dones, done.float())
        decoded_obs = self.decoder(
            prior_state_sequence.get_features()
        )
        recon_loss = torch.nn.functional.mse_loss(decoded_obs, obs)
        loss = self.params.recon_coeff * recon_loss + \
            self.params.dynamic_coeff * dynamic_model_kl_loss + \
            self.params.reward_coeff * reward_loss + \
            self.params.done_coeff * done_loss

        self.opt.backward(loss)
        self.opt.update_parameters()
        
        return WorldModelLosses(
            recon_loss=recon_loss.item(),
            dynamic_model_kl_loss=dynamic_model_kl_loss.item(),
            reward_loss=reward_loss.item(),
            done_loss=done_loss.item(),
            loss=loss.item()
        )

    def imagine_rollout(
            self,
            initial_states: InternalState,
            actor: Actor,
            o_emb: torch.Tensor,
            n_steps: int,
            with_observations: bool = False
        ):
        state_sequence = self.dynamic_model.imagine_rollout(
            initial_states=initial_states,
            obs_embed=o_emb,
            actor=actor,
            n_steps=n_steps
        )
        features = state_sequence.get_features()
        rewards = self.reward_model(features)
        dones = self.done_model(features)
        obs = None
        if with_observations:
            obs = self.decoder(features)
        return ImaginedRollouts(
            rewards=rewards,
            dones=dones,
            observations=obs
        )