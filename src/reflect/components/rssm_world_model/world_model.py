from typing import Optional
from dataclasses import dataclass
from reflect.components.rssm_world_model.rssm import (
    RSSM,
    InternalStateContinuous,
    InternalStateContinuousSequence
)
from reflect.components.rssm_world_model.models import DenseModel
from reflect.components.actor import Actor
from reflect.components.observation_model import ConvEncoder, ConvDecoder
import torch.distributions as D
import torch.nn.functional as F
import torch
from reflect.utils import AdamOptim, FreezeParameters


@dataclass
class WorldModelTrainingParams:
    recon_coeff: float = 1.0
    dynamic_coeff: float = 1.0
    reward_coeff: float = 10.0
    done_coeff: float = 1.0
    rho: float = 3.0
    lr: float = 6e-4
    grad_clip: float = 1.0


@dataclass
class WorldModelLosses:
    recon_loss: float
    dynamic_model_loss: float
    dynamic_model_loss_clamped: float
    reward_loss: float
    done_loss: float
    loss: float
    grad_norm: float


@dataclass
class ImaginedRollouts:
    rewards: torch.Tensor
    dones: torch.Tensor
    features: torch.Tensor
    observations: Optional[torch.Tensor] = None

    @property
    def shapes(self):
        return (
            self.rewards.shape,
            self.dones.shape,
            self.features.shape
        )


def get_norm_dist(means, stds=None):
    if stds is None:
        stds = torch.ones_like(means)
    normal = D.Normal(means, stds)
    return D.independent.Independent(normal, 1)


def observation_loss(obs, decoded_obs):
    obs = obs.reshape(-1, *obs.shape[2:])
    decoded_obs = decoded_obs.reshape(-1, *decoded_obs.shape[2:])
    normal = D.Independent(D.Normal(decoded_obs, 1.0), 3)
    return -normal.log_prob(obs).mean()


class WorldModel(torch.nn.Module):
    model_list = [
        'encoder',
        'decoder',
        'dynamic_model',
        'reward_model',
        'done_model',
        'opt'
    ]

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
            prior_state_sequence: InternalStateContinuousSequence,
            posterior_state_sequence: InternalStateContinuousSequence,
            obs: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
        ) -> WorldModelLosses:
        posterior_features = posterior_state_sequence.get_features()
        posterior_rewards = self.reward_model(posterior_features)
        posterior_dones = self.done_model(posterior_features)
        prior_dist = prior_state_sequence.get_dist()
        posterior_dist = posterior_state_sequence.get_dist()
        reward_dist = get_norm_dist(posterior_rewards)
        dynamic_model_loss = D.kl_divergence(
            prior_dist,
            posterior_dist
        )
        dynamic_model_loss_clamped = torch.max(
            dynamic_model_loss,
            torch.ones_like(dynamic_model_loss) * self.params.rho
        ).mean()
        dynamic_model_loss_clamped = dynamic_model_loss_clamped.mean()
        reward_loss = -reward_dist.log_prob(reward[:, 1:]).mean()
        done_loss = F.binary_cross_entropy_with_logits(posterior_dones, done[:, 1:].float())
        decoded_obs = self.decoder(posterior_features)
        recon_loss = observation_loss(decoded_obs, obs[:, 1:])
        loss = self.params.recon_coeff * recon_loss + \
            self.params.dynamic_coeff * dynamic_model_loss_clamped + \
            self.params.reward_coeff * reward_loss + \
            self.params.done_coeff * done_loss

        grad_norm = self.opt.backward(loss)
        self.opt.update_parameters()
        
        return WorldModelLosses(
            recon_loss=recon_loss.item(),
            dynamic_model_loss=dynamic_model_loss.mean().item(),
            dynamic_model_loss_clamped=dynamic_model_loss_clamped.item(),
            reward_loss=reward_loss.item(),
            done_loss=done_loss.item(),
            loss=loss.item(),
            grad_norm=grad_norm.item()
        )

    def imagine_rollout(
            self,
            initial_states: InternalStateContinuous,
            actor: Actor,
            n_steps: int,
            with_observations: bool = False
        ):
        with FreezeParameters([self]):
            state_sequence = self.dynamic_model.imagine_rollout(
                initial_states=initial_states,
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
            features=features,
            observations=obs
        )

    def load(
            self,
            path,
            name="world-model-checkpoint.pth",
            targets=None
        ):
        device = next(self.parameters()).device
        checkpoint = torch.load(
            f'{path}/{name}',
            map_location=torch.device(device)
        )
        if targets is None: targets = self.model_list
        
        for target in targets:
            print(f'Loading {target}...')
            getattr(self, target).load_state_dict(
                checkpoint[target]
            )

    def save(
            self,
            path,
            name="world-model-checkpoint.pth",
            targets=None
        ):
        if targets is None: targets = self.model_list
        
        checkpoint = {
            target: getattr(self, target).state_dict()
            for target in targets
        }
        torch.save(checkpoint, f'{path}/{name}')