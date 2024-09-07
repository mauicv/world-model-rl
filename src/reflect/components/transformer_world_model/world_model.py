from typing import Optional
from dataclasses import dataclass
from reflect.components.observation_model import ConvEncoder, ConvDecoder
from reflect.components.actor import Actor
import torch
from reflect.utils import AdamOptim, AnnealingParams, FreezeParameters
import torch.distributions as D
from reflect.components.base import Base
from reflect.components.transformer_world_model.transformer import Sequence, Transformer, ImaginedRollout


done_loss_fn = torch.nn.BCELoss()


@dataclass
class WorldModelLosses:
    recon_loss: float
    dynamic_model_loss: float
    dynamic_model_loss_clamped: float
    reward_loss: float
    done_loss: float
    loss: float
    grad_norm: float
    lr: float


@dataclass
class WorldModelTrainingParams:
    reg_coeff: float = 0.0
    recon_coeff: float = 1.0
    dynamic_coeff: float = 1.0
    consistency_coeff: float = 0.0
    reward_coeff: float = 10.0
    done_coeff: float = 1.0
    lr: float = 2e-4
    grad_clip: float = 100.0
    rho: float = 3.0
    annealing_params: Optional[AnnealingParams] = AnnealingParams()


class TransformerWorldModel(Base):
    model_list = [
        'encoder',
        'decoder',
        'dynamic_model',
        'opt'
    ]

    def __init__(
            self,
            encoder: ConvEncoder,
            decoder: ConvDecoder,
            dynamic_model: Transformer,
            params: Optional[WorldModelTrainingParams] = None,
        ):
        super().__init__()
        if params is None:
            params = WorldModelTrainingParams()
        self.params = params
        self.encoder = encoder
        self.decoder = decoder
        self.dynamic_model = dynamic_model
        self.opt = AdamOptim(
            self.parameters(),
            lr=params.lr,
            grad_clip=params.grad_clip,
            annealing_params=params.annealing_params,
            optimizer='AdamW'
        )


    def observe_rollout(
            self,
            observation: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor
        ):
        num_ts = self.dynamic_model.num_ts
        latent_dim = self.dynamic_model.latent_dim
        num_cat = self.dynamic_model.num_cat
        assert num_ts + 1 == observation.shape[1], (
            "Observation sequence length must be num_ts + 1"
            f" currently num_ts + 1 = {num_ts + 1}"
            f" observation.shape[1] = {observation.shape[1]}"
        )
        b, t, *_ = observation.shape
        state = (
            self.encoder(observation)
            .reshape(b, t, latent_dim, num_cat)
        )
        sequence = Sequence.from_sard(
            state=state,
            action=action,
            reward=reward,
            done=done,
        )
        num_ts = self.dynamic_model.num_ts
        target = sequence.last(ts=num_ts)
        output = self.dynamic_model(sequence.first(ts=num_ts))
        return target, output

    def update(
            self,
            target: Sequence,
            output: Sequence,
            observations: torch.Tensor,
            params: Optional[WorldModelTrainingParams] = None,
            global_step: Optional[int] = None
        ):
        if params is None:
            params = self.params

        # dynamic loss
        dynamic_model_loss = D.kl_divergence(
            target.state_dist,
            output.state_dist
        )
        dynamic_model_loss_clamped = torch.max(
            dynamic_model_loss,
            torch.ones_like(dynamic_model_loss) * self.params.rho
        ).mean()

        # reward and done loss
        reward_loss = - output.reward.log_prob(target.reward.base_dist.mean).mean()
        done_loss = - output.done.log_prob(target.done.base_dist.mean).mean()

        # reconstruction loss
        decoder_input = torch.cat(
            [output.hdn_state, target.state_sample],
            dim=-1
        )
        recon_observations = self.decoder(decoder_input)
        recon_dist = D.Normal(
            recon_observations,
            torch.ones_like(recon_observations)
        )
        recon_dist = D.Independent(recon_dist, 3)
        recon_loss = - recon_dist.log_prob(observations[:, 1:]).mean()

        loss = (
            params.dynamic_coeff * dynamic_model_loss_clamped 
            + params.reward_coeff * reward_loss
            + params.done_coeff * done_loss
            + params.recon_coeff * recon_loss
        )

        grad_norm = self.opt.backward(loss, retain_graph=False)
        lr = self.opt.update_parameters(global_step=global_step)

        return WorldModelLosses(
            recon_loss=recon_loss.detach().cpu().item(),
            dynamic_model_loss=dynamic_model_loss.mean().detach().cpu().item(),
            dynamic_model_loss_clamped=dynamic_model_loss_clamped.detach().cpu().item(),
            reward_loss=reward_loss.detach().cpu().item(),
            done_loss=done_loss.detach().cpu().item(),
            loss=loss.detach().cpu().item(),
            grad_norm=grad_norm.detach().cpu().item(),
            lr=lr
        )

    def imagine_rollout(
            self,
            initial_state: ImaginedRollout,
            actor: Actor,
            n_steps: int,
            with_observations: bool = False
        ):
        with FreezeParameters([self]):
            state_sequence = self.dynamic_model.imagine_rollout(
                initial_state=initial_state,
                actor=actor,
                n_steps=n_steps
            )
            obs = None
            if with_observations:
                obs = self.decoder(state_sequence.to_decoder_input())
                state_sequence.observations = obs
        return state_sequence
