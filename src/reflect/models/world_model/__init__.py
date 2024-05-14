from reflect.models.td3_policy.actor import Actor
from reflect.data.loader import EnvDataLoader
from reflect.models.world_model.observation_model import ObservationalModel
from reflect.models.world_model.dynamic_model import DynamicsModel
import torch
from reflect.models.td3_policy import EPS
from reflect.utils import (
    recon_loss_fn,
    reg_loss_fn,
    cross_entropy_loss_fn,
    reward_loss_fn,
    AdamOptim
)
import torch.distributions as D
from torchvision.transforms import Resize, Compose
torch.autograd.set_detect_anomaly(True)

done_loss_fn = torch.nn.BCELoss()


def create_z_dist(logits, temperature=1):
    assert temperature > 0
    dist = D.OneHotCategoricalStraightThrough(logits=logits / temperature)
    return D.Independent(dist, 1)


def get_causal_mask(l):
    mask = torch.tril(torch.ones(l, l))
    masked_indices = mask[None, None, :l, :l] == 0
    return masked_indices


class WorldModel(torch.nn.Module):
    def __init__(
            self,
            observation_model: ObservationalModel,
            dynamic_model: DynamicsModel,
            num_ts: int,
        ):
        super().__init__()
        self.observation_model = observation_model
        self.dynamic_model = dynamic_model
        self.num_ts = num_ts
        self.mask = get_causal_mask(self.num_ts)
        self.observation_model_opt = AdamOptim(
            self.observation_model.parameters(),
            lr=0.0001,
            eps=1e-5,
            weight_decay=1e-6,
            grad_clip=100
        )
        self.dynamic_model_opt = AdamOptim(
            self.dynamic_model.parameters(),
            lr=0.0001,
            eps=1e-5,
            weight_decay=1e-6,
            grad_clip=100
        )

    def step(
            self,
            z: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor,
        ):
        z_dist, new_r, new_d = self.dynamic_model((
            z[:, -self.num_ts:],
            a[:, -self.num_ts:],
            r[:, -self.num_ts:]
        ))

        new_z = z_dist.sample()[:, -1].reshape(-1, 1, 32 * 32)
        z = torch.cat([z, new_z], dim=1)

        new_r = new_r[:, -1].reshape(-1, 1, 1)
        r = torch.cat([r, new_r], dim=1)

        new_d = new_d[:, -1].reshape(-1, 1, 1)
        d = torch.cat([d, new_d], dim=1)

        return z, r, d

    def update(
            self,
            o: torch.Tensor,
            a: torch.Tensor,
            r: torch.Tensor,
            d: torch.Tensor
        ):
        self.mask.to(o.device)
        b, t, *_  = o.shape
        o = o.flatten(0, 1)

        # Observational Model
        r_o, z, z_dist = self.observation_model(o)
        recon_loss = recon_loss_fn(o, r_o)
        reg_loss = 2 * reg_loss_fn(z_dist)

        # Dynamic Models
        z = z.detach()
        _, num_z, num_c = z_dist.base_dist.logits.shape
        z_logits = z_dist.base_dist.logits
        z_logits = z_logits.reshape(b, t, num_z, num_c)
        z_logits = z_logits[:, 1:]
        next_z_dist = create_z_dist(z_logits.detach())

        z = z.reshape(b, t, -1)
        r_targets = r[:, 1:]
        d_targets = d[:, 1:]
        z_inputs, r_inputs, a_inputs = z[:, :-1], r[:, :-1], a[:, :-1]
        z_pred, r_pred, d_pred = self.dynamic_model(
            (z_inputs, a_inputs, r_inputs),
            mask=self.mask
        )
        dynamic_loss = cross_entropy_loss_fn(z_pred, next_z_dist)
        reward_loss = reward_loss_fn(r_targets, r_pred)
        done_loss = done_loss_fn(d_pred, d_targets.float())

        # Update observation_model and dynamic_model
        consistency_loss = 0.01 * cross_entropy_loss_fn(next_z_dist, z_pred)
        obs_loss = recon_loss + reg_loss + consistency_loss
        self.observation_model_opt.step(obs_loss, retain_graph=True)

        dyn_loss = dynamic_loss + 10 * reward_loss + done_loss
        self.dynamic_model_opt.step(dyn_loss, retain_graph=False)

        return {
            'recon_loss': recon_loss.cpu().item(),
            'reg_loss': reg_loss.cpu().item(),
            'consistency_loss': consistency_loss.cpu().item(),
            'dynamic_loss': dynamic_loss.cpu().item(),
            'reward_loss': reward_loss.cpu().item(),
            'done_loss': done_loss.cpu().item(),
        }

    def load(self):
        pass

    def save(self):
        pass