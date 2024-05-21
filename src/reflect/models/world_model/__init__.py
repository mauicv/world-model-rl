from reflect.models.world_model.observation_model import ObservationalModel
from pytfex.transformer.gpt import GPT
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
            dynamic_model: GPT,
            num_ts: int,
            num_cat: int=32,
            num_latent: int=32,
        ):
        super().__init__()
        self.observation_model = observation_model
        self.dynamic_model = dynamic_model
        self.num_ts = num_ts
        self.num_cat = num_cat
        self.num_latent = num_latent
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
            lr=0.00001,
            eps=1e-5,
            weight_decay=1e-6,
            grad_clip=100
        )

    def forward(self, x):
        z_dist = self.obs_model.encode(x)
        z = z_dist.rsample()
        y = self.observation_model.decoder(z)
        y_hat = self.dynamic_model(y)
        return y_hat, z, z_dist

    def encode(self, image):
        b, t, c, h, w = image.shape
        image = image.reshape(b * t, c, h, w)
        z = self.observation_model.encode(image)
        return z.reshape(b, t, -1)

    def decode(self, z):
        b, t, _ = z.shape
        z = z.reshape(b * t, -1)
        image = self.observation_model.decode(z)
        return image.reshape(b, t, *image.shape[1:])

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

        new_z = z_dist.sample()[:, -1].reshape(-1, 1, self.num_cat * self.num_latent)
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
            d: torch.Tensor,
        ):
        #############
        # test code #
        #############
        torch.autograd.set_detect_anomaly(True)
        def check_no_grad(*tensors):
            return all((t is None or not t.requires_grad) for t in tensors)

        self.mask.to(o.device)
        b, t, c, h, w  = o.shape
        o = o.reshape(b * t, c, h, w)

        # Observational Model
        r_o, z, z_dist = self.observation_model(o)
        recon_loss = recon_loss_fn(o, r_o)
        reg_loss = 2 * reg_loss_fn(z_dist)

        # Dynamic Models
        z = z.detach()
        _, num_z, num_c = z_dist.base_dist.logits.shape
        # Note this is wrong when training observation model, detaching z_dist logits
        # will cause the gradients to not flow through the observation model!
        z_logits = z_dist.base_dist.logits.detach()
        z_logits = z_logits.reshape(b, t, num_z, num_c)
        z_logits = z_logits[:, 1:]
        next_z_dist = create_z_dist(z_logits)

        z = z.reshape(b, t, -1)
        r_targets = r[:, 1:].detach()
        d_targets = d[:, 1:].detach()
        
        assert check_no_grad(next_z_dist.base_dist.logits, r_targets, d_targets) # test code

        z_inputs, r_inputs, a_inputs = z[:, :-1], r[:, :-1], a[:, :-1]

        assert check_no_grad(z_inputs, r_inputs, a_inputs) # test code

        z_pred, r_pred, d_pred = self.dynamic_model(
            (z_inputs, a_inputs, r_inputs),
            mask=self.mask
        )
        dynamic_loss = cross_entropy_loss_fn(z_pred, next_z_dist)
        reward_loss = reward_loss_fn(r_targets, r_pred)
        done_loss = done_loss_fn(d_pred, d_targets.float())

        # Update observation_model and dynamic_model
        dyn_loss = dynamic_loss + 10 * reward_loss + done_loss
        consistency_loss = cross_entropy_loss_fn(next_z_dist, z_pred)
        obs_loss = recon_loss + reg_loss + consistency_loss

        self.dynamic_model_opt.backward(dyn_loss, retain_graph=True)
        # self.observation_model_opt.backward(obs_loss, retain_graph=False)
        self.dynamic_model_opt.update_parameters()
        # self.observation_model_opt.update_parameters()

        return {
            'recon_loss': recon_loss.detach().cpu().item(),
            'reg_loss': reg_loss.detach().cpu().item(),
            'consistency_loss': consistency_loss.detach().cpu().item(),
            'dynamic_loss': dynamic_loss.detach().cpu().item(),
            'reward_loss': reward_loss.detach().cpu().item(),
            'done_loss': done_loss.detach().cpu().item(),
        }

    def load(
            self,
            path,
            name="world-model-checkpoint.pth",
            targets=None
        ):
        checkpoint = torch.load(f'{path}/{name}')
        if targets is None:
            targets = [
                'observation_model',
                'dynamic_model',
                'observation_model_opt',
                'dynamic_model_opt'
            ]
        
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
        if targets is None:
            targets = [
                'observation_model',
                'dynamic_model',
                'observation_model_opt',
                'dynamic_model_opt'
            ]
        
        checkpoint = {
            target: getattr(self, target).state_dict()
            for target in targets
        }
        torch.save(checkpoint, f'{path}/{name}')