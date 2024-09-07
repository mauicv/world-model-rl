from typing import Iterable
from torch.nn import Module
import torch.optim as optim
import torch
import math
import torch.nn.functional as F
import torch.distributions as D
import csv
from dataclasses import dataclass


@dataclass
class AnnealingParams:
    warmup_iter: int = 4e3 * 16
    decay_step: int = 1e5 * 16
    base_lr: int = 2e-4
    end_lr: int = 1e-4
    exp_rate: float = 0.5


def anneal_learning_rate(global_step, params: AnnealingParams):
    if global_step < params.warmup_iter:
        lr = params.base_lr / params.warmup_iter * (global_step)
    else:
        lr = params.base_lr

    lr = lr * params.exp_rate ** (global_step/ params.decay_step)
    if global_step > params.decay_step:
        lr = max(lr, params.end_lr)

    return lr


class AdamOptim:
    def __init__(
            self,
            parameters,
            lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            grad_clip=torch.inf,
            annealing_params=None,
            optimizer='Adam'
        ):
        self.annealing_params = annealing_params
        self.parameters = list(parameters)
        self.grad_clip = grad_clip
        optimizer_cls = getattr(optim, optimizer)
        self.optimizer = optimizer_cls(
            self.parameters,
            lr=lr,
            betas=betas,
            eps=eps
        )

    def backward(self, loss, retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        return grad_norm

    def update_parameters(self, global_step=None):
        if global_step is not None and self.annealing_params is not None:
            lr = anneal_learning_rate(global_step, self.annealing_params)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def to(self, device):
        # see https://github.com/pytorch/pytorch/issues/2830
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


def recon_loss_fn(x, y):
    y_dist = D.Independent(D.Normal(y, torch.ones_like(y)), 3)
    return - y_dist.log_prob(x).mean()


def reg_loss_fn(z_logits, temperature=1):
    z_dist = D.OneHotCategoricalStraightThrough(logits=z_logits / temperature)
    z_dist = D.Independent(z_dist, 1)
    _, a, b = z_dist.base_dist.probs.shape
    max_entropy = a * math.log(b)
    return - z_dist.entropy().mean() / max_entropy


def create_z_dist(logits, temperature=1):
    assert temperature > 0
    dist = D.OneHotCategoricalStraightThrough(logits=logits / temperature)
    return D.Independent(dist, 1)

def cross_entropy_loss_fn(z, z_hat):
    """
    In the case of the observational model, the cross_entropy_loss_fn is the
    consistency loss. In that case the z is the output of the observational
    model for o_(i) and z_hat is the output of the dynamic model from z_(i-1)

    In the case of the dynamic loss, these are the otherway around. So z is
    the output of the dynamic model given z_(i-1) and z_hat is the output of the
    observational model given o_(i).
    """
    cross_entropy = (
        z.base_dist.logits * z_hat.base_dist.probs.detach()
    ).sum(-1)
    return - cross_entropy.sum()


def reward_loss_fn(r, r_pred):
    r_pred_dist = D.Independent(D.Normal(r_pred, torch.ones_like(r_pred)), 1)
    return - r_pred_dist.log_prob(r).mean()


class CSVLogger:
    def __init__(self, fieldnames, path='results.csv'):
        self.fieldnames = fieldnames
        self.path = path
        self.data = []
    
    def initialize(self):
        with open(self.path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        return self

    def log(self, **kwargs):
        self.data.append(kwargs)
        with open(self.path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(kwargs)

    def read(self):
        with open(self.path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def plot(self, plot_map):
        # plot_map is a list of tuples. Each tuple contains 
        # the fields to be plotted together
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(nrows=len(plot_map), figsize=(15, 7))

        all_data = self.read()
        for i, fields in enumerate(plot_map):
            for fieldname in fields:
                data = [float(row[fieldname]) for row in all_data]
                axs[i].plot(data, label=fieldname)
                axs[i].set_title(fields)
                axs[i].legend()
        plt.show()



# see https://github.com/juliusfrost/dreamer-pytorch/blob/main/dreamer/utils/module.py
def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        if module is not None:
            model_parameters += list(module.parameters())
    return model_parameters

def set_eval(modules: Iterable[Module]):
    for module in modules:
        if module is not None:
            module.eval()

def set_train(modules: Iterable[Module]):
    for module in modules:
        if module is not None:
            module.train()


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        set_eval(self.modules)
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_train(self.modules)
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]