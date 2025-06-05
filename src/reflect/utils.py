from typing import Iterable
from torch.nn import Module
from torch.optim import Adam
import torch
import math
import torch.nn.functional as F
import torch.distributions as D
import csv


class AdamOptim:
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, grad_clip=torch.inf):
        self.parameters = list(parameters)
        self.grad_clip = grad_clip
        self.lr = lr
        self.optimizer = Adam(self.parameters, lr=lr, betas=betas, eps=eps)

    def backward(self, loss, retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        return grad_norm

    def update_parameters(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()
    
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def update_lr_by_factor(self, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * factor

    def to(self, device):
        # see https://github.com/pytorch/pytorch/issues/2830
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


def recon_loss_fn(x, y):
    if len(x.shape) == 4:
        x, y = x.permute(0, 2, 3, 1), y.permute(0, 2, 3, 1)
        y_dist = D.Independent(D.Normal(y, torch.ones_like(y)), 3)
    elif len(x.shape) == 2:
        y_dist = D.Independent(D.Normal(y, torch.ones_like(y)), 1)
    else:
        raise ValueError(f"Expected input shape to be 2 or 4, got {len(x.shape)}")
    ts_loss = - y_dist.log_prob(x)
    return ts_loss.mean(), ts_loss.detach()


def reg_loss_fn(z_logits, temperature=1):
    z_dist = D.OneHotCategoricalStraightThrough(logits=z_logits / temperature)
    z_dist = D.Independent(z_dist, 1)
    _, a, b = z_dist.base_dist.probs.shape
    max_entropy = a * math.log(b)
    return - z_dist.entropy().mean() / max_entropy


def create_z_dist(logits, temperature=1):
    assert temperature > 0
    # Add small epsilon to prevent numerical instability
    logits = logits / (temperature + 1e-6)
    # Add small epsilon to prevent exactly zero probabilities
    probs = torch.softmax(logits, dim=-1)
    probs = (probs + 1e-6) / (1.0 + 1e-6 * probs.shape[-1])
    dist = D.OneHotCategoricalStraightThrough(probs=probs)
    return D.Independent(dist, 1)


def kl_divergence_loss_fn(z, z_hat):
    kl = D.kl_divergence(z, z_hat)
    return kl.mean(), kl.detach()


def reward_loss_fn(r, r_pred):
    r_pred_dist = D.Independent(D.Normal(r_pred, torch.ones_like(r_pred)), 1)
    ts_loss = - r_pred_dist.log_prob(r)
    return ts_loss.mean(), ts_loss.detach()


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