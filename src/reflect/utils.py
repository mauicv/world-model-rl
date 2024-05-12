
from torch.optim import Adam
import torch
import math
import torch.nn.functional as F
import torch.distributions as D
import csv


class AdamOptim:
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_clip=0):
        self.parameters = list(parameters)
        self.grad_clip = grad_clip
        self.optimizer = Adam(self.parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self, loss, retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        self.optimizer.step()


def recon_loss_fn(x, y):
    x, y = x.permute(0, 2, 3, 1), y.permute(0, 2, 3, 1)
    y_dist = D.Independent(D.Normal(y, torch.ones_like(y)), 3)
    return - y_dist.log_prob(x).mean()


def reg_loss_fn(z_dist):
    _, a, b = z_dist.base_dist.probs.shape
    max_entropy = a * math.log(b)
    return - z_dist.entropy().mean() / max_entropy


def cross_entropy_loss_fn(z, z_hat):
    """
    In the case of the observational model, the cross_entropy_loss_fn is the 
    consistency loss. In that case the z is the output of the observational 
    model for o_(i) and z_hat is the output of the dynamic model from z_(i-1)

    In the case of the dynamic loss, these are the otherway around. So z is
    the output of the dynamic model given z_(i-1) and z_hat is the output of the
    observational model given o_(i).
    """
    cross_entropy = (z.base_dist.logits * z_hat.base_dist.probs).sum(-1)
    return cross_entropy.mean()


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
