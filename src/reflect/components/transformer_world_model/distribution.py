import torch.distributions as D
import torch


def create_z_dist(logits, temperature=1):
    assert temperature > 0
    dist = D.OneHotCategoricalStraightThrough(logits=logits / temperature)
    return D.Independent(dist, 1)


def create_norm_dist(mean, std=None):
    if std is not None:
        return D.Independent(D.Normal(mean, std), 1)
    return D.Independent(D.Normal(mean, torch.ones_like(mean)), 1)
