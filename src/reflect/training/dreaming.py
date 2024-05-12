from reflect.models.world_model import DynamicsModel
from reflect.models.world_model.embedder import Embedder as Embedder
from reflect.models.world_model.head import Head as Head
from reflect.models.nes_policy.model import NESPolicy, rank_transform
import torch


def train_policy_in_world_model(
        policy: NESPolicy,
        world_model: DynamicsModel,
        z: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        rollout_length: int,
        eps: float=0.1
    ):
    policy.perturb(eps=eps)
    scores = compute_rewards(
        policy,
        world_model,
        z, a, r,
        rollout_length
    )
    scores = scores.sum(1).squeeze()
    scores = rank_transform(scores)
    grads = list(policy.compute_grads(scores, eps=eps))
    for (w_grad, b_grad), layer in zip(grads, policy.layers):
        assert w_grad.shape == layer._weight.shape
        assert b_grad.shape == layer._bias.shape
    policy.update(grads)
    return scores


def compute_rewards(
        policy: NESPolicy,
        world_model: DynamicsModel,
        z: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        rollout_length: int
    ):
    for _ in range(rollout_length):
        z_dist, new_r = world_model((z, a, r))
        new_z = z_dist.sample()[:, -1].reshape(-1, 1, 32 * 32)
        z = torch.cat([z, new_z], dim=1)
        new_r = new_r[:, -1].reshape(-1, 1, 1)
        r = torch.cat([r, new_r], dim=1)
        new_a = policy(z[:, -1], training=True)
        a = torch.cat([a, new_a[:, None]], dim=1)
    return r