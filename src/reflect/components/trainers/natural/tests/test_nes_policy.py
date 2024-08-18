import torch
from reflect.components.trainers.natural.model import NESPolicy, NESLinear, rank_transform


def test_nes_linear_layer_train():
    population_size = 3
    input_dim = 4
    output_dim = 10

    nes_linear_layer = NESLinear(
        input_dim=input_dim,
        output_dim=output_dim,
        population_size=population_size
    )
    nes_linear_layer.weight_eps = torch.nn.Parameter(torch.cat([
        torch.zeros((1, input_dim, output_dim)),
        torch.ones((1, input_dim, output_dim)),
        2 * torch.ones((1, input_dim, output_dim))
    ], dim=0))
    nes_linear_layer._weight = torch.nn.Parameter(torch.zeros(input_dim, output_dim))
    nes_linear_layer.bias_eps = torch.nn.Parameter(torch.ones(population_size, output_dim))
    nes_linear_layer._bias = torch.nn.Parameter(torch.zeros(output_dim))
    assert nes_linear_layer.weight_eps.shape == (3, 4, 10)

    x = torch.ones(3, 4)
    y = nes_linear_layer(x)

    assert y.shape == (3, 10)
    assert torch.all(y == torch.tensor([
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
        [9., 9., 9., 9., 9., 9., 9., 9., 9., 9.]
    ]))


def test_nes_linear_layer_non_train():
    population_size = 3
    input_dim = 4
    output_dim = 10
    nes_linear_layer = NESLinear(
        input_dim=input_dim,
        output_dim=output_dim,
        population_size=population_size
    )
    nes_linear_layer.weight = (
        nes_linear_layer
            ._weight[None, :, :]
            .repeat(population_size, 1, 1)
    )
    nes_linear_layer.bias = (
        nes_linear_layer
            ._bias[None, :]
            .repeat(population_size, 1)
    )
    x_all = torch.ones(population_size, 4)
    x_one = torch.ones(1, 4)
    y_one = nes_linear_layer(
        x_one,
        training=False
    )
    y_all = nes_linear_layer(
        x_all,
        training=False
    )
    assert y_one.shape == (1, 10)
    assert y_all.shape == (3, 10)
    for y in y_all:
        assert torch.allclose(y, y_one)
    

def test_nes_policy_forward_train():
    policy = NESPolicy(
        population_size=3,
        input_dim=5,
        hidden_dims=[10, 10],
        output_dim=10
    )
    x = torch.ones(3, 5)
    y = policy(x)
    assert y.shape == (3, 10)


def test_nes_policy_forward_non_train():
    policy = NESPolicy(
        population_size=3,
        input_dim=5,
        hidden_dims=[10, 10],
        output_dim=10
    )
    x = torch.ones(1, 5)
    y = policy(x, training=False)
    assert y.shape == (1, 10)


def test_nes_grads():
    policy = NESPolicy(
        population_size=3,
        input_dim=5,
        hidden_dims=[10, 10],
        output_dim=10
    )
    eps = 1e-6
    policy.perturb(eps)
    scores = torch.randn(3)
    scores = rank_transform(scores)
    grads = list(policy.compute_grads(scores, eps))
    for (w_grad, b_grad), layer in zip(grads, policy.layers):
        assert w_grad.shape == layer._weight.shape
        assert b_grad.shape == layer._bias.shape
    policy.update(grads)
