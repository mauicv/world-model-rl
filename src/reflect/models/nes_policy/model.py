import torch


def rank_transform(scores):
    score_inds = torch.argsort(scores)
    _scores = torch.argsort(score_inds).float()
    _scores = (_scores ) / (len(_scores) - 1) - 0.5
    return _scores


class NESLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim, population_size) -> None:
        super().__init__()
        self.population_size = population_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer(
            '_weight',
            torch.randn(
                input_dim,
                output_dim
            )
        )
        self.register_buffer(
            '_bias',
            torch.randn(
                output_dim
            )
        )
        self.register_buffer(
            'weight_eps',
            torch.randn(
                population_size,
                input_dim,
                output_dim
            )
        )
        self.register_buffer(
            'bias_eps',
            torch.randn(
                population_size,
                output_dim
            )
        )
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._weight = self._weight.to(*args, **kwargs)
        self._bias = self._bias.to(*args, **kwargs)
        self.weight_eps = self.weight_eps.to(*args, **kwargs)
        self.bias_eps = self.bias_eps.to(*args, **kwargs)
        return self
    
    def forward(self, x, training=True):
        if training:
            assert x.shape[0] == self.population_size
            assert x.shape[1] == self.input_dim
            weight = self._weight[None, :, :] + self.weight_eps
            bias = self._bias[None, :] + self.bias_eps
            x = x.unsqueeze(1)
            output = torch.bmm(x, weight)
            output = output.squeeze(1)
            output += bias
            return output
        else:
            return x @ self._weight + self._bias

    def perturb(self, epsilon):
        self.weight_eps = epsilon * torch.randn_like(self.weight_eps)
        self.bias_eps = epsilon * torch.randn_like(self.bias_eps)

    def update(self, dw, db):
        self._weight += dw
        self._bias += db

    def compute_grads(self, scores, eps):
        dw = (self.weight_eps * scores[:, None, None]).mean(0)
        db = (self.bias_eps * scores[:, None]).mean(0)
        db = db / eps
        dw = dw / eps
        return dw, db


class NESPolicy(torch.nn.Module):
    def __init__(
            self,
            population_size,
            input_dim,
            hidden_dims,
            output_dim,
            output_activation=torch.nn.functional.tanh
        ):
        """NESPolicy

        NES stands for Natural Evolution Strategies. This class is a PyTorch module
        that represents a policy network for a reinforcement learning agent. The
        policy network is trained using the Natural Evolution Strategies
        algorithm.

        Args:
            population_size (int): The number of samples to use in the NES
        """
        super().__init__()
        self.population_size = population_size
        self.output_activation = output_activation
        dims = [input_dim, *hidden_dims, output_dim]
        self.layers: torch.nn.ModuleList[NESLinear] = torch.nn.ModuleList([
            NESLinear(dims[i], dims[i+1], self.population_size)
            for i in range(len(dims) - 1)
        ])

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for layer in self.layers:
            layer.to(*args, **kwargs)
        return self

    def forward(self, x, training=True):
        """forward

        Forward pass through the policy network.

        Args:
            x (torch.Tensor): The input to the policy network.

        Returns:
            torch.Tensor: The output of the policy network.
        """
        if training: assert x.shape[0] == self.population_size
        for model in self.layers[:-1]:
            x = model(x, training=training)
            x = torch.nn.functional.relu(x)
        x = self.layers[-1](x, training=training)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x

    def perturb(self, eps):
        """perturb

        Perturb the parameters of the policy network.

        Args:
            eps (float): The perturbation amount.
        """
        for layer in self.layers:
            layer.perturb(eps)

    def update(self, grads):
        """update

        Update the parameters of the policy network.

        Args:
            grads (List[torch.Tensor]): The gradients to update the parameters of the policy network.
        """
        for layer, grad in zip(self.layers, grads):
            layer.update(*grad)

    def compute_grads(self, scores, eps):
        """Compute the gradients of the policy network.
        
        Args:
            scores (torch.Tensor): The scores of the population of samples.
        """
        for layer in self.layers:
            dw, db = layer.compute_grads(scores, eps)
            yield dw, db