# taken from https://github.com/adityabingi/Dreamer/blob/main/models.py
import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, embed_size=None, activation=nn.ReLU(), depth=32):

        super().__init__()

        self.input_shape = input_shape
        self.act_fn = activation
        self.depth = depth
        self.kernels = [4, 4, 4, 4]
        self.embed_size = embed_size
        
        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = input_shape[0] if i==0 else self.depth * (2 ** (i-1))
            out_ch = self.depth * (2 ** i)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=2))
            layers.append(self.act_fn)

        self.conv_block = nn.Sequential(*layers)

        if self.embed_size is not None:
            self.fc = nn.LazyLinear(self.embed_size)

    def forward_batch(self, inputs: torch.Tensor):
        b, c, h, w = inputs.shape
        assert (c, h, w) == self.input_shape
        embed = self.conv_block(inputs)
        embed = embed.reshape(b, -1)
        if self.embed_size is not None:
            embed = self.fc(embed)
            embed = self.act_fn(embed)
        return embed

    def forward_batch_time(self, inputs: torch.Tensor):
        b, t, c, h, w = inputs.shape
        assert (c, h, w) == self.input_shape
        reshaped = inputs.reshape(b * t, c, h, w)
        embed = self.conv_block(reshaped)
        embed = embed.reshape(b, t, -1)
        if self.embed_size is not None:
            embed = self.fc(embed)
            embed = self.act_fn(embed)
        return embed

    def forward(self, inputs: torch.Tensor):
        if len(inputs.shape) == 4:
            return self.forward_batch(inputs=inputs)
        return self.forward_batch_time(inputs=inputs)
