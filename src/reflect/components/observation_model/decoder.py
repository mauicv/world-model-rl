# taken from https://github.com/adityabingi/Dreamer/blob/main/models.py

import torch
import torch.nn as nn
import torch.distributions as distributions


class ConvDecoder(nn.Module):
    def __init__(self,
            input_size,
            output_shape,
            activation=nn.ReLU(),
            depth=32,
            output_activation=nn.Tanh()
        ):

        super().__init__()

        self.output_shape = output_shape
        self.depth = depth
        self.kernels = [5, 5, 6, 6]
        self.act_fn = activation
        self.ouput_act = output_activation
        self.input_size = input_size
        
        self.dense = nn.Linear(input_size, 32*self.depth)


        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = 32*self.depth if i==0 else self.depth * (2 ** (len(self.kernels)-1-i))
            out_ch = output_shape[0] if i== len(self.kernels)-1 else self.depth * (2 ** (len(self.kernels)-2-i))
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2))
            if i!=len(self.kernels)-1:
                layers.append(self.act_fn)

        self.convtranspose = nn.Sequential(*layers)

    # TODO: This is for the RSSM world model, once testing is done
    # Need to refactor to have a single forward method

    def forward_batch(self, features):
        b, *_ = features.shape
        features = features.reshape(-1, self.input_size)
        out = self.dense(features)
        out = torch.reshape(out, [-1, 32*self.depth, 1, 1])
        out = self.convtranspose(out)
        out = torch.reshape(out, (b, *self.output_shape))
        return self.ouput_act(out) * 0.5

    def forward_batch_time(self, features):
        print('features.shape 0', features.shape)
        b, t, *_ = features.shape
        features = features.reshape(-1, self.input_size)
        print('features.shape 1', features.shape)
        out = self.dense(features)
        print('out.shape 0', out.shape)
        out = torch.reshape(out, [-1, 32*self.depth, 1, 1])
        print('out.shape 1', out.shape)
        out = self.convtranspose(out)
        print('out.shape 2', out.shape)
        out = torch.reshape(out, (b, t, *self.output_shape))
        print('out.shape 3', out.shape)
        return self.ouput_act(out) * 0.5

    def forward(self, features):
        if len(features.shape) == 3:
            return self.forward_batch_time(features=features)
        return self.forward_batch(features=features)

    # def forward(self, features):
    #     b, t, *_ = features.shape
    #     features = features.reshape(-1, self.input_size)
    #     out = self.dense(features)
    #     out = torch.reshape(out, [-1, 32*self.depth, 1, 1])
    #     out = self.convtranspose(out)
    #     out = torch.reshape(out, (b, t, *self.output_shape))
    #     return self.ouput_act(out) * 0.5

    # def forward(self, features):
    #     b, *_ = features.shape
    #     features = features.reshape(-1, self.input_size)
    #     out = self.dense(features)
    #     out = torch.reshape(out, [-1, 32*self.depth, 1, 1])
    #     out = self.convtranspose(out)
    #     out = torch.reshape(out, (b, *self.output_shape))
    #     return self.ouput_act(out) * 0.5