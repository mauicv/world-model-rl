"""Data loader for the environment data.

See also: https://colab.research.google.com/drive/10-QQlnSFZeWBC7JCm0mPraGBPLVU2SnS
"""

from torchvision.transforms import Resize, Compose
import torch
import numpy as np


def to_tensor(t):
    if isinstance(t, torch.Tensor):
        return t
    if isinstance(t, np.ndarray):
        return torch.tensor(t.copy(), dtype=torch.float32)
    return torch.tensor(t, dtype=torch.float32)


class Processing:
    def __init__(self, transforms):
        self.transforms = transforms

    def preprocess(self, x):
        raise NotImplementedError

    def postprocess(self, x):
        raise NotImplementedError


class GymRenderImgProcessing(Processing):
    def __init__(
            self,
            transforms=None
        ):
        if transforms is None:
            transforms = Compose([Resize((64, 64))])
        self.transforms = transforms

    def preprocess(self, x):
        x = x.permute(2, 0, 1)
        x = self.transforms(x)
        x = x / 256 - 0.5
        return x

    def postprocess(self, x):
        x = x.permute(1, 2, 0)
        x = (x + 0.5) * 256
        return x


class GymStateProcessing(Processing):
    def __init__(self, transforms=None):
        self.transforms = transforms

    def preprocess(self, x):
        return x

    def postprocess(self, x):
        return x
