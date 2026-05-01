from typing import Tuple

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    CNN encoder for pixel observations (Dreamer-style depth scaling).

    input_shape: (C, H, W)
    latent_dim:  output vector dimension
    depth:       base channel multiplier; channels scale as depth, depth*2, depth*4, depth*8

    Accepts (b, C, H, W) → (b, latent_dim)
         or (b, t, C, H, W) → (b, t, latent_dim)

    Uses LazyLinear so any input spatial size is supported without precomputing
    the flattened conv output size.
    """
    def __init__(self, input_shape: Tuple[int, int, int], latent_dim: int, depth: int = 32):
        super().__init__()
        self.input_shape = input_shape
        C = input_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(C,       depth,   4, stride=2), nn.ELU(),
            nn.Conv2d(depth,   depth*2, 4, stride=2), nn.ELU(),
            nn.Conv2d(depth*2, depth*4, 4, stride=2), nn.ELU(),
            nn.Conv2d(depth*4, depth*8, 4, stride=2), nn.ELU(),
        )
        # LazyLinear infers in_features on first forward pass
        self.fc = nn.LazyLinear(latent_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:                            # (b, C, H, W)
            return self._encode(x)
        b, t = x.shape[:2]                          # (b, t, C, H, W)
        return self._encode(x.flatten(0, 1)).unflatten(0, (b, t))
