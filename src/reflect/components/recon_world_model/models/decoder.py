from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDecoder(nn.Module):
    """
    CNN decoder for pixel observations. Mirrors ConvEncoder (Dreamer-style).

    latent_dim:   input vector dimension
    output_shape: (C, H, W) target observation shape
    depth:        base channel multiplier — must match the paired ConvEncoder

    Accepts (b, latent_dim) → (b, C, H, W)
         or (b, t, latent_dim) → (b, t, C, H, W)

    Architecture: linear → reshape to (depth*32, 1, 1) → 4× ConvTranspose2d.
    Designed to invert a ConvEncoder with the same depth. For 64×64 inputs the
    transposed convs land exactly on 64×64; for other sizes a bilinear resize
    corrects any off-by-a-few mismatch.
    """
    def __init__(self, latent_dim: int, output_shape: Tuple[int, int, int], depth: int = 32):
        super().__init__()
        self.output_shape = output_shape
        self.depth = depth
        C = output_shape[0]

        self.fc = nn.Linear(latent_dim, depth * 32)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(depth * 32, depth * 4, 5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(depth * 4,  depth * 2, 5, stride=2), nn.ELU(),
            nn.ConvTranspose2d(depth * 2,  depth,     6, stride=2), nn.ELU(),
            nn.ConvTranspose2d(depth,      C,          6, stride=2),
        )

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        x = self.fc(z).reshape(b, self.depth * 32, 1, 1)
        x = self.deconv(x)
        H, W = self.output_shape[1], self.output_shape[2]
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:                            # (b, latent_dim)
            return self._decode(z)
        b, t = z.shape[:2]                          # (b, t, latent_dim)
        return self._decode(z.flatten(0, 1)).unflatten(0, (b, t))
