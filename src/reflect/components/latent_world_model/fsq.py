# taken from https://github.com/aidanscannell/dcmpc/blob/main/utils/layers.py#L49

import torch
from typing import List
#!/usr/bin/env python3
import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ as _FSQ


class FSQ(_FSQ):
    """
    Finite Scalar Quantization
    """

    def __init__(self, levels: List[int]):
        super().__init__(levels=levels)
        self.levels = levels
        self.num_channels = len(levels)

    def forward(self, z):
        shp = z.shape
        z = z.view(*shp[:-1], -1, self.num_channels)
        if z.ndim > 3:  # TODO this might not work for CNN
            codes, indices = torch.func.vmap(super().forward)(z)
        else:
            codes, indices = super().forward(z)
        codes = codes.flatten(-2)
        return {"codes": codes, "indices": indices, "z": z, "state": codes}

    def __repr__(self):
        return f"FSQ(levels={self.levels})"