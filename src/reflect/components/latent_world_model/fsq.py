# taken from https://github.com/aidanscannell/iqrl/tree/main

import torch
from typing import List
#!/usr/bin/env python3
import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import utils
from torch.func import functional_call, stack_module_state
from torch.linalg import cond, matrix_rank
from vector_quantize_pytorch import FSQ as _FSQ


class FSQ(torch.nn.Module):
    """
    Finite Scalar Quantization
    """

    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.num_channels = len(levels)
        self._fsq = _FSQ(levels)

    def forward(self, z):
        shp = z.shape
        z = z.view(*shp[:-1], -1, self.num_channels)
        if z.ndim > 3:  # TODO this might not work for CNN
            codes, indices = torch.func.vmap(self._fsq)(z)
        else:
            codes, indices = self._fsq(z)
        return {
            "codes": codes,
            "codes_flat": codes.flatten(-2),
            "indices": indices,
            "z": z,
            "state": codes.flatten(-2),
        }

    def __repr__(self):
        return f"FSQ(levels={self.levels})"