
import math

import numpy as np
import torch
import torch.nn as nn

__all__ =('Concat2')

class Concat2(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x1, x2):
        return torch.add(x1, x2, self.d)