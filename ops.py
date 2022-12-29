import torch.nn as nn
from genotypes import PRIMITIVES


class Operation(nn.Module):
    def __init__(self, primitive_name: str):
        super().__init__()
        self.act_fn = PRIMITIVES[primitive_name]

    def forward(self, x):
        return self.act_fn(x)

