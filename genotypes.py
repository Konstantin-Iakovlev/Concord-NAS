from collections import namedtuple
import torch

# recurrent = List[Tuple[previous_node, activation_name]]
# concat = output node is mean of the selected nodes
Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = {
        'none': lambda x: torch.zeros_like(x),
        'tanh': lambda x: torch.tanh(x),
        'relu': lambda x: torch.relu(x),
        'sigmoid': lambda x: torch.sigmoid(x),
        'identity': lambda x: x,
    }
STEPS = 8
CONCAT = 8

ENAS = Genotype(
    recurrent = [
        ('tanh', 0),
        ('tanh', 1),
        ('relu', 1),
        ('tanh', 3),
        ('tanh', 3),
        ('relu', 3),
        ('relu', 4),
        ('relu', 7),
        ('relu', 8),
        ('relu', 8),
        ('relu', 8),
    ],
    concat = [2, 5, 6, 9, 10, 11]
)

DARTS_V1 = Genotype(recurrent=[('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4),
                               ('identity', 1), ('relu', 5), ('relu', 1)], concat=range(1, 9))
DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2),
                               ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))

DARTS = DARTS_V2

