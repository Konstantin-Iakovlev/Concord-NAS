from collections import namedtuple

# recurrent = List[Tuple[previous_node, activation_name]]
# concat = output node is mean of the selected nodes
Genotype = namedtuple('Genotype', 'recurrent concat')

STEPS = 8
CONCAT = 8
INITRANGE = 0.02

DARTS_V1 = [('relu', 0), ('relu', 1), ('tanh', 2), ('relu', 3), ('relu', 4), ('identity', 1), ('relu', 5), ('relu', 1)]
