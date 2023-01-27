from collections import namedtuple

# recurrent = List[Tuple[previous_node, activation_name]]
# concat = output node is mean of the selected nodes
Genotype = namedtuple('Genotype', 'recurrent concat')

STEPS = 8
CONCAT = 8
INITRANGE = 0.02

