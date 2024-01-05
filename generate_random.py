import json
import numpy as np
from argparse import ArgumentParser
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n_nodes', type=int, required=False, default=3)
    parser.add_argument('--n_domains', type=int, required=False, default=1)
    parser.add_argument('--is_single', action='store_true')
    args = parser.parse_args()

    op_names = ('maxpool', 'avgpool', 'skipconnect', 'conv3x3', 'conv5x5', 
                'conv7x7', 'dilconv3x3', 'dilconv5x5', 'dilconv7x7')#, 'zero')
    np.random.seed(args.seed)
    final_arch = []
    for _ in range(args.n_domains):
        genotype = []
        for n in range(args.n_nodes):
            edges = np.random.choice(n + 2, size=(2,), replace=False)
            operations = np.random.choice(op_names, size=(2,), replace=True)
            genotype.append([(operations[0], int(edges[0])), (operations[1], int(edges[1]))])
        if not args.is_single or len(final_arch) == 0:
            final_arch.append(genotype)
        else:
            final_arch.append(final_arch[-1])
    
    with open(os.path.join('random_arch', f'arch_{args.seed}.json'), 'w') as f:
        f.write(json.dumps(final_arch))


if __name__ == '__main__':
    main()

