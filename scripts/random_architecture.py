import numpy as np
from argparse import ArgumentParser
import json
import os


OPS = [
    "maxpool",
    "avgpool",
    "skipconnect",
    "sepconv3x3",
    "sepconv5x5",
    "dilconv3x3",
    "dilconv5x5"
]


def get_random_reduce_architecture(n_nodes: int):
    arch = {}
    for i in range(2, 2 + n_nodes):
        for j in range(i):
            op = np.random.choice(OPS)
            arch[f'reduce_n{i}_p{j}'] = op
        arch[f'reduce_n{i}_switch'] = np.random.choice(i, size=2, replace=False).tolist()
    return arch


if __name__ == '__main__':
    parser = ArgumentParser("darts")
    parser.add_argument("--n_nodes", type=int, default=4)
    parser.add_argument("--n_domains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default='.')
    args = parser.parse_args()
    np.random.seed(args.seed)
    architecture = [get_random_reduce_architecture(args.n_nodes) for _ in range(args.n_domains)]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    json.dump(architecture, open(os.path.join(args.save_dir, f'architecture_{args.seed}.json'), 'w'), indent=2)

