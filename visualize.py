import torch
import torch.nn as nn

import os
from argparse import ArgumentParser

from nni.nas.pytorch.fixed import FixedArchitecture
from nni.nas.pytorch.utils import to_list
import json

from graphviz import Digraph


def get_architecture(human_arc):
    # convert from an exported architecture
    result_arc = {k: to_list(v) for k, v in human_arc.items()}  # there could be tensors, numpy arrays, etc.
    # First, convert non-list to list, because there could be {"op1": 0} or {"op1": "conv"},
    # which means {"op1": [0, ]} ir {"op1": ["conv", ]}
    result_arc = {k: v if isinstance(v, list) else [v] for k, v in result_arc.items()}
    return result_arc


def plot(genotype, file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor='lightblue')

    for i, edges in enumerate(genotype):
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)


if __name__ == "__main__":
    parser = ArgumentParser("architecture")
    parser.add_argument('--dir', default='basic_mnist')
    parser.add_argument("--architecture", default='mnist_basic.json')
    args = parser.parse_args()
    
    with open(os.path.join('searchs', args.dir, args.architecture), 'r') as inp:
        architecture = json.load(inp)
    architecture = get_architecture(architecture)
    architecture = {key: architecture[key][0] for key in architecture if 'switch' not in key} 
    arch_normal = {key: architecture[key] for key in architecture if 'normal' in key}
    arch_reduce = {key: architecture[key] for key in architecture if 'reduce' in key}
    n_nodes = max([int(key.split('_')[1][1:]) for key in architecture]) - 1
    print(n_nodes)
    print(arch_normal, arch_reduce)
    # construct DAG
    genotype_reduce = [[] for _ in range(n_nodes)]
    for key in arch_reduce:
        n, p = list(map(lambda s: int(s[1:]), key.split('_')[1:]))
        genotype_reduce[n - 2].append((arch_reduce[key], p))
    print(genotype_reduce)
    plot(genotype_reduce, os.path.join('cells', args.architecture.split('.')[0]))




