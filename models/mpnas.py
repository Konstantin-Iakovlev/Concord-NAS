import torch
import torch.nn as nn
import models.ops as ops

from typing import List, Tuple

from models.controller import Controller


class MPNAS(nn.Module):
    def __init__(self, input_size, input_channels, channels, layers,
                 nodes_per_layer, n_classes=10, num_domains=1, learning_rate=3e-4):
        super(MPNAS, self).__init__()
        self.n_layers = layers
        self.preproc = nn.Conv2d(input_channels, channels, 3, 1, 1, bias=False)
        self.preproc_bn = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(num_domains)])
        self.supernetwork = nn.ModuleDict()
        for layer_idx in range(1, layers + 1):
            for from_idx in range(nodes_per_layer):
                for to_idx in range(nodes_per_layer):
                    self.supernetwork.update({
                        f'maxpool_{layer_idx}_{from_idx}_{to_idx}': ops.PoolBN('max', channels, 3, 1, 1, affine=False,
                                                                               num_domains=num_domains)
                    })
                    self.supernetwork.update({
                        f'avgpool_{layer_idx}_{from_idx}_{to_idx}': ops.PoolBN('avg', channels, 3, 1, 1, affine=False,
                                                                               num_domains=num_domains)
                    })
                    self.supernetwork.update({
                        f'skipconnect_{layer_idx}_{from_idx}_{to_idx}': ops.Identity()
                    })
                    self.supernetwork.update({
                        f'sepconv3x3_{layer_idx}_{from_idx}_{to_idx}': ops.SepConv(channels, channels, 3, 1, 1,
                                                                                   affine=False, num_domains=num_domains)
                    })
                    self.supernetwork.update({
                        f'sepconv5x5_{layer_idx}_{from_idx}_{to_idx}': ops.SepConv(channels, channels, 5, 1, 2,
                                                                                   affine=False, num_domains=num_domains)
                    })
                    self.supernetwork.update({
                        f'dilconv3x3_{layer_idx}_{from_idx}_{to_idx}': ops.DilConv(channels, channels, 3, 1, 2, 2,
                                                                                   affine=False, num_domains=num_domains)
                    })
                    self.supernetwork.update({
                        f'dilconv5x5_{layer_idx}_{from_idx}_{to_idx}': ops.DilConv(channels, channels, 5, 1, 4, 2,
                                                                                   affine=False, num_domains=num_domains)
                    })
                    self.drop_path = ops.DropPath()

        self.gap = nn.AdaptiveAvgPool2d(4)
        self.linear = nn.ModuleList([nn.Linear(channels * 4 * 4, n_classes) for _ in range(num_domains)])

        prim_names = list(set([op.split('_')[0] for op in self.supernetwork]))
        self.controller = Controller(layers, nodes_per_layer, prim_names, num_domains=num_domains,
                                     learning_rate=learning_rate)

    def forward(self, x: torch.tensor, path: List[Tuple[int, int, str]], domain_idx=0):
        assert len(path) == self.n_layers
        x = self.preproc_bn[domain_idx](self.preproc(x))
        for i, (from_idx, to_idx, name) in enumerate(path):
            x = self.supernetwork.get_submodule(f'{name}_{i + 1}_{from_idx}_{to_idx}')(x)
        # x of shape (batch_size, channels, inp_size, inp_size)
        x = self.gap(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear[domain_idx](x)
        return x
