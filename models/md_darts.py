from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

from models import ops
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from trainers.md_darts_trainer import DartsInputChoice, DartsLayerChoice
from models.losses import MdTripletLoss

class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """

    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size - 5, padding=0, count_include_pad=False),  # 2x2 out
            nn.Conv2d(C, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False),  # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect,
                 num_domains):
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                LayerChoice(OrderedDict([
                    ("maxpool", ops.PoolBN('max', channels, 3, stride, 1, affine=False, num_domains=num_domains)),
                    ("avgpool", ops.PoolBN('avg', channels, 3, stride, 1, affine=False, num_domains=num_domains)),
                    ("skipconnect", ops.Identity() if stride == 1 else ops.FactorizedReduce(channels, \
                                                                                            channels, affine=False,
                                                                                            num_domains=num_domains)),
                    (
                    "sepconv3x3", ops.SepConv(channels, channels, 3, stride, 1, affine=False, num_domains=num_domains)),
                    (
                    "sepconv5x5", ops.SepConv(channels, channels, 5, stride, 2, affine=False, num_domains=num_domains)),
                    ("dilconv3x3",
                     ops.DilConv(channels, channels, 3, stride, 2, 2, affine=False, num_domains=num_domains)),
                    ("dilconv5x5",
                     ops.DilConv(channels, channels, 5, stride, 4, 2, affine=False, num_domains=num_domains))
                ]), label=choice_keys[-1]))
        self.drop_path = ops.DropPath()
        self.input_switch = InputChoice(n_candidates=len(choice_keys), n_chosen=2, label="{}_switch".format(node_id))

    def forward(self, prev_nodes, domain_idx):
        assert len(self.ops) == len(prev_nodes)
        out = [op(node, domain_idx) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out, domain_idx)

    def concord_loss(self):
        assert isinstance(self.input_switch, DartsInputChoice)
        concord_loss = self.input_switch.concord_loss()
        for op in self.ops:
            assert isinstance(op, DartsLayerChoice)
            concord_loss += op.concord_loss()
        return concord_loss


class Cell(nn.Module):
    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction,
                 num_domains):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(channels_pp, channels, affine=False, \
                                                 num_domains=num_domains)
        else:
            self.preproc0 = ops.StdConv(channels_pp, channels, 1, 1, 0, affine=False, \
                                        num_domains=num_domains)
        self.preproc1 = ops.StdConv(channels_p, channels, 1, 1, 0, affine=False, \
                                    num_domains=num_domains)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth),
                                         depth, channels, 2 if reduction else 0, num_domains))

    def forward(self, s0, s1, domain_idx):
        # s0, s1 are the outputs of previous previous cell and previous cell, respectively.
        tensors = [self.preproc0(s0), self.preproc1(s1)]
        for node in self.mutable_ops:
            cur_tensor = node(tensors, domain_idx)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output

    def concord_loss(self):
        return sum([node.concord_loss() for node in self.mutable_ops])


class CNN(nn.Module):
    def __init__(self, input_size, in_channels, channels, n_classes, n_layers, n_heads=1,
                 n_nodes=4, stem_multiplier=3, drop_path_proba=0.0, auxiliary=False,
                 common_head=True, linear_stem = False):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1
        if linear_stem:
            self.linear_stem = ops.ToyLinear(input_size, n_heads)
        else:
            self.linear_stem = None 

        c_cur = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
        )
        self.stem_bn = nn.ModuleList([nn.BatchNorm2d(c_cur) for _ in range(n_heads)])

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = Cell(n_nodes, channels_pp, channels_p, c_cur, reduction_p, reduction, n_heads)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

            if i == self.aux_pos:
                self.aux_head = nn.ModuleList([AuxiliaryHead(input_size // 4,
                                                             channels_p, n_classes)] for _ in range(n_heads))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.common_head = common_head
        if common_head:
            self.linear = nn.Linear(channels_p, n_classes)
        else:
            self.linear = nn.ModuleList([nn.Linear(channels_p, n_classes) for _ in range(n_heads)])

        self.set_drop_path_proba(drop_path_proba)

    def forward(self, x: torch.Tensor, domain_idx: int) -> Dict[str, torch.Tensor]:
        s0 = s1 = self.stem_bn[domain_idx](self.stem(x))
        if self.linear_stem:
            s0 = s1 = self.linear_stem(s0, domain_idx)

        cell_out_dict = {'hidden_states': [], 'aux_logits': None}

        aux_logits = None
        for i, cell in enumerate(self.cells):
            # cell(s0, s1)
            s0, s1 = s1, cell(s0, s1, domain_idx)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head[domain_idx](s1)
            cell_out_dict['hidden_states'].append(s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        if self.common_head:
            logits = self.linear(out)
        else:
            logits = self.linear[domain_idx](out)

        cell_out_dict['aux_logits'] = aux_logits
        cell_out_dict['logits'] = logits
        return cell_out_dict

    def set_drop_path_proba(self, p: float):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                module.p = p

    def get_drop_path_proba(self):
        for module in self.modules():
            if isinstance(module, ops.DropPath):
                return module.p

    def concord_loss(self):
        for cell in self.cells:
            return cell.concord_loss()
