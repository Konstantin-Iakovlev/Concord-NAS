# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

class ToyLinear(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size
        self.linear = torch.nn.Linear(size*size, size*size)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.linear(x)
        return x.view(x.shape[0], x.shape[1], self.size, self.size)
    
class DropPath(nn.Module):
    def __init__(self, p=0.):
        """
        Drop path with probability.

        Parameters
        ----------
        p : float
            Probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), 1, 1, 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask

        return x


class Identity(nn.Module):
    """
    Identity operation
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, domain_idx=0):
        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool with BN. `pool_type` must be `max` or `avg`.
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True,
            num_domains=1):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.ModuleList([nn.BatchNorm2d(C, affine=affine) for _ in range(num_domains)])

    def forward(self, x, domain_idx=0):
        out = self.pool(x)
        out = self.bn[domain_idx](out)
        return out


class StdConv(nn.Module):
    """
    Standard conv: ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True,
            num_domains=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
        )
        self.bn = nn.ModuleList([nn.BatchNorm2d(C_out, affine=affine) for _ in range(num_domains)])

    def forward(self, x, domain_idx=0):
        return self.bn[domain_idx](self.net(x))


class FacConv(nn.Module):
    """
    Factorized conv: ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True,
            num_domains=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
        )
        self.bn = nn.ModuleList([nn.BatchNorm2d(C_out, affine=affine) for _ in range(num_domains)])

    def forward(self, x, domain_idx=0):
        return self.bn[domain_idx](self.net(x))


class DilConv(nn.Module):
    """
    (Dilated) depthwise separable conv.
    ReLU - (Dilated) depthwise separable - Pointwise - BN.
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 9x9 receptive field.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True,
            num_domains=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
        )
        self.bn = nn.ModuleList([nn.BatchNorm2d(C_out, affine=affine) for _ in range(num_domains)])

    def forward(self, x, domain_idx=0):
        return self.bn[domain_idx](self.net(x))


class SepConv(nn.Module):
    """
    Depthwise separable conv.
    DilConv(dilation=1) * 2.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True,
            num_domains=1):
        super().__init__()
        self.conv_1 = DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine,
            num_domains=num_domains)
        self.conv_2 = DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine,
            num_domains=num_domains)

    def forward(self, x, domain_idx=0):
        return self.conv_2(self.conv_1(x, domain_idx=domain_idx), domain_idx=domain_idx)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise (stride=2).
    """
    def __init__(self, C_in, C_out, affine=True, num_domains=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(C_out, affine=affine) for _ in range(num_domains)])

    def forward(self, x, domain_idx=0):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn[domain_idx](out)
        return out
