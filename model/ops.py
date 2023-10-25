import torch
import torch.nn as nn

class Pool(nn.Module):
    def __init__(self, pool_type='max', kernel_size=3, padding=1) -> None:
        super().__init__()
        if pool_type == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, padding=padding, stride=1)
        else:
            self.pool = nn.MaxPool1d(kernel_size=kernel_size, padding=padding, stride=1)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        :param x: tensor of shape (bs, seq_len, hidden)
        :param mask: tensor of shape (bs, seq_len)
        """
        att_mask = mask[..., None]
        out = (x * att_mask).transpose(1, 2)
        return self.pool(out).transpose(1, 2)


class ConvBN(nn.Module):
    def __init__(self, channels, kernel_size, dilation=False) -> None:
        super().__init__()
        padding = ((2 if dilation else 1) * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, 1, padding, 2 if dilation else 1, bias=False)
        # self.bn = nn.LayerNorm(channels)
        self.bn = nn.BatchNorm1d(channels)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        :param x: tensor of shape (bs, seq_len, hidden)
        :param mask: tensor of shape (bs, seq_len)
        """
        out = torch.relu(x)
        att_mask = mask[..., None]
        out = out * att_mask
        out = self.bn(self.conv(out.transpose(1, 2))).transpose(1, 2)
        # out = self.bn(out)
        return out

class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, msk):
        return x


class Zero(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, msk):
        return torch.zeros_like(x)


def test_pool():
    h = torch.ones(1, 3, 10)
    h[:, 1] = 2
    h[:, 2] = 3
    msk = torch.ones(1, 3)
    msk[:, -1] = 0
    pool = Pool('max')
    out = pool(h, msk)
    assert out[:, 0].mean() == 2.0, out[:, 0].mean()
    assert out[:, 1].mean() == 2.0, out[:, 1].mean()


def test_conv():
    conv = ConvBN(10, 3, True)
    h = torch.ones(1, 15, 10)
    msk = torch.ones(1, 15)
    assert h.shape == conv(h, msk).shape, conv(h, msk).shape
    
