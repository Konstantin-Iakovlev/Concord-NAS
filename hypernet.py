import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class PWNet(nn.Module):
    def __init__(self, size, kernel_num, init_='random'):
        """
        Params:
            size: tuple or int, output size
            kernel_num: int, number of kernels in pivots
            init_: str, type of initialization
        """
        nn.Module.__init__(self)

        if not isinstance(size, tuple):  # check if size is 1d
            size = (size,)

        self.size = size

        full_param_size = np.prod(self.size)
        total_size = [kernel_num] + list(self.size)
        self.kernel_num = kernel_num

        self.const = nn.Parameter(torch.randn(total_size, dtype=torch.float32))
        if init_ == 'random':
            for i in range(kernel_num):
                if len(self.size) > 1:
                    init.kaiming_uniform_(self.const.data[i], a=np.sqrt(5))
                else:

                    self.const.data[i] *= 0
                    self.const.data[i] += torch.randn(size)
        else:
            self.const.data *= 0
            self.const.data += init_
        self.pivots = nn.Parameter(torch.tensor(np.linspace(0, 1, kernel_num)), requires_grad=True)

    def forward(self, lam):
        """
        Params:
            lam: torch.tensor of shape []
        """
        lam_ = lam * 0.99999
        left = torch.floor(lam_ * (self.kernel_num - 1)).long()
        right = left + 1
        dist = (self.pivots[right] - lam_) / (self.pivots[right] - self.pivots[left])
        res = self.const[left] * (dist) + (1.0 - dist) * self.const[right]

        return res


class BasicExpertNet(nn.Module):
    def __init__(self, input_size, output_size):
        '''
            Params:
            input_size: size of an input image (intermediate representation)
            output_size: number of operations corresponding each edge
        '''
        super(BasicExpertNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * input_size, output_size)
        )

    def forward(self, img):
        return self.net(img.mean(dim=1))
        

        


# class PWLinear(nn.Module):
#     def __init__(self, size, kernel_num):
#         nn.Module.__init__(self)
#         self.weight = PWNet(size, kernel_num)
#         self.bias = PWNet(size[1], kernel_num)
#
#     def forward(self, x, lam):
#         weight = self.weight(lam).float()
#         bias = self.bias(lam).float()
#         res = torch.matmul(x, weight) + bias
#         return res
#
#
# class HyperNet(nn.Module):
#     """
#     гиперсеть, управляющая нашей структурой
#     """
#
#     def __init__(self, hidden_layer_num, hidden_size, out_size, kernel_num):
#         """
#         :param hidden_layer_num: количество скрытых слоев (может быть нулевым)
#         :param hidden_size: количество нейронов на скрытом слое (актуально, если скрытые слои есть)
#         """
#         nn.Module.__init__(self)
#         self.out_size = out_size
#         layers = []
#         in_ = 1  # исходная входная размерность
#         for l in range(hidden_layer_num):
#             layers.append(PWNet((in_, hidden_size), kernel_num))
#             layers.append(nn.ReLU())
#             in_ = hidden_size
#         layers.append(PWNet((in_, out_size), kernel_num))
#         # layers.append(nn.Linear(in_, out_size))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         # x --- одномерный вектор (задающий сложность)
#         res = self.model(x).view(1, self.out_size).float()
#         return res
