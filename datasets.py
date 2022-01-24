# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


class Flip(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return torch.transpose(tensor, -1, -2) 

class Broadcast(object):
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, tensor):
        return tensor.broadcast_to(self.channels, *tensor.shape[1:])


def get_dataset(cls, input_size, channels, cutout_length=0):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    normalize = [
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(input_size),
        Broadcast(channels)] + normalize + cutout)
    valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(input_size), 
        Broadcast(channels)] + normalize)

    # list of pairs: (dataset_train, dataset_valid)
    datasets_train = []
    datasets_valid = []
    if type(cls) != list:
        cls = [cls]
    for ds_name in cls:
        if ds_name == "cifar10":
            print('The dataset is CIFAR-10')
            dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
            dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
        elif ds_name == 'MNIST':
            print('The dataset is MNIST')
            # composed = transforms.Compose([transforms.ToTensor()])
            dataset_train = MNIST(root="./data", train=True, download=True, transform=train_transform)
            dataset_valid = MNIST(root="./data", train=False, download=True, transform=valid_transform)
        elif ds_name == 'MNIST-TR':
            print('The dataset is MNIST-TR')
            dataset_train = MNIST(root="./data", train=True, download=True, 
                    transform=transforms.Compose([train_transform, Flip()]))
            dataset_valid = MNIST(root="./data", train=False, download=True,
                    transform=transforms.Compose([valid_transform, Flip()]))
        elif ds_name == 'SVHN':
            print('dataset is SVHN')
            dataset_train = SVHN(root="./data", split='train', download=True, transform=train_transform)
            dataset_valid = SVHN(root="./data", split='test', download=True, transform=valid_transform)
        else:
            raise NotImplementedError
        datasets_train.append(dataset_train)
        datasets_valid.append(dataset_valid)
    return datasets_train, datasets_valid 
