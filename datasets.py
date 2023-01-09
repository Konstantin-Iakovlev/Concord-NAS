import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
from torchvision.transforms.functional import rotate
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import Lambda

from typing import Union, List


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


class Rotate(object):
    def __init__(self, angle: float):
        self.angle = angle

    def __call__(self, tensor):
        return rotate(tensor, self.angle)


class Negative(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return -tensor


class Broadcast(object):
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, tensor):
        return tensor.broadcast_to(self.channels, *tensor.shape[1:])


MINI_SUBSET = 128  # twice # 500: 50/65


def get_dataset(ds_name: str, input_size: int, channels: int, cutout_length: int = 0, seed=0):
    """
    Args:
        ds_name (str): dataset name
            ex. MNIST-180, MNIST-0+MNIST-180
        input_size (int): height and width of the image
        channels (int): number of channels
        cutout_length (int): length of cutout

    Returns:
        dataset_train, dataset_val
    """
    rs = np.random.RandomState(seed)
    if channels == 1:
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
    else:
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

    datasets_train = []
    datasets_test = []
    for sub_ds in ds_name.split('+'):
        if sub_ds == "cifar10":
            dataset_train = CIFAR10(
                root="./data", train=True, download=True, transform=train_transform)
            dataset_test = CIFAR10(
                root="./data", train=False, download=True, transform=valid_transform)
        elif sub_ds.startswith('MNIST'):
            angle = float(sub_ds.split('-')[-1])
            dataset_train = MNIST(root="./data", train=True, download=True,
                                  transform=transforms.Compose([train_transform, Rotate(angle)]))
            ind = list([i for i in range(len(dataset_train))])
            rs.shuffle(ind)
            dataset_train = Subset(dataset_train, ind[:MINI_SUBSET])
            dataset_test = MNIST(root="./data", train=False, download=True,
                                 transform=transforms.Compose([valid_transform, Rotate(angle)]))
        elif sub_ds.startswith('MINI-FMNIST'):
            suffix = sub_ds.split('-')[-1]
            if suffix.startswith('r'):
                angle = int(sub_ds.split('-')[-1][1:])
                train_transform = transforms.Compose(
                    [train_transform, Rotate(angle)])
                valid_transform = transforms.Compose(
                    [valid_transform, Rotate(angle)])
                target_transform = None
                
            dataset_train = FashionMNIST(root="./data", train=True, download=True,
                                         transform=train_transform, target_transform=target_transform)

            ind = list([i for i in range(len(dataset_train))])
            rs.shuffle(ind)
            dataset_train = Subset(dataset_train, ind[:MINI_SUBSET])
            #print (angle, dataset_train[0][0].mean(), dataset_train[0][1])
            dataset_test = FashionMNIST(root="./data", train=False, download=True,
                                        transform=valid_transform, target_transform=target_transform)
            ind = list([i for i in range(len(dataset_test))])
            dataset_test = Subset(dataset_test, ind)

        elif sub_ds == 'SVHN':
            dataset_train = SVHN(root="./data", split='train',
                                 download=True, transform=train_transform)
            dataset_test = SVHN(root="./data", split='test',
                                download=True, transform=valid_transform)
        else:
            raise NotImplementedError
        datasets_train.append(dataset_train)
        datasets_test.append(dataset_test)
    return ConcatDataset(datasets_train), ConcatDataset(datasets_test)


def get_datasets(ds_names: List[str], input_size: int, channels: int, cutout_length: int = 0, seed: int = 0):
    """
    Args:
        ds_names (List[str]):
        input_size (int):
        channels (int):
        cutout_length (int):

    Returns:
        List of pairs of datasets
    """
    train_datasets = []
    test_datasets = []
    for id, ds_name in enumerate(ds_names):
        train_ds, val_ds = get_dataset(
            ds_name, input_size, channels, cutout_length, seed=seed+id)
        train_datasets.append(train_ds)
        test_datasets.append(val_ds)
    return train_datasets, test_datasets
