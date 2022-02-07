# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import torch

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = dict()
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res


def has_checkpoint(ckpt_dir, epoch):
    """ returns True if there is a checkpoint_{epoch}.ckp """
    return f'checkpoint_{epoch}.ckp' in os.listdir(ckpt_dir)


def save_checkpoint(ckpt_dir, epoch, state):
    torch.save(state, os.path.join(ckpt_dir, f'checkpoint_{epoch}.ckp'))


def load_checkpoint(ckpt_dir, epoch, model):
    assert has_checkpoint(ckpt_dir, epoch)
    file = os.path.join(ckpt_dir, f'checkpoint_{epoch}.ckp')
    model.load_state_dict(torch.load(file))

