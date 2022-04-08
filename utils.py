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


def save_checkpoint(ckpt_dir, epoch, model_state_dict, model_optim_state_dict,
                    ctrl_optim_state_dict):
    """Saves model and optimizers"""
    # TODO: save scheduler
    torch.save({'model_state_dict': model_state_dict,
                'model_optim_state_dict': model_optim_state_dict,
                'ctrl_optim_state_dict': ctrl_optim_state_dict},
               os.path.join(ckpt_dir, f'checkpoint_{epoch}.ckp'))


def load_checkpoint(ckpt_dir, epoch, model, model_optimizer, ctrl_optimizer):
    """Loads model and optimizers"""
    assert has_checkpoint(ckpt_dir, epoch)
    file = os.path.join(ckpt_dir, f'checkpoint_{epoch}.ckp')
    checkpoint = torch.load(file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model_optimizer.load_state_dict(checkpoint['model_optim_state_dict'])
    if ctrl_optimizer is not None:
        ctrl_optimizer.load_state_dict(checkpoint['ctrl_optim_state_dict'])


def js_divergence(pr_1: torch.tensor, pr_2: torch.tensor):
    """Jensen–Shannon divergence"""
    p_distr = torch.distributions.Categorical(probs=pr_1)
    q_distr = torch.distributions.Categorical(probs=pr_2)
    m_distr = torch.distributions.Categorical(probs=0.5 * (pr_1 + pr_2))
    return 0.5 * torch.distributions.kl.kl_divergence(p_distr, m_distr) + \
           0.5 * torch.distributions.kl.kl_divergence(q_distr, m_distr)
