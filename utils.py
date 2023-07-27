import torch
import torch.nn as nn
import os, shutil
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import json
from genotypes import PRIMITIVES, STEPS
from typing import Optional


class ConstantSchedulerWithWarmup:
    def __init__(self, warmup_steps: int, value: float) -> None:
        if not warmup_steps >= 0:
            raise ValueError(f'Expected {warmup_steps} >= 0')
        self.warmup_steps = warmup_steps
        self.value = value
        self._step = 0
    
    def get_value(self) -> float:
        if self._step >= self.warmup_steps or self.warmup_steps == 0:
            return self.value
        return self._step / self.warmup_steps * self.value
    
    def step(self):
        self._step += 1


def to_device(a, device):
    if type(a) == torch.Tensor:
        return a.to(device)
    for el in a:
        if type(el) == torch.Tensor:
            el = el.to(device)
        else:
            to_device(el, device)
    return a


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h) 


def batchify(data, bsz, args):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    print(data.size())
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len])
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, epoch, path, finetune=False):
    if finetune:
        torch.save(model.state_dict(), os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        architecture = model.export()
        json.dump(architecture, open(os.path.join(path, 'final_architecture.json'), 'w'))
    torch.save({'epoch': epoch+1}, os.path.join(path, 'misc.pt'))


def embedded_dropout(embed: torch.nn.Embedding, words: torch.LongTensor, dropout: float=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
                1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = F.embedding(words, masked_embed_weight, padding_idx=padding_idx, max_norm=embed.max_norm,
    norm_type=embed.norm_type, scale_grad_by_freq=embed.scale_grad_by_freq, sparse=embed.sparse)
    return X


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


def mask2d(B, D, keep_prob):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    return m


def generate_arch(seed: int, shared_edges: int):
    np.random.seed(seed)
    op_names = list(set(PRIMITIVES.keys()) - {'none'})
    shared_positions = np.random.choice(STEPS, size=(shared_edges,), replace=False)
    arch = [[], []]
    for i in range(1, STEPS + 1):
        if i - 1 in shared_positions:
            prev_node = np.random.choice(i)
            op_name = np.random.choice(op_names)
            arch[0].append([op_name, prev_node])
            arch[1].append([op_name, prev_node])
        else:
            arch[0].append([np.random.choice(op_names), np.random.choice(i)])
            arch[1].append([np.random.choice(op_names), np.random.choice(i)])
    return arch         
