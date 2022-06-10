import logging
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
from configobj import ConfigObj

import datasets
from models.sparce_md_darts import SparceMdDartsModel
from utils import has_checkpoint, save_checkpoint, load_checkpoint, accuracy
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup


def evaluate(model, dataloader, domain_idx, metric):
    # TODO: AverageMeter has approx 1e-4 float variance. Use another meter
    model.eval()
    val_meters = AverageMeterGroup()
    n_true = 0
    n_total = 0
    for idx, (X, y) in tqdm(enumerate(dataloader)):
        logits = model(X, domain_idx)
        metrics = metric(logits, y)
        val_meters.update(metrics)
        n_true += (logits.argmax(-1).reshape(-1) == y.reshape(-1)).sum().item()
        n_total += logits.shape[0]
    # return val_meters
    return n_true / n_total

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--checkpoint", default='retrain/mnist_0/checkpoint_14.ckpt')
    parser.add_argument("--config", default='md_main_retrain.cfg')
    args = parser.parse_args()

    config = ConfigObj(args.config)

    _, datasets_valid = datasets.get_datasets(config['datasets'].split(';'), int(config['darts']['input_size']),
                                              int(config['darts']['input_channels']))

    model = SparceMdDartsModel(config, args.checkpoint)

    valid_loaders = []
    for ds in datasets_valid:
        valid_loaders.append(torch.utils.data.DataLoader(ds,
                                                          batch_size=64,
                                                         shuffle=False,
                                                          num_workers=2))

    for domain_idx, (ds_name, dl) in enumerate(zip(config['datasets'].split(';'), valid_loaders)):
        res = evaluate(model, dl, domain_idx, lambda output, target: accuracy(output, target, topk=(1,)))
        print(ds_name, res)
