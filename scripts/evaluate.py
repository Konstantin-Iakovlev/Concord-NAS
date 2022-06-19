import logging
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
from configobj import ConfigObj

import datasets
from models.mpnas import MPNAS
from models.sparce_md_darts import SparceMdDartsModel
from utils import has_checkpoint, save_checkpoint, load_checkpoint, accuracy
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup


@torch.no_grad()
def evaluate(model, dataloader, domain_idx, metric, paths=None):
    # TODO: AverageMeter has approx 1e-4 float variance. Use another meter
    model.eval()
    val_meters = AverageMeterGroup()
    n_true = 0
    n_total = 0
    for idx, (X, y) in tqdm(enumerate(dataloader)):
        if type(model) == SparceMdDartsModel:
            logits = model(X, domain_idx)
        else:
            logits = model(X, paths[domain_idx], domain_idx)
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
    parser.add_argument('--model-name', default='darts')
    args = parser.parse_args()

    config = ConfigObj(args.config)

    _, datasets_valid = datasets.get_datasets(config['datasets'].split(';'), int(config[args.model_name]['input_size']),
                                              int(config[args.model_name]['input_channels']))

    if args.model_name == 'darts':
        model = SparceMdDartsModel(config, args.checkpoint)
        paths = None
    else:
        # TODO: refactor: init with (config, checkpoint)
        model = MPNAS(int(config['mpnas']['input_size']),
                      int(config['mpnas']['input_channels']),
                      int(config['mpnas']['channels']),
                      int(config['mpnas']['layers']),
                      int(config['mpnas']['nodes_per_layer']),
                      int(config['mpnas']['n_classes']),
                      num_domains=len(datasets_valid),
                      learning_rate=0.0
                      )
        model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
        paths = json.loads(open(config['architecture_path']).read())

    valid_loaders = []
    for ds in datasets_valid:
        valid_loaders.append(torch.utils.data.DataLoader(ds,
                                                         batch_size=64,
                                                         shuffle=False,
                                                         num_workers=2))

    for domain_idx, (ds_name, dl) in enumerate(zip(config['datasets'].split(';'), valid_loaders)):
        res = evaluate(model, dl, domain_idx, lambda output, target: accuracy(output, target, topk=(1,)), paths)
        print(ds_name, res)
