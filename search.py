# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import time
from argparse import ArgumentParser
from configobj import ConfigObj
import os

import torch
import torch.nn as nn

import datasets
from model import CNN
from nni.nas.pytorch.callbacks import ArchitectureCheckpoint, LRSchedulerCallback
from utils import accuracy

from hyperDARTS import HyperDartsTrainer

logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--config", default='cifar_basic.cfg')
    args = parser.parse_args()

    config = ConfigObj(os.path.join('configs', args.config))
    print(config)

    dataset_train, dataset_valid = datasets.get_dataset(config['dataset'])

    model = CNN(int(config['darts']['input_size']),
            int(config['darts']['input_channels']),
            int(config['darts']['channels']),
            int(config['darts']['n_classes']),
            int(config['darts']['layers']),
            n_nodes=int(config['darts']['n_nodes']),
            stem_multiplier=int(config['darts']['stem_multiplier']))
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), float(config['darts']['optim']['w_lr']),
            momentum=float(config['darts']['optim']['w_momentum']),
            weight_decay=float(config['darts']['optim']['w_weight_decay']))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, int(config['epochs']), eta_min=0.001)
    trainer = HyperDartsTrainer(config['folder_name'],
                                model,
                                loss=criterion,
                                metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                                optimizer=optim,
                                num_epochs=int(config['epochs']),
                                dataset=dataset_train,
                                batch_size=int(config['batch_size']),
                                log_frequency=int(config['log_frequency']),
                                arc_learning_rate=float(config['darts']['optim']['alpha_lr']),
                                arc_weight_decay=float(config['darts']['optim']['alpha_weight_decay']),
                                unrolled=eval(config['darts']['unrolled']),
                                betas=(float(config['darts']['optim']['alpha_beta_1']),
                                    float(config['darts']['optim']['alpha_beta_2'])),
                                sampling_mode=config['darts']['sampling_mode'],
                                t=float(config['darts']['initial_temp'])
                                )
    print('Trainer initialized')
    print('---' * 20)
    trainer.fit()
    # export architecture
    dl = torch.utils.data.DataLoader(dataset_valid, batch_size=64)
    for i, (x, _) in enumerate(dl):
        x = torch.stack([torch.pca_lowrank(x[i].squeeze(), 2)[0] for i in range(64)])
        x = x.view(64, -1)
        final_architecture = trainer.export(x)
        print('Final architecture, regular:', final_architecture)
        json.dump(final_architecture, open(os.path.join('.', 'searchs', 
            config['folder_name'], f'checkpoint_regular_{i}.json'), 'w'))
        if i >= 4:
            break
    
    for i, (x, _) in enumerate(dl):
        x = x.transpose(-1, -2)
        x = torch.stack([torch.pca_lowrank(x[i].squeeze(), 2)[0] for i in range(64)])
        x = x.view(64, -1)
        final_architecture = trainer.export(x)
        print('Final architecture, transposed:', final_architecture)
        json.dump(final_architecture, open(os.path.join('.', 'searchs', 
            config['folder_name'], f'checkpoint_transposed_{i}.json'), 'w'))
        if i >= 4:
            break

