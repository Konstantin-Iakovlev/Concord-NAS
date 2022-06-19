import json
import logging
from argparse import ArgumentParser
from configobj import ConfigObj
import os

import torch
import torch.nn as nn

import datasets
from models.mpnas import MPNAS
from utils import accuracy

from trainers.mpnas_trainer import MPNASTrainer

logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("mpnas")
    parser.add_argument("--config", default='mpnas_basic.cfg')
    args = parser.parse_args()

    config = ConfigObj(args.config)
    print(config)

    datasets_train, datasets_valid = datasets.get_datasets(config['datasets'].split(';'),
                                                           int(config['mpnas']['input_size']),
                                                           int(config['mpnas']['input_channels']))

    model = MPNAS(int(config['mpnas']['input_size']),
                  int(config['mpnas']['input_channels']),
                  int(config['mpnas']['channels']),
                  int(config['mpnas']['layers']),
                  int(config['mpnas']['nodes_per_layer']),
                  int(config['mpnas']['n_classes']),
                  num_domains=len(datasets_train),
                  learning_rate=float(config['mpnas']['controller']['learning_rate'])
                  )
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.RMSprop(model.supernetwork.parameters(), float(config['mpnas']['optim']['learning_rate']),
                                momentum=float(config['mpnas']['optim']['momentum']),
                                weight_decay=float(config['mpnas']['optim']['weight_decay']))
    # TODO: adjust scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, int(config['epochs']), eta_min=1e-4)
    trainer = MPNASTrainer(config['folder_name'],
                           model,
                           loss=criterion,
                           metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                           optimizer=optim,
                           lr_scheduler=lr_scheduler,
                           num_epochs=int(config['epochs']),
                           datasets=datasets_train,
                           seed=int(config['seed']),
                           batch_size=int(config['batch_size']),
                           log_frequency=int(config['log_frequency']),
                           )
    print('Trainer initialized')
    print('---' * 20)
    trainer.fit()
    # export architecture
    json.dump(trainer.export(), open(os.path.join('searchs', config['folder_name'], 'final_architecture.json'), 'w'))
