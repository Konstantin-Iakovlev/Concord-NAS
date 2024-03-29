import json
import logging
from argparse import ArgumentParser
from configobj import ConfigObj
import os

import torch
import torch.nn as nn

import datasets
from models.md_darts import CNN
from utils import accuracy

from trainers.md_darts_trainer import MdDartsTrainer

logger = logging.getLogger('nni')

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument(
        "--config", default='configs/md_mnist_basic/md_darts.cfg')
    args = parser.parse_args()

    config = ConfigObj(args.config)
    print(config)
    for seed in config['seed'].split(';'):
        seed = int(seed)

        datasets_train, datasets_test = datasets.get_datasets(config['datasets'].split(';'),
                                                              int(config['darts']
                                                                  ['input_size']),
                                                              int(config['darts']
                                                                  ['input_channels']),
                                                              int(config['cutout_length']),
                                                              seed=seed)
        # for d in datasets_train:
        #    print (next(iter(datasets_train))[0][0].mean())
        # continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = CNN(int(config['darts']['input_size']),
                    int(config['darts']['input_channels']),
                    int(config['darts']['channels']),
                    int(config['darts']['n_classes']),
                    int(config['darts']['layers']),
                    n_heads=len(datasets_train),
                    n_nodes=int(config['darts']['n_nodes']),
                    stem_multiplier=int(config['darts']['stem_multiplier']),
                    drop_path_proba=float(config['darts']['drop_path_proba_init']),
                common_head=bool(config['darts']['common_head']),
                linear_stem=True)
        criterion = nn.CrossEntropyLoss()

        optim = torch.optim.SGD(model.parameters(), float(config['darts']['optim']['w_lr']),
                                momentum=float(
                                    config['darts']['optim']['w_momentum']),
                                weight_decay=float(config['darts']['optim']['w_weight_decay']))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, int(config['epochs']),
                                                                  eta_min=float(config['darts']['optim']['w_lr_min']))
        folder_name = config['folder_name']+'_'+str(seed)
        trainer = MdDartsTrainer(folder_name,
                                 model,
                                 loss=criterion,
                                 metrics=lambda output, target: accuracy(
                                     output, target, topk=(1,)),
                                 optimizer=optim,
                                 lr_scheduler=lr_scheduler,
                                 workers=0,  # os.cpu_count(),
                                 num_epochs=int(config['epochs']),
                                 datasets=datasets_train,
                                 test_datasets=datasets_test,
                                 seed=seed,
                                 concord_coeff=float(
                                     config['darts']['concord_coeff']),
                                 contrastive_coeff=float(
                                     config['darts']['contrastive_coeff']),
                                 batch_size=int(config['batch_size']),
                                 log_frequency=int(config['log_frequency']),
                                 arc_learning_rate=float(
                                     config['darts']['optim']['alpha_lr']),
                                 arc_weight_decay=float(
                                     config['darts']['optim']['alpha_weight_decay']),
                                 unrolled=eval(config['darts']['unrolled']),
                                 eta_lr=float(
                                     config['darts']['optim']['eta_lr']),
                                 betas=(float(config['darts']['optim']['alpha_beta_1']),
                                        float(config['darts']['optim']['alpha_beta_2'])),
                                 sampling_mode=config['darts']['sampling_mode'],
                                 t=float(config['darts']['initial_temp']),
                                 delta_t=float(config['darts']['delta_t']),
                                 drop_path_proba_delta=float(
                                     config['darts']['drop_path_proba_delta'])
                                 )
        print(f'Trainer initialized for seed {seed}')
        print('---' * 20)
        trainer.fit()
        trainer.final_eval()
        trainer.clear_logger()

        architectures = []
        for i in range(len(config['datasets'].split(';'))):
            final_architecture = trainer.export(i)
            architectures.append(final_architecture)
            print(f'final_architecture_{i}\n', final_architecture)
        json.dump(architectures, open(os.path.join('searchs', folder_name, 'final_architecture.json'), 'w'),
                  indent=2)
