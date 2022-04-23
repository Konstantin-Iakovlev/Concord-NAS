from argparse import ArgumentParser

import torch
import torch.nn as nn
from configobj import ConfigObj

import datasets
from models.mpnas import MPNAS
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, to_device
import logging
import os
import json
from torch.utils.tensorboard import SummaryWriter

from utils import has_checkpoint, load_checkpoint, save_checkpoint, accuracy


class MPNASRetrainer:
    def __init__(self, arch_path: str, folder_name: str, model: MPNAS, loss, metrics,
                 optimizer: torch.optim.Optimizer, lr_scheduler,
                 num_epochs: int, datasets, seed=0, grad_clip=5.,
                 batch_size=64, workers=2,
                 device=None, log_frequency=None,
                 ):
        self.paths = json.loads(open(arch_path).read())
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.datasets = datasets
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else device
        self.log_frequency = log_frequency
        self._ckpt_dir = os.path.join('retrain', folder_name)
        self._seed = seed
        os.makedirs(os.path.join('retrain', folder_name), exist_ok=True)
        self._logger = logging.getLogger('mpnas')
        self._logger.addHandler(logging.FileHandler(os.path.join('retrain', folder_name,
                                                                 folder_name + '.log')))
        self.writer = SummaryWriter(os.path.join('retrain', folder_name))
        self._step_num = 0

        self.model.to(self.device)

        self.model_optim = optimizer
        self._w = 1.0
        self.lr_scheduler = lr_scheduler
        self.grad_clip = grad_clip

        self._init_dataloaders()
        self._step_num = 0

    def _init_dataloaders(self):
        self.train_loaders = []
        self.valid_loaders = []
        for ds in self.datasets:
            n_train = len(ds)
            # 50% on validation
            split = n_train // 2
            indices = list(range(n_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
            self.train_loaders.append(torch.utils.data.DataLoader(ds,
                                                                  batch_size=self.batch_size,
                                                                  sampler=train_sampler,
                                                                  num_workers=self.workers))
            self.valid_loaders.append(torch.utils.data.DataLoader(ds,
                                                                  batch_size=self.batch_size,
                                                                  sampler=valid_sampler,
                                                                  num_workers=self.workers))

    def _logits_and_loss(self, X, y, path, domain_idx=0):
        logits = self.model(X, path, domain_idx)
        loss = self.loss(logits, y)
        return logits, loss

    def _boosting_fn(self, l: torch.Tensor):
        return torch.exp(l / self._w)

    def _train_one_epoch(self, epoch):
        if has_checkpoint(self._ckpt_dir, epoch):
            load_checkpoint(self._ckpt_dir, epoch, self.model, self.model_optim,
                            self.model.controller.optimizer)
            self._logger.info(f'Loaded checkpoint_{epoch}.ckp')
            return
        trn_meters = [AverageMeterGroup() for _ in range(len(self.datasets))]
        val_meters = [AverageMeterGroup() for _ in range(len(self.datasets))]
        seed = self._seed + epoch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        for step, (train_objects, valid_objects) in enumerate(zip(zip(*self.train_loaders), zip(*self.valid_loaders))):
            self.model_optim.zero_grad()
            paths = self.paths
            rewards = []
            for domain_idx in range(len(self.datasets)):
                trn_X, trn_y = train_objects[domain_idx]
                val_X, val_y = valid_objects[domain_idx]
                trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
                val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)

                # run on the validation set
                self.model.eval()
                with torch.no_grad():
                    logits, loss = self._logits_and_loss(val_X, val_y, paths[domain_idx], domain_idx)
                self.writer.add_scalar(f'Loss/val_{domain_idx}',
                                       loss.item(), self._step_num)
                metrics = self.metrics(logits, val_y)
                metrics['loss'] = loss.item()
                val_meters[domain_idx].update(metrics)
                assert 'acc1' in metrics
                rewards.append(metrics['acc1'])

                # run on the training set
                self.model.train()
                logits, loss = self._logits_and_loss(trn_X, trn_y, paths[domain_idx], domain_idx)
                self._boosting_fn(loss).backward()
                metrics = self.metrics(logits, trn_y)
                metrics['loss'] = loss.item()
                trn_meters[domain_idx].update(metrics)
                self.writer.add_scalar(f'Loss/train_{domain_idx}',
                                       loss.item(), self._step_num)
                if self.log_frequency is not None and step % self.log_frequency == 0:
                    self._logger.info('Epoch [%s/%s] Step [%s/%s], Seed:[%s], Domain: %s\nTrain: %s\nValid: %s',
                                      epoch + 1, self.num_epochs, step + 1,
                                      min([len(loader) for loader in self.train_loaders]),
                                      seed, domain_idx, trn_meters[domain_idx], val_meters[domain_idx])

            # update network
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.supernetwork.parameters(), self.grad_clip)
            self.model_optim.step()

        self.lr_scheduler.step()
        save_checkpoint(self._ckpt_dir, epoch, self.model.state_dict(),
                        self.model_optim.state_dict(), self.model.controller.optimizer.state_dict())

    def fit(self):
        for i in range(self.num_epochs):
            self._train_one_epoch(i)


if __name__ == "__main__":
    parser = ArgumentParser("mpnas_retrain")
    parser.add_argument("--config", default='mpnas_basic_retrain.cfg')
    args = parser.parse_args()

    config = ConfigObj(os.path.join('configs', args.config))
    print(config)
    # print(config['datasets'].split(';'))

    datasets_train, datasets_valid = datasets.get_dataset(config['datasets'].split(';'),
                                                          int(config['mpnas']['input_size']),
                                                          int(config['mpnas']['input_channels']))
    # print(datasets_train[1][0][0].shape, datasets_train[0][0][0].shape)

    model = MPNAS(int(config['mpnas']['input_size']),
                  int(config['mpnas']['input_channels']),
                  int(config['mpnas']['channels']),
                  int(config['mpnas']['layers']),
                  int(config['mpnas']['nodes_per_layer']),
                  int(config['mpnas']['n_classes']),
                  num_domains=len(datasets_train),
                  )
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.RMSprop(model.supernetwork.parameters(), float(config['mpnas']['optim']['learning_rate']),
                                momentum=float(config['mpnas']['optim']['momentum']),
                                weight_decay=float(config['mpnas']['optim']['weight_decay']))
    # TODO: adjust scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, int(config['epochs']), eta_min=1e-4)
    trainer = MPNASRetrainer(config['architecture_path'],
                             config['folder_name'],
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
