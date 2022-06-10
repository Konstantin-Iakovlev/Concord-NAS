import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mpnas import MPNAS
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, to_device
import logging
import os
from torch.utils.tensorboard import SummaryWriter

from utils import has_checkpoint, load_checkpoint, save_checkpoint


class MPNASTrainer:
    def __init__(self, folder_name: str, model: MPNAS, loss, metrics, optimizer: torch.optim.SGD, lr_scheduler,
                 num_epochs: int, datasets, seed=0, grad_clip=5.,
                 batch_size=64, workers=2,
                 device=None, log_frequency=None,
                 ):
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
        self._ckpt_dir = os.path.join('searchs', folder_name)
        self._seed = seed
        os.makedirs(os.path.join('searchs', folder_name), exist_ok=True)
        self._logger = logging.getLogger('mpnas')
        self._logger.addHandler(logging.FileHandler(os.path.join('searchs', folder_name,
                                                                 folder_name + '.log')))
        self.writer = SummaryWriter(os.path.join('searchs', folder_name))
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
            np.random.seed(0)
            indices = np.random.permutation(np.arange(n_train))
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
            # sample paths for each domain
            paths = self.model.controller.sample()
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

            # update supernetwork
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.supernetwork.parameters(), self.grad_clip)
            self.model_optim.step()

            # update controllers
            self.model.controller.update(torch.tensor(rewards).to(self.device), paths)
            self._step_num += 1

        self.lr_scheduler.step()
        save_checkpoint(self._ckpt_dir, epoch, self.model.state_dict(),
                        self.model_optim.state_dict(), self.model.controller.optimizer.state_dict())

    def fit(self):
        for i in range(self.num_epochs):
            self._train_one_epoch(i)

    @torch.no_grad()
    def export(self):
        return self.model.controller.sample(greedy=True)
