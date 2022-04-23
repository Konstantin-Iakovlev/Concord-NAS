import logging
import os
import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
from configobj import ConfigObj
from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict
from typing import List, Dict
from nni.retiarii.nn.pytorch import LayerChoice, InputChoice
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, \
    replace_layer_choice, replace_input_choice, to_device
from nni.retiarii.oneshot.pytorch.darts import DartsTrainer

import datasets
from models.md_darts import CNN
from utils import has_checkpoint, save_checkpoint, load_checkpoint, accuracy


class MdDartsSparceLayerChoice(nn.Module):
    def __init__(self, layer_choice, domain_to_name: List[str]):
        """
        param: layer_choice: len of layer_choice <= number of domains
        param: domain_to_name: list of len (num_domains)
        """
        super(MdDartsSparceLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.domain_to_name = domain_to_name

    def forward(self, inputs, domain_idx: int):
        output = self.op_choices[self.domain_to_name[domain_idx]](inputs, domain_idx)
        return output


class MdDartsSparceInputChoice(nn.Module):
    def __init__(self, input_choice, domain_to_chosen: List[List[int]]):
        """
        param: input_choice: input choice
        param: domain_to_chosen: list of chosen indices for each domain
        """
        super(MdDartsSparceInputChoice, self).__init__()
        self.name = input_choice.label
        self.n_chosen = input_choice.n_chosen or 1
        assert len(domain_to_chosen[0]) == self.n_chosen
        self.domain_to_chosen = domain_to_chosen

    def forward(self, inputs, domain_idx: int):
        inputs = torch.stack(inputs)
        output = inputs[self.domain_to_chosen[domain_idx]].mean(dim=0)
        return output


class MdDartsRetrainer(DartsTrainer):
    def __init__(self, arch_path: str, folder_name, model, loss, metrics, optimizer, lr_scheduler,
                 num_epochs, datasets, seed=0, grad_clip=5.,
                 batch_size=64, workers=0,
                 device=None, log_frequency=None,
                 ):
        self.architectures = json.loads(open(arch_path).read())
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
        self._logger = logging.getLogger('darts')
        self._logger.addHandler(logging.FileHandler(os.path.join('retrain', folder_name,
                                                                 folder_name + '.log')))
        self.writer = SummaryWriter(os.path.join('retrain', folder_name))
        self._step_num = 0

        self.model.to(self.device)

        # initialize sparce model
        def apply_layer_choice(m):
            for name, child in m.named_children():
                if isinstance(child, LayerChoice):
                    setattr(m, name, MdDartsSparceLayerChoice(child, [a[child.key] for a in self.architectures]))
                else:
                    apply_layer_choice(child)

        def apply_input_choice(m):
            for name, child in m.named_children():
                if isinstance(child, InputChoice):
                    setattr(m, name, MdDartsSparceInputChoice(child, [a[child.key] for a in self.architectures]))
                else:
                    apply_input_choice(child)

        apply_layer_choice(self.model)
        apply_input_choice(self.model)

        self.model_optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_clip = grad_clip
        # TODO: Theoretical comparison: proposed optimization vs boosting functions
        self.p = torch.tensor([1 / len(self.datasets)] * len(self.datasets)).to(self.device)

        self._init_dataloaders()

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

    def _logits_and_loss(self, X, y):
        domain_idx = self.curr_domain

        logits = self.model(X, domain_idx)
        loss = self.loss(logits, y)
        return logits, loss * self.p[domain_idx]

    def _train_one_epoch(self, epoch):
        if has_checkpoint(self._ckpt_dir, epoch):
            load_checkpoint(self._ckpt_dir, epoch, self.model, self.model_optim,
                            None)
            self._logger.info(f'Loaded checkpoint_{epoch}.ckp')
            return
        self.model.train()
        trn_meters = [AverageMeterGroup() for _ in range(len(self.datasets))]
        val_meters = [AverageMeterGroup() for _ in range(len(self.datasets))]
        seed = self._seed + epoch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        for step, (train_objects, valid_objects) in enumerate(zip(zip(*self.train_loaders), zip(*self.valid_loaders))):
            self.model_optim.zero_grad()
            for domain_idx in range(len(self.datasets)):
                trn_X, trn_y = train_objects[domain_idx]
                val_X, val_y = valid_objects[domain_idx]
                trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
                val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)

                # save current domain
                self.curr_domain = domain_idx

                logits, loss = self._logits_and_loss(trn_X, trn_y)
                self.writer.add_scalar(f'Loss/train_{domain_idx}', loss.item() / self.p[domain_idx],
                                       self._step_num)
                loss.backward()

                metrics = self.metrics(logits, trn_y)
                metrics['loss'] = loss.item() / self.p[domain_idx]
                trn_meters[domain_idx].update(metrics)

                # validate
                self.model.eval()
                with torch.no_grad():
                    logits, loss = self._logits_and_loss(val_X, val_y)
                    self.writer.add_scalar(f'Loss/val_{domain_idx}',
                                           loss.item() / self.p[domain_idx], self._step_num)
                    metrics = self.metrics(logits, val_y)
                    metrics['loss'] = loss.item() / self.p[domain_idx]
                    val_meters[domain_idx].update(metrics)
                    # update p[domain_idx] using validation loss
                    self.p[domain_idx] *= torch.exp(loss / self.p[domain_idx] * 0.01)

                if self.log_frequency is not None and step % self.log_frequency == 0:
                    self._logger.info('Epoch [%s/%s] Step [%s/%s], Seed:[%s], Domain: %s\nTrain: %s\nValid: %s',
                                      epoch + 1, self.num_epochs, step + 1,
                                      min([len(loader) for loader in self.train_loaders]),
                                      seed, domain_idx, trn_meters[domain_idx], val_meters[domain_idx])

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # perform a step after calculating loss on each domain
            self.model_optim.step()

            self.p /= self.p.sum()
            for i in range(self.p.shape[0]):
                self.writer.add_scalar(f'P_{i}', self.p[i].item(), self._step_num)
            self._step_num += 1

        self.lr_scheduler.step()
        save_checkpoint(self._ckpt_dir, epoch, self.model.state_dict(),
                        self.model_optim.state_dict(), None)


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--config", default='md_main_retrain.cfg')
    args = parser.parse_args()

    config = ConfigObj(os.path.join('configs', args.config))
    # print(config)
    # print(config['datasets'].split(';'))

    datasets_train, datasets_valid = datasets.get_dataset(config['datasets'].split(';'),
                                                          int(config['darts']['input_size']),
                                                          int(config['darts']['input_channels']))
    # print(datasets_train[1][0][0].shape, datasets_train[0][0][0].shape)

    model = CNN(int(config['darts']['input_size']),
                int(config['darts']['input_channels']),
                int(config['darts']['channels']),
                int(config['darts']['n_classes']),
                int(config['darts']['layers']),
                n_heads=len(datasets_train),
                n_nodes=int(config['darts']['n_nodes']),
                stem_multiplier=int(config['darts']['stem_multiplier']))
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), float(config['darts']['optim']['w_lr']),
                            momentum=float(config['darts']['optim']['w_momentum']),
                            weight_decay=float(config['darts']['optim']['w_weight_decay']))
    # TODO: adjust scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, int(config['epochs']), eta_min=0.025)

    trainer = MdDartsRetrainer(config['architecture_path'],
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
