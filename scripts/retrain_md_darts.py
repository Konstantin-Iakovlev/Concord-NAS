import logging
import os
import json
import sys
from argparse import ArgumentParser

import torch
import torch.nn as nn
from configobj import ConfigObj
from torch.utils.tensorboard import SummaryWriter

from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, \
    replace_layer_choice, replace_input_choice, to_device
from nni.retiarii.oneshot.pytorch.darts import DartsTrainer

import datasets
from utils import has_checkpoint, save_checkpoint, load_checkpoint, accuracy, contrastive_loss

from models.sparce_md_darts import SparceMdDartsModel
import numpy as np


class MdDartsRetrainer(DartsTrainer):
    def __init__(self, arch_path: str, folder_name, model: SparceMdDartsModel, loss, metrics, optimizer, lr_scheduler,
                 num_epochs, datasets, test_datasets, seed=0, grad_clip=5., eta_lr=0.01,
                 batch_size=64, workers=os.cpu_count(),
                 device=None, log_frequency=None,
                 drop_path_proba_delta=0.0,
                 contrastive_coeff  = 0.0
                 ):
        self.architectures = json.loads(open(arch_path).read())
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.datasets = datasets
        self.test_datasets = test_datasets
        self.eta_lr = eta_lr
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else device
        self.log_frequency = log_frequency
        self._ckpt_dir = os.path.join('retrain', folder_name)
        self._seed = seed
        os.makedirs(os.path.join('retrain', folder_name), exist_ok=True)
        
        self._logger = logging.getLogger('darts')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')        
        handler = logging.FileHandler(os.path.join('retrain', folder_name,
                                                                 folder_name + '.log'))
        handler.setFormatter(formatter)                                                                 
        self._logger.setLevel(logging.DEBUG)                                                                 
        self._logger.addHandler(handler)        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)   
        self.writer = SummaryWriter(os.path.join('retrain', folder_name))
        self._step_num = 0

        self.model.to(self.device)

        self.model_optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_clip = grad_clip
        # TODO: Theoretical comparison: proposed optimization vs boosting functions
        self.p = torch.tensor([1 / len(self.datasets)] * len(self.datasets)).to(self.device)
        self.contrastive_coeff = contrastive_coeff
        self._init_dataloaders()
        self.drop_path_proba_delta = drop_path_proba_delta

    def _init_dataloaders(self):
        self.train_loaders = []
        self.valid_loaders = []
        for ds, tds in zip(self.datasets, self.test_datasets):
            self.train_loaders.append(torch.utils.data.DataLoader(ds,
                                                                  batch_size=self.batch_size,
                                                                  pin_memory = True,                                                           
                                                                  num_workers=self.workers))
            self.valid_loaders.append(torch.utils.data.DataLoader(tds,
                                                                  batch_size=self.batch_size,
                                                                  pin_memory = True,
                                                                  num_workers=self.workers))    
    
    def final_eval(self):
        test_meters = [AverageMeterGroup() for _ in range(len(self.test_datasets))]
        self.model.eval()
        for step, (test_objects) in enumerate(zip(*self.valid_loaders)):                 
            for domain_idx in range(len(self.datasets)):
                self.curr_domain = domain_idx
                tt_X, tt_y = test_objects[domain_idx]                
                tt_X, tt_y = to_device(tt_X, self.device), to_device(tt_y, self.device)
                with torch.no_grad():
                    logits, loss = self._logits_and_loss(tt_X, tt_y)
                    metrics = self.metrics(logits, tt_y)
                    metrics['loss'] = loss.item() / self.p[domain_idx]
                    test_meters[domain_idx].update(metrics)
        for domain_idx in range(len(self.datasets)):                    
            self._logger.info(f'Final eval. Domain: {domain_idx} \nTest: {test_meters[domain_idx]}')
                                                                              

    def _logits_and_loss(self, X, y):
        domain_idx = self.curr_domain
        out = self.model(X, domain_idx)
        logits = out['logits']
        hidden_1_list = out['hidden_states']
        loss = self.loss(logits, y)
        if len(self.datasets) > 1 and self.contrastive_coeff>0:
            another_domain_idx = np.random.choice(list(set(range(len(self.datasets))) - {domain_idx}))
            hidden_2_list = self.model(X, another_domain_idx)['hidden_states']
            loss += self.contrastive_coeff * sum([contrastive_loss(h1, h2, self.t)
                                                  for h1, h2 in zip(hidden_1_list, hidden_2_list)]) / len(hidden_1_list)
                                                          
        return logits, loss# * self.p[domain_idx]

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
                if step == 0:
                    self.writer.add_scalar(f'Loss/train_{domain_idx}', loss.item()# / self.p[domain_idx],
                                           self._step_num)
                loss.backward()

                metrics = self.metrics(logits, trn_y)
                metrics['loss'] = loss.item()# / self.p[domain_idx]
                trn_meters[domain_idx].update(metrics)
                if step == 0:
                    self.writer.add_scalar(f'Acc/train_{domain_idx}', metrics['acc1'], self._step_num)



                if self.log_frequency is not None and step % self.log_frequency == 0:
                                    # validate
                    self.model.eval()
                    with torch.no_grad():
                        logits, loss = self._logits_and_loss(val_X, val_y)                        
                        self.writer.add_scalar(f'Loss/val_{domain_idx}',
                                               loss.item() / self.p[domain_idx], self._step_num)
                        metrics = self.metrics(logits, val_y)
                        metrics['loss'] = loss.item() / self.p[domain_idx]
                        val_meters[domain_idx].update(metrics)
                        self.writer.add_scalar(f'Acc/val_{domain_idx}', metrics['acc1'], self._step_num)
                        # update p[domain_idx] using validation loss
                        #self.p[domain_idx] *= torch.exp(loss / self.p[domain_idx] * self.eta_lr)
                    self.model.train()                    
                    self._logger.info('Epoch [%s/%s] Step [%s/%s], Seed:[%s], Domain: %s\nTrain: %s\nValid: %s',
                                      epoch + 1, self.num_epochs, step + 1,
                                      min([len(loader) for loader in self.train_loaders]),
                                      seed, domain_idx, trn_meters[domain_idx], val_meters[domain_idx])

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # perform a step after calculating loss on each domain
            self.model_optim.step()

            #self.p /= self.p.sum()
            #for i in range(self.p.shape[0]):
            #    self.writer.add_scalar(f'P_{i}', self.p[i].item(), self._step_num)
            self._step_num += 1

        self.lr_scheduler.step()
        # update drop path proba
        self.model.model.set_drop_path_proba(self.model.model.get_drop_path_proba() + self.drop_path_proba_delta)
        save_checkpoint(self._ckpt_dir, epoch, self.model.state_dict(),
                        self.model_optim.state_dict(), None)


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--config", default='md_main_retrain.cfg')
    args = parser.parse_args()

    config = ConfigObj(args.config)

    datasets_train, datasets_valid = datasets.get_datasets(config['datasets'].split(';'),
                                                           int(config['darts']['input_size']),
                                                           int(config['darts']['input_channels']),
                                                           int(config['cutout_length']))

    model = SparceMdDartsModel(config)
    criterion = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(model.parameters(), float(config['darts']['optim']['w_lr']),
                            momentum=float(config['darts']['optim']['w_momentum']),
                            weight_decay=float(config['darts']['optim']['w_weight_decay']))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, int(config['epochs']),
                                                              eta_min=float(config['darts']['optim']['w_lr_min']))

    trainer = MdDartsRetrainer(config['architecture_path'],
                               config['folder_name'],
                               model,
                               loss=criterion,
                               metrics=lambda output, target: accuracy(output, target, topk=(1,)),
                               optimizer=optim,
                               lr_scheduler=lr_scheduler,
                               num_epochs=int(config['epochs']),
                               datasets=datasets_train,
                               test_datasets = datasets_valid,
                               eta_lr=config['darts']['optim']['eta_lr'],
                               seed=int(config['seed']),
                               batch_size=int(config['batch_size']),
                               log_frequency=int(config['log_frequency']),
                               contrastive_coeff = float(config['darts']['contrastive_coeff']),
                               drop_path_proba_delta=float(config['darts']['drop_path_proba_delta'])
                               )
    print('Trainer initialized')
    print('---' * 20)
    trainer.fit()
    trainer.final_eval()
