import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from nni.retiarii.oneshot.pytorch.darts import DartsLayerChoice, \
    DartsInputChoice, DartsTrainer
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, \
    replace_layer_choice, replace_input_choice, to_device
from utils import has_checkpoint, save_checkpoint, load_checkpoint, js_divergence


class HyperDartsLayerChoice(DartsLayerChoice):
    def __init__(self, layer_choice, num_domains=1,
                 sampling_mode='softmax', t=0.2, *args, **kwargs):
        """
        Params:
            sampling_mode: str, sampling mode
            t: float, temperature
        """
        super(HyperDartsLayerChoice, self).__init__(layer_choice)
        self.sampling_mode = sampling_mode
        assert sampling_mode in ['gumbel-softmax', 'softmax']
        self.t = t
        delattr(self, 'alpha')
        self.alpha = nn.ParameterList([nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
                                       for _ in range(num_domains)])
        self.op_param_num = []
        for op in self.op_choices.values():
            self.op_param_num.append(sum([torch.prod(torch.tensor(p.size())).item() \
                                          for p in op.parameters()]))
        self.op_param_num = nn.Parameter(torch.tensor(self.op_param_num), requires_grad=False)

    def forward(self, inputs, batch, lam=torch.tensor(0.0)):
        """
            Params:
            inputs: prev node value
            batch: PCA representation (domain is unknown) or index of domain
            lam: regularization coefficient
        """
        op_results = torch.stack([op(inputs, domain_idx=batch) for op in self.op_choices.values()])
        rbf_outputs = self.alpha[batch]
        if self.sampling_mode == 'gumbel-softmax':
            weights = torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=rbf_outputs
            ).sample()
        elif self.sampling_mode == 'softmax':
            weights = F.softmax(rbf_outputs / self.t, dim=-1)
        alpha_shape = list(weights.shape) + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * weights.view(*alpha_shape), 0)

    def _hyperloss(self, batch, lam=torch.tensor(0.0)):
        rbf_outputs = self.alpha[batch]
        weights = F.softmax(rbf_outputs, dim=-1)
        return lam * (weights @ self.op_param_num.float()).mean()

    def _concord_loss(self):
        loss = torch.tensor(0.0).to(self.alpha[0].device)
        if len(self.alpha) == 1:
            return loss
        for i in range(len(self.alpha)):
            for j in range(i + 1, len(self.alpha)):
                loss = js_divergence(self.alpha[i].softmax(dim=0),
                                     self.alpha[j].softmax(dim=0))
        return 2 * loss / (len(self.alpha) * (len(self.alpha) - 1))

    def export(self, batch, lam=torch.tensor(0.0)):
        rbf_outputs = self.alpha[batch]
        return list(self.op_choices.keys())[torch.argmax(rbf_outputs).item()]


class HyperDartsInputChoice(DartsInputChoice):
    def __init__(self, input_choice, num_domains=1,
                 sampling_mode='softmax', t=0.2):
        super(HyperDartsInputChoice, self).__init__(input_choice)
        delattr(self, 'alpha')
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
            for _ in range(num_domains)])

        assert sampling_mode in ['gumbel-softmax', 'softmax']
        self.sampling_mode = sampling_mode
        self.t = t

    def forward(self, inputs, batch, lam=torch.tensor(0.0)):
        inputs = torch.stack(inputs)
        # batch has int type
        rbf_outputs = self.alpha[batch]
        alpha_shape = list(rbf_outputs.shape) + [1] * (len(inputs.size()) - 1)
        if self.sampling_mode == 'softmax':
            return torch.sum(inputs * F.softmax(rbf_outputs / self.t, -1).view(*alpha_shape), 0)
        if self.sampling_mode == 'gumbel-softmax':
            weights = torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=rbf_outputs
            ).sample()
            return torch.sum(inputs * weights.view(*alpha_shape), 0)

    def _hyperloss(self, batch, lam=torch.tensor(0.0)):
        return lam * 0

    def _concord_loss(self):
        loss = torch.tensor(0.0).to(self.alpha[0].device)
        if len(self.alpha) == 1:
            return loss
        for i in range(len(self.alpha)):
            for j in range(i + 1, len(self.alpha)):
                loss += js_divergence(self.alpha[i].softmax(dim=0),
                                      self.alpha[j].softmax(dim=0))
        return 2 * loss / (len(self.alpha) * (len(self.alpha) - 1))

    def export(self, batch, lam=torch.tensor(0.0)):
        rbf_outputs = self.alpha[batch]
        return torch.argsort(-rbf_outputs).cpu().numpy().tolist()[:self.n_chosen]


class HyperDartsTrainer(DartsTrainer):
    def __init__(self, folder_name, model, loss, metrics, optimizer, lr_scheduler,
                 num_epochs, datasets, seed=0, concord_coeff=0.0, grad_clip=5.,
                 batch_size=64, workers=0,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, betas=(0.5, 0.999),
                 arc_weight_decay=1e-3, unrolled=False,
                 sampling_mode='softmax', t=1.0
                 ):
        # super(HyperDartsTrainer, self).__init__(*args, **kwargs)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.datasets = datasets
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else device
        self.concord_coeff = torch.tensor(concord_coeff).to(self.device)
        self.log_frequency = log_frequency
        logging.basicConfig(filename=os.path.join('searchs', folder_name,
                                                  folder_name + '.log'), level=logging.INFO)
        self._ckpt_dir = os.path.join('searchs', folder_name)
        self._seed = seed
        self._logger = logging.getLogger('darts')
        self.writer = SummaryWriter(os.path.join('.', 'searchs', folder_name))
        self._step_num = 0
        self.p = torch.tensor([1 / len(self.datasets)] * len(self.datasets)).to(self.device)

        self.model.to(self.device)

        self.nas_modules = []
        replace_layer_choice(self.model, lambda lay_choice: HyperDartsLayerChoice(
            lay_choice, sampling_mode=sampling_mode, t=t, num_domains=len(self.datasets)),
                             self.nas_modules)
        replace_input_choice(self.model, lambda lay_choice: HyperDartsInputChoice(
            lay_choice, sampling_mode=sampling_mode, t=t, num_domains=len(self.datasets)),
                             self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)

        self.model_optim = optimizer
        self.lr_scheduler = lr_scheduler
        # use the same architecture weight for modules with duplicated names
        ctrl_params = {}
        for _, m in self.nas_modules:
            if m.name in ctrl_params:
                assert [p.size() for p in m.alpha] == \
                       [p.size() for p in ctrl_params[m.name]], \
                    'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha

        list_of_params = []
        for params in ctrl_params.values():
            list_of_params.extend(params)
        self.ctrl_optim = torch.optim.Adam(list_of_params, arc_learning_rate, betas=betas,
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled
        self.grad_clip = grad_clip

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
        batch = self.curr_domain

        logits = self.model(X, batch, self.sampled_lambda)
        loss = self.loss(logits, y) + self.concord_coeff * self.model._concord_loss()
        return logits, loss * self.p[batch]

    def _train_one_epoch(self, epoch):
        if has_checkpoint(self._ckpt_dir, epoch):
            load_checkpoint(self._ckpt_dir, epoch, self.model, self.model_optim,
                            self.ctrl_optim)
            self._logger.info(f'Loaded checkpoint_{epoch}.ckp')
            return
        self.model.train()
        trn_meters = [AverageMeterGroup() for _ in range(len(self.datasets))]
        val_meters = [AverageMeterGroup() for _ in range(len(self.datasets))]
        seed = self._seed + epoch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        for step, (train_objects, valid_objects) in enumerate(zip(zip(*self.train_loaders), zip(*self.valid_loaders))):
            self.ctrl_optim.zero_grad()
            self.model_optim.zero_grad()
            for domain_idx in range(len(self.datasets)):
                trn_X, trn_y = train_objects[domain_idx]
                val_X, val_y = valid_objects[domain_idx]
                trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
                val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)

                # sample lambda
                lam = torch.tensor(0.0).to(self.device)
                self.sampled_lambda = lam

                # save current domain
                self.curr_domain = domain_idx

                # phase 1. architecture step
                if self.unrolled:
                    self._unrolled_backward(trn_X, trn_y, val_X, val_y)
                else:
                    self._backward(val_X, val_y)

                # phase 2: child network step
                logits, loss = self._logits_and_loss(trn_X, trn_y)
                self.writer.add_scalar(f'Loss/train_{domain_idx}', loss.item() / self.p[domain_idx],
                                       self._step_num)
                loss.backward()

                metrics = self.metrics(logits, trn_y)
                metrics['loss'] = loss.item() / self.p[domain_idx]
                trn_meters[domain_idx].update(metrics)

                # validate
                self.model.eval()
                self.sampled_lambda = torch.tensor(0.0).to(self.device)
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
            self.ctrl_optim.step()

            self.p /= self.p.sum()
            for i in range(self.p.shape[0]):
                self.writer.add_scalar(f'P_{i}', self.p[i].item(), self._step_num)
            self._step_num += 1

        self.lr_scheduler.step()
        save_checkpoint(self._ckpt_dir, epoch, self.model.state_dict(),
                        self.model_optim.state_dict(), self.ctrl_optim.state_dict())

    @torch.no_grad()
    def export(self, batch):
        batch = to_device(batch, self.device)
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export(batch)
        return result
