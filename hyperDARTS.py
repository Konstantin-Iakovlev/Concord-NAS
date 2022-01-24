import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from hypernet import PWNet, RBF, BasicExpertNet
from nni.retiarii.oneshot.pytorch.darts import DartsLayerChoice, \
        DartsInputChoice, DartsTrainer
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup,\
        replace_layer_choice, replace_input_choice, to_device


class HyperDartsLayerChoice(DartsLayerChoice):
    def __init__(self, layer_choice, input_size=32*2*3, 
            sampling_mode='gumbel-softmax', t=0.2, *args, **kwargs):
        """
        Params:
            input_size: int, input size of BasicExpertNet
            sampling_mode: str, sampling mode
            t: float, temperature
        """
        super(HyperDartsLayerChoice, self).__init__(layer_choice)
        # self.rbf_net = RBF(2 * 28, 10, len(self.op_choices))
        self.rbf_net = BasicExpertNet(input_size, len(self.op_choices))
        self.sampling_mode = sampling_mode
        assert sampling_mode in ['gumbel-softmax', 'softmax']
        self.t = t
        delattr(self, 'alpha')
        self.alpha = nn.ParameterList([nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
            for _ in range(2)])
        # self.alpha = self.rbf_net.parameters()
        self.op_param_num = []
        for op in self.op_choices.values():
            self.op_param_num.append(sum([torch.prod(torch.tensor(p.size())).item() \
                    for p in op.parameters()]))
        self.op_param_num = nn.Parameter(torch.tensor(self.op_param_num), requires_grad=False)

    def forward(self, inputs, batch, lam=torch.tensor(0.0)):
        '''
            Params:
            inputs: prev node value
            batch: PCA representation (domain is unknown) or index of domain 
            lam: regularization coefficient
        '''
        op_results = torch.stack([op(inputs) for op in self.op_choices.values()])
        # rbf_outputs = self.rbf_net(batch)
        rbf_outputs = self.alpha[batch].unsqueeze(0)
        if self.sampling_mode == 'gumbel-softmax':
            weights = torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=rbf_outputs
            ).sample()
        if self.sampling_mode == 'softmax':
            weights = F.softmax(rbf_outputs / self.t, dim=-1)
        weights = weights.transpose(0, 1)
        alpha_shape = list(weights.shape) + [1] * (len(op_results.size()) - 2)
        return torch.sum(op_results * weights.view(*alpha_shape), 0)

    def _hyperloss(self, batch, lam=torch.tensor(0.0)):
        # rbf_outputs = self.rbf_net(batch)
        rbf_outputs = self.alpha[batch]
        weights = F.softmax(rbf_outputs, dim=-1)
        return lam * (weights @ self.op_param_num.float()).mean()


    def _concord_loss(self):
        alpha_1, alpha_2 = self.alpha[0], self.alpha[1]
        p_distr = torch.distributions.Categorical(probs=alpha_1.softmax(dim=0))
        q_distr = torch.distributions.Categorical(probs=alpha_2.softmax(dim=0))
        return torch.distributions.kl.kl_divergence(p_distr, q_distr) + \
                torch.distributions.kl.kl_divergence(q_distr, p_distr)



    def export(self, batch, lam=torch.tensor(0.0)):
        # rbf_outputs = self.rbf_net(batch).mean(dim=0).squeeze()
        rbf_outputs = self.alpha[batch]
        return list(self.op_choices.keys())[torch.argmax(rbf_outputs).item()]


class HyperDartsInputChoice(DartsInputChoice):
    def __init__(self, input_choice, input_size=32*2*3, sampling_mode='gumbel-softmax', t=0.2):
        super(HyperDartsInputChoice, self).__init__(input_choice)
        self.rbf_net = BasicExpertNet(input_size, input_choice.n_candidates)
        delattr(self, 'alpha')
        # self.alpha = self.rbf_net.parameters()
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
            for _ in range(2)])

        assert sampling_mode in ['gumbel-softmax', 'softmax']
        self.sampling_mode = sampling_mode
        self.t = t

    def forward(self, inputs, batch, lam=torch.tensor(0.0)):
        '''
            inputs: inputs
            batch: PCA representation (domain is unknown) or index of domain 
        '''
        inputs = torch.stack(inputs)
        # rbf_outputs = self.rbf_net(batch)
        # batch has int type
        rbf_outputs = self.alpha[batch].unsqueeze(0)
        alpha_shape = list(rbf_outputs.shape)[::-1] + [1] * (len(inputs.size()) - 2)
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
        alpha_1, alpha_2 = self.alpha[0], self.alpha[1]
        p_distr = torch.distributions.Categorical(probs=alpha_1.softmax(dim=0))
        q_distr = torch.distributions.Categorical(probs=alpha_2.softmax(dim=0))
        return torch.distributions.kl.kl_divergence(p_distr, q_distr) + \
                torch.distributions.kl.kl_divergence(q_distr, p_distr)


    def export(self, batch, lam=torch.tensor(0.0)):
        '''
            batch: PCA representation (domain is unknown) or index of domain 
        '''
        rbf_outputs = self.alpha[batch]
        # rbf_outputs = self.rbf_net(batch).mean(dim=0).squeeze()
        return torch.argsort(-rbf_outputs).cpu().numpy().tolist()[:self.n_chosen]


class HyperDartsTrainer(DartsTrainer):
    def __init__(self, folder_name, model, loss, metrics, optimizer,
                 num_epochs, datasets, concord_coeff=0.0, grad_clip=5.,
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
        logging.basicConfig(filename=os.path.join('.', 'searchs', folder_name, 
            folder_name + '.log'))
        self._logger = logging.getLogger('darts')
        self.writer = SummaryWriter(os.path.join('.', 'searchs', folder_name))
        self._step_num = 0
        self.p = torch.tensor([1/len(self.datasets)] * len(self.datasets)).to(self.device)

        self.model.to(self.device)

        self.nas_modules = []
        input_size=32 * 3 * 2
        replace_layer_choice(self.model, lambda lay_choice: HyperDartsLayerChoice(
            lay_choice, sampling_mode=sampling_mode, t=t, input_size=input_size),
            self.nas_modules)
        replace_input_choice(self.model, lambda lay_choice: HyperDartsInputChoice(
            lay_choice,sampling_mode=sampling_mode, t=t, input_size=input_size),
            self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)

        self.model_optim = optimizer
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
        # perform PCA
        # batch = torch.pca_lowrank(X, 2)[0].reshape(X.shape[0], -1)
        batch = self.curr_domain

        logits = self.model(X, batch, self.sampled_lambda)
        loss = self.loss(logits, y) + self.concord_coeff * self.model._concord_loss()
        return logits, loss * self.p[batch]

    
    def _train_one_epoch(self, epoch):
        self.model.train()
        trn_meters = AverageMeterGroup()
        val_meters= AverageMeterGroup()
        for step, (train_objects, valid_objects) in enumerate(zip(zip(*self.train_loaders), zip(*self.valid_loaders))):
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
                self.ctrl_optim.zero_grad()
                if self.unrolled:
                    self._unrolled_backward(trn_X, trn_y, val_X, val_y)
                else:
                    self._backward(val_X, val_y)
                self.ctrl_optim.step()

                # phase 2: child network step
                self.model_optim.zero_grad()
                logits, loss = self._logits_and_loss(trn_X, trn_y)
                self.writer.add_scalar(f'Loss/train_{domain_idx}', loss.item() / self.p[domain_idx],
                        self._step_num)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
                self.model_optim.step()

                metrics = self.metrics(logits, trn_y)
                metrics['loss'] = loss.item() / self.p[domain_idx]
                trn_meters.update(metrics)
                
                # validate
                self.model.eval()
                self.sampled_lambda = torch.tensor(0.0).to(self.device)
                with torch.no_grad():
                    logits, loss = self._logits_and_loss(val_X, val_y)
                    self.writer.add_scalar(f'Loss/val_{domain_idx}', 
                            loss.item() / self.p[domain_idx], self._step_num)
                    metrics = self.metrics(logits, val_y)
                    metrics['loss'] = loss.item() / self.p[domain_idx]
                    val_meters.update(metrics)
                    # update p[domain_idx]
                    self.p[domain_idx] *= torch.exp(loss / self.p[domain_idx] * 0.1)
                    

                if self.log_frequency is not None and step % self.log_frequency == 0:
                    self._logger.info('Epoch [%s/%s] Step [%s/%s], Domain: %s\nTrain: %s\nValid: %s',
                            epoch + 1, self.num_epochs, step + 1,
                            min([len(loader) for loader in self.train_loaders]),
                            domain_idx, trn_meters, val_meters)
            self.p /= self.p.sum()
            self.writer.add_scalar(f'P[0]', self.p[0].item(), self._step_num)
            self._step_num += 1


    @torch.no_grad()
    def export(self, batch):
        batch = to_device(batch, self.device)
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export(batch)
        return result

