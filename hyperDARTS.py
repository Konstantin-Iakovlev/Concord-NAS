import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from hypernet import PWNet, BasicExpertNet
from nni.retiarii.oneshot.pytorch.darts import DartsLayerChoice, \
        DartsInputChoice, DartsTrainer
from nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup,\
        replace_layer_choice, replace_input_choice, to_device


class HyperDartsLayerChoice(DartsLayerChoice):
    def __init__(self, layer_choice, num_kernels=5, 
            sampling_mode='gumbel-softmax', t=0.2, *args, **kwargs):
        """
        Params:
            num_kernels: int, number of kernels in PWNet
            sampling_mode: str, sampling mode
            t: float, temperature
        """
        super(HyperDartsLayerChoice, self).__init__(layer_choice)
        self.pw_net = PWNet(len(self.op_choices), num_kernels)
        # basic expert net
        # self.expert_net = BasicExpertNet(28, len(self.op_choices))
        self.sampling_mode = sampling_mode
        assert sampling_mode in ['gumbel-softmax', 'softmax']
        self.t = t
        delattr(self, 'alpha')
        self.alpha = self.pw_net.parameters()
        self.op_param_num = []
        for op in self.op_choices.values():
            self.op_param_num.append(sum([torch.prod(torch.tensor(p.size())).item() \
                    for p in op.parameters()]))
        self.op_param_num = torch.tensor(self.op_param_num)

    def forward(self, inputs, lam=torch.tensor(0.0)):
        op_results = torch.stack([op(inputs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        pw_net_outputs = self.pw_net(lam)
        if self.sampling_mode == 'gumbel-softmax':
            weights = torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=pw_net_outputs
            ).sample()
        if self.sampling_mode == 'softmax':
            weights = F.softmax(pw_net_outputs / self.t, dim=-1)
        return torch.sum(op_results * weights.view(*alpha_shape), 0)

    def _hyperloss(self, lam=torch.tensor(0.0)):
        pw_net_outputs = self.pw_net(lam)
        weights = F.softmax(pw_net_outputs, dim=-1)
        return lam * self.op_param_num @ weights

    def export(self, lam=torch.tensor(0.0)):
        pw_net_outputs = self.pw_net(lam)
        return list(self.op_choices.keys())[torch.argmax(pw_net_outputs).item()]


class HyperDartsInputChoice(DartsInputChoice):
    def __init__(self, input_choice, num_kernels=5, sampling_mode='gumbel-softmax', t=0.2):
        super(HyperDartsInputChoice, self).__init__(input_choice)
        self.pw_net = PWNet(input_choice.n_candidates, num_kernels)
        # self.expert_net = BasicExpertNet(14, input_choice.n_candidates)
        delattr(self, 'alpha')
        self.alpha = self.pw_net.parameters()

        assert sampling_mode in ['gumbel-softmax', 'softmax']
        self.sampling_mode = sampling_mode
        self.t = t

    def forward(self, inputs, lam=torch.tensor(0.0)):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        pw_net_outputs = self.pw_net(lam)
        if self.sampling_mode == 'softmax':
            return torch.sum(inputs * F.softmax(pw_net_outputs / self.t, -1).view(*alpha_shape), 0)
        if self.sampling_mode == 'gumbel-softmax':
            weights = torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=pw_net_outputs
            ).sample()
            return torch.sum(inputs * weights.view(*alpha_shape), 0)

    def _hyperloss(self, lam=torch.tensor(0.0)):
        return torch.tensor(0.0)

    def export(self, lam=torch.tensor(0.0)):
        pw_net_outputs = self.pw_net(lam)
        return torch.argsort(-pw_net_outputs).cpu().numpy().tolist()[:self.n_chosen]


class HyperDartsTrainer(DartsTrainer):
    def __init__(self, model, loss, metrics, optimizer,
                 num_epochs, dataset, grad_clip=5.,
                 batch_size=64, workers=4,
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
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
                if device is None else device
        self.log_frequency = log_frequency
        self._logger = logging.getLogger('darts')

        self.model.to(self.device)

        self.nas_modules = []
        replace_layer_choice(self.model, lambda lay_choice: HyperDartsLayerChoice(
            lay_choice, sampling_mode=sampling_mode, t=t), self.nas_modules)
        replace_input_choice(self.model, lambda lay_choice: HyperDartsInputChoice(
            lay_choice,sampling_mode=sampling_mode, t=t), self.nas_modules)
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
                ctrl_params[m.name] = list(m.alpha)

        list_of_params = []
        for params in ctrl_params.values():
            list_of_params.extend(params)
        self.ctrl_optim = torch.optim.Adam(list_of_params, arc_learning_rate, betas=betas,
                                           weight_decay=1.0E-3)
        self.unrolled = unrolled
        self.grad_clip = grad_clip

        self._init_dataloader()

    def _logits_and_loss(self, X, y, lam=torch.tensor(0.0)):
        logits = self.model(X, lam)
        hyperloss = self.model._hyperloss(lam)
        loss = self.loss(logits, y) + hyperloss
        return logits, loss

    
    def _train_one_epoch(self, epoch):
        self.model.train()
        trn_meters = AverageMeterGroup()
        val_meters= AverageMeterGroup()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(self.train_loader, self.valid_loader)):
            trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
            val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)

            # phase 1. architecture step
            self.ctrl_optim.zero_grad()
            if self.unrolled:
                self._unrolled_backward(trn_X, trn_y, val_X, val_y)
            else:
                self._backward(val_X, val_y)
            self.ctrl_optim.step()

            # phase 2: child network step
            self.model_optim.zero_grad()
            # sample lambda
            lam = torch.tensor(0.0).to(self.device)
            logits, loss = self._logits_and_loss(trn_X, trn_y, lam)
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            self.model_optim.step()

            metrics = self.metrics(logits, trn_y)
            metrics['loss'] = loss.item()
            trn_meters.update(metrics)
            
            # validate
            lam = torch.tensor(0.0).to(self.device)
            with torch.no_grad():
                logits, loss = self._logits_and_loss(val_X, val_y, lam)
                metrics = self.metrics(logits, val_y)
                metrics['loss'] = loss.item()
                val_meters.update(metrics)
                

            if self.log_frequency is not None and step % self.log_frequency == 0:
                self._logger.info('Epoch [%s/%s] Step [%s/%s]\nTrain: %s\nValid: %s',
                        epoch + 1, self.num_epochs, step + 1,
                        len(self.train_loader), trn_meters, val_meters)







