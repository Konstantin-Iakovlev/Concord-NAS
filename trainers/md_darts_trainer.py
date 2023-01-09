import copy
import sys 

import numpy as np
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

from utils import has_checkpoint, save_checkpoint, load_checkpoint, js_divergence, contrastive_loss, multi_js_divergence
from models.losses import MdTripletLoss

MAX_T = 0.2

class MdDartsLayerChoice(DartsLayerChoice):
    def __init__(self, layer_choice, num_domains=1,
                 sampling_mode: str = 'softmax', t: float = MAX_T):
        """
        param: sampling_mode: sampling mode
        param: t: temperature
        """
        super(MdDartsLayerChoice, self).__init__(layer_choice)
        self.sampling_mode = sampling_mode
        assert sampling_mode in ['gumbel-softmax', 'softmax']
        self.t = t
        delattr(self, 'alpha')
        self.alpha = nn.ParameterList([nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)
                                       for _ in range(num_domains)])

    def forward(self, inputs, domain_idx: int):
        """
        param: inputs: prev node value
        param: domain_idx: domain index
        """
        op_results = torch.stack([op(inputs, domain_idx=domain_idx) for op in self.op_choices.values()])
        rbf_outputs = self.alpha[domain_idx]
        if self.sampling_mode == 'gumbel-softmax':
            weights = torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=rbf_outputs
            ).rsample()
        elif self.sampling_mode == 'softmax':
            weights = F.softmax(rbf_outputs / self.t, dim=-1)
        alpha_shape = list(weights.shape) + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * weights.view(*alpha_shape), 0)

    def concord_loss(self):
        # TODO: try to use "linear" regularizer instead of quadratic
        loss = torch.tensor(0.0).to(self.alpha[0].device)
        if len(self.alpha) == 1:
            return loss
        return multi_js_divergence([alpha.softmax(dim=0) for alpha in self.alpha])
        

    def export(self, domain_idx: int):
        rbf_outputs = self.alpha[domain_idx]
        return list(self.op_choices.keys())[torch.argmax(rbf_outputs).item()]


class MdDartsInputChoice(DartsInputChoice):
    def __init__(self, input_choice, num_domains=1,
                 sampling_mode='softmax', t=MAX_T):
        super(MdDartsInputChoice, self).__init__(input_choice)
        delattr(self, 'alpha')
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
            for _ in range(num_domains)])

        assert sampling_mode in ['gumbel-softmax', 'softmax']
        self.sampling_mode = sampling_mode
        self.t = t

    def forward(self, inputs, domain_idx: int):
        inputs = torch.stack(inputs)
        # batch has int type
        rbf_outputs = self.alpha[domain_idx]
        alpha_shape = list(rbf_outputs.shape) + [1] * (len(inputs.size()) - 1)
        if self.sampling_mode == 'softmax':
            return torch.sum(inputs * F.softmax(rbf_outputs / self.t, -1).view(*alpha_shape), 0)
        if self.sampling_mode == 'gumbel-softmax':
            weights = torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=rbf_outputs
            ).rsample()
            return torch.sum(inputs * weights.view(*alpha_shape), 0)

    def concord_loss(self):
        loss = torch.tensor(0.0).to(self.alpha[0].device)
        if len(self.alpha) == 1:
            return loss
        return multi_js_divergence([alpha.softmax(dim=0) for alpha in self.alpha])

    def export(self, domain_idx: int):
        rbf_outputs = self.alpha[domain_idx]
        return torch.argsort(-rbf_outputs).cpu().numpy().tolist()[:self.n_chosen]


class MdDartsTrainer(DartsTrainer):
    def __init__(self, folder_name, model, loss, metrics, optimizer, lr_scheduler,
                 num_epochs, datasets,  test_datasets, seed=0, concord_coeff=0.0, contrastive_coeff=0.0, grad_clip=5.,
                 batch_size=64, workers=0,
                 device=None, log_frequency=None,
                 arc_learning_rate=3.0E-4, betas=(0.5, 0.999),
                 arc_weight_decay=1e-3, unrolled=False,
                 eta_lr=0.01,
                 sampling_mode='softmax', t=1.0,
                 delta_t=-0.026,
                 drop_path_proba_delta=0.0,
                 ):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.datasets = datasets
        self.test_datasets = test_datasets        

        
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else device
        self.concord_coeff = torch.tensor(concord_coeff).to(self.device)
        self.contrastive_loss = MdTripletLoss()
        self.contrastive_coeff = torch.tensor(contrastive_coeff).to(self.device)
        self.eta_lr = eta_lr
        self.log_frequency = log_frequency
        self._ckpt_dir = os.path.join('searchs', folder_name)
        self._seed = seed        
        os.makedirs(os.path.join('searchs', folder_name), exist_ok=True)
        self._logger = logging.getLogger('darts')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')        
        handler = logging.FileHandler(os.path.join('searchs', folder_name,
                                                                 folder_name + '.log'))
        handler.setFormatter(formatter)                                                                 
        self._logger.setLevel(logging.DEBUG)                                                                 
        self._logger.addHandler(handler)        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)                                                                 
        
        self.writer = SummaryWriter(os.path.join('searchs', folder_name))
        self._step_num = 0
        self.p = torch.tensor([1 / len(self.datasets)] * len(self.datasets)).to(self.device)

        self.model.to(self.device)

        self.nas_modules = []
        replace_layer_choice(self.model, lambda lay_choice: MdDartsLayerChoice(
            lay_choice, sampling_mode=sampling_mode, t=t, num_domains=len(self.datasets)),
                             self.nas_modules)
        replace_input_choice(self.model, lambda lay_choice: MdDartsInputChoice(
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
                                           weight_decay=arc_weight_decay)
        self.unrolled = unrolled
        self.grad_clip = grad_clip
        self.t = t
        self.delta_t = delta_t
        self.drop_path_proba_delta = drop_path_proba_delta

        self._init_dataloaders()

    def _init_dataloaders(self):
        self.train_loaders = []
        self.valid_loaders = []
        self.test_loaders = []        
        
        for tds, ds in zip(self.test_datasets, self.datasets):
            n_train = len(ds)
            # 50% on validation
            split = n_train // 2
            np.random.seed(self._seed)
            indices = np.random.permutation(np.arange(n_train))
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
            self.train_loaders.append(torch.utils.data.DataLoader(ds,
                                                                  batch_size=self.batch_size,
                                                                  sampler=train_sampler,
                                                                  num_workers=self.workers, pin_memory=True))
            self.valid_loaders.append(torch.utils.data.DataLoader(ds,
                                                                  batch_size=self.batch_size,
                                                                  sampler=valid_sampler,
                                                                  num_workers=self.workers, pin_memory=True))
            self.test_loaders.append(torch.utils.data.DataLoader(tds,
                                                                  batch_size=self.batch_size,                                                                  
                                                                  num_workers=self.workers, pin_memory=True))                                                                  


    def _logits_and_loss(self, X, y, regularizer=True):
        domain_idx = self.curr_domain
        model_out = self.model(X, domain_idx)
        logits = model_out['logits']
        hidden_1_list = model_out['hidden_states']
        loss = self.loss(logits, y) 
        if regularizer:
            if self.concord_coeff > 0:
                coef = self.concord_coeff                
                loss += coef * self.model.concord_loss()
            # contrastive loss
            #raise ValueError([len(self.datasets)>1, self.contrastive_coeff>0, self.contrastive_coeff]) 
            if len(self.datasets) > 1 and self.contrastive_coeff>0:
                hidden_2_list = self.model(self.another_batch['x'], self.another_domain)['hidden_states']
                loss += self.contrastive_coeff * sum([self.contrastive_loss.forward(h1, h2, y, self.another_batch['y'])
                                                  for h1, h2 in zip(hidden_1_list, hidden_2_list)]) / len(hidden_1_list)

        return logits, loss # * self.p[domain_idx]

    def final_eval(self, fast=False, epoch=None):
        test_meters = [AverageMeterGroup() for _ in range(len(self.test_datasets))]
        for m in self.nas_modules:
            if hasattr(m, 't'):
                m.t = max(MAX_T, self.t) 
        self.model.eval()
        evaluated = 0
        for step, (test_objects) in enumerate(zip(*self.test_loaders)):                 
            for domain_idx in range(len(self.datasets)):
                self.curr_domain = domain_idx
                tt_X, tt_y = test_objects[domain_idx]                
                tt_X, tt_y = to_device(tt_X, self.device), to_device(tt_y, self.device)
                with torch.no_grad():
                    logits, loss = self._logits_and_loss(tt_X, tt_y, False)
                    metrics = self.metrics(logits, tt_y)
                    metrics['loss'] = loss.item() / self.p[domain_idx]
                    test_meters[domain_idx].update(metrics)
                if domain_idx == 0:
                    evaluated += len(tt_y)
            if fast and evaluated > 4000/(len(self.datasets)):
                break
        mean = 0.0
        for domain_idx in range(len(self.datasets)):                    
            self._logger.info(f'Final eval (epoch: {epoch}). Domain: {domain_idx} \nTest: {test_meters[domain_idx]}')
            self.writer.add_scalar(f'Acc/test_{domain_idx}', test_meters[domain_idx]['acc1'].avg, self._step_num) 
            mean += test_meters[domain_idx]['acc1'].avg/len(test_meters)
        self._logger.info(f'Final eval. Avg Domain. \nTest: {mean}')
        self.writer.add_scalar(f'Acc/test_avg', mean, self._step_num) 
        
        
                                                 
    def _train_one_epoch(self, epoch):
        if has_checkpoint(self._ckpt_dir, epoch):
            load_checkpoint(self._ckpt_dir, epoch, self.model, self.model_optim,
                            self.ctrl_optim)
            self._logger.info(f'Loaded checkpoint_{epoch}.ckp')
        self.model.train()
        trn_meters = [AverageMeterGroup() for _ in range(len(self.datasets))]
        val_meters = [AverageMeterGroup() for _ in range(len(self.datasets))]
        seed = self._seed + epoch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        
        if epoch % 10 == 0:
            self.final_eval(fast=True, epoch=epoch)
        for step, (train_objects, valid_objects) in enumerate(zip(zip(*self.train_loaders), zip(*self.valid_loaders))):      
        
            

            for domain_idx in range(len(self.datasets)):
                self.ctrl_optim.zero_grad()
                self.model_optim.zero_grad()
                trn_X, trn_y = train_objects[domain_idx]
                val_X, val_y = valid_objects[domain_idx]
                trn_X, trn_y = to_device(trn_X, self.device), to_device(trn_y, self.device)
                val_X, val_y = to_device(val_X, self.device), to_device(val_y, self.device)

                # save current domain
                self.curr_domain = domain_idx
                
                self.another_domain = np.random.choice(len(self.datasets))                
                an_X, an_y = valid_objects[self.another_domain]
                an_X = to_device(an_X, self.device)
                an_y = to_device(an_y, self.device)
                self.another_batch = {'x': an_X, 'y': an_y}
                
                
                # phase 1. architecture step
                if self.unrolled:
                    self._unrolled_backward(trn_X, trn_y, val_X, val_y)
                else:
                    self._backward(val_X, val_y)

                # phase 2: child network step
                # TODO: note that contrastive loss depends on w and alpha => we may need to remove in from valid loss
                an_X, an_y = train_objects[self.another_domain]
                an_X = to_device(an_X, self.device)
                an_y = to_device(an_y, self.device)
                self.another_batch = {'x': an_X, 'y': an_y}

                logits, loss = self._logits_and_loss(trn_X, trn_y)
                if step == 0:
                    self.writer.add_scalar(f'Loss/train_{domain_idx}', loss.item() / self.p[domain_idx],
                                           self._step_num)
                loss.backward()
                


                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                # perform a step after calculating loss on each domain
                self.model_optim.step()
                self.ctrl_optim.step()
             

                metrics = self.metrics(logits, trn_y)
                metrics['loss'] = loss.item() / self.p[domain_idx]
                trn_meters[domain_idx].update(metrics)
                if step == 0:
                    self.writer.add_scalar(f'Acc/train_{domain_idx}', metrics['acc1'], self._step_num)
            


            if self.log_frequency is not None and step % self.log_frequency == 0:
                # validate
                self.model.eval()
                
                with torch.no_grad():
                    logits, loss = self._logits_and_loss(val_X, val_y)
                    if step == 0:
                        self.writer.add_scalar(f'Loss/val_{domain_idx}',
                                               loss.item() / self.p[domain_idx], self._step_num)
                    metrics = self.metrics(logits, val_y)
                    metrics['loss'] = loss.item() / self.p[domain_idx]
                    val_meters[domain_idx].update(metrics)                  
                    self.writer.add_scalar(f'Acc/val_{domain_idx}', metrics['acc1'], self._step_num)
                # update p[domain_idx] using validation loss
                self.p[domain_idx] *= torch.exp(loss / self.p[domain_idx] * self.eta_lr)
                self._logger.info('Epoch [%s/%s] Step [%s/%s], Seed:[%s], Domain: %s\nTrain: %s\nValid: %s',
                                  epoch + 1, self.num_epochs, step + 1,
                                  min([len(loader) for loader in self.train_loaders]),
                                  seed, domain_idx, trn_meters[domain_idx], val_meters[domain_idx])
            self.model.train()
       

            self.p /= self.p.sum()
            if step == 0:
                for i in range(self.p.shape[0]):
                    self.writer.add_scalar(f'P_{i}', self.p[i].item(), self._step_num)
            self._step_num += 1

        self.lr_scheduler.step()
        # update temperature
        self.t += self.delta_t
        for m in self.nas_modules:
            if hasattr(m, 't'):
                m.t = max(MAX_T, self.t)                                
        # update drop path proba
        self.model.set_drop_path_proba(self.model.get_drop_path_proba() + self.drop_path_proba_delta)
        try:
            save_checkpoint(self._ckpt_dir, epoch, self.model.state_dict(),
                            self.model_optim.state_dict(), self.ctrl_optim.state_dict())
        except Exception as e:
            print ('could not save checkpoint', repr(e))

    @torch.no_grad()
    def export(self, domain_idx: int):
        domain_idx = to_device(domain_idx, self.device)
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export(domain_idx)
        return result

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.model.parameters()))

        # do virtual step on training data
        lr = self.model_optim.param_groups[0]["lr"]
        momentum = self.model_optim.param_groups[0]["momentum"]
        weight_decay = self.model_optim.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        _, loss = self._logits_and_loss(val_X, val_y)
        w_model, w_ctrl = tuple(self.model.parameters()), \
                          tuple([p for _, c in self.nas_modules for p in c.alpha.parameters()])
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl, allow_unused=True)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian; accumulate gradients
                if param.grad is None:
                    continue
                    param.grad = torch.zeros_like(param)
                if d is None:
                    continue
                    d = torch.zeros_like(param)
                if h is None:
                    continue
                    h = torch.zeros_like(param)
                param.grad -= d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        _, loss = self._logits_and_loss(X, y)
        gradients = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        with torch.no_grad():
            for w, g in zip(self.model.parameters(), gradients):
                if g is None:
                    g = torch.zeros_like(w)
                m = self.model_optim.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _compute_hessian(self, backup_params, dw, trn_X, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw if w is not None]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            self._logger.warning('In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.', norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.model.parameters(), dw):
                    if d is not None:
                        p += e * d

            _, loss = self._logits_and_loss(trn_X, trn_y)              
            dalphas.append(torch.autograd.grad(loss, [p for _, c in self.nas_modules for p in c.alpha.parameters()],
                                               allow_unused=True))
                                                           
                                               

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / (2. * eps) if (p is not None and n is not None) else None  for p, n in zip(dalpha_pos, dalpha_neg) ]
        return hessian
    
    def clear_logger(self):
        self._logger.handlers.clear()