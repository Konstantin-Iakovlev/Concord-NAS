import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import STEPS, PRIMITIVES, DARTS_V2, CONCAT
from collections import OrderedDict
from utils import mask2d
from typing import List, Tuple, Optional
from utils import LockedDropout
from utils import embedded_dropout

from nni.retiarii.nn.pytorch import LayerChoice
from nni.retiarii.oneshot.pytorch.utils import replace_layer_choice
from ops import Operation

INITRANGE = 0.04


class MdDartsRnnLayerChoice(nn.Module):
    def __init__(self, layer_choice: LayerChoice,
                 n_domains: int = 1,
                 sampling_mode: str = 'softmax'):
        """DARTS layer choice parametrized by alpha

        :param layer_choice: layer choice
        :param n_domains: number of domains, defaults to 1
        :param sampling_mode: sampling mode, defaults to 'softmax'
        """
        super().__init__()
        self.label = layer_choice.label
        self.sampling_mode = sampling_mode
        self.op_choices = nn.ModuleDict(OrderedDict(
            [(name, layer_choice[name]) for name in layer_choice.names]))
        num_prev_nodes = int(self.label[5:])
        self.alpha = nn.ParameterList([torch.randn(num_prev_nodes, len(self.op_choices)) * 1e-3
                                       for _ in range(n_domains)])

    def forward(self, c: torch.Tensor, h: torch.Tensor, states: torch.Tensor, domain_idx: int = 1) -> torch.Tensor:
        """Performs forward pass

        :param c: tensor of shape (node_idx, batch_size, nhid)
        :param h: tensor of shape (node_idx, batch_size, nhid)
        :param states: tensor of shape (node_idx, batch_size, nhid)
        :param domain_idx: domain index, defaults to 1
        :return: weightted output
        """
        unweighted = torch.stack([states + c * (op(h) - states) for op in self.op_choices.values()],
                                 dim=0)  # (num_ops, num_prev, *)
        # TODO: add Gumbel-Softmax support
        if self.sampling_mode == 'softmax':
            weights = self.alpha[domain_idx].softmax(-1).t()
            weights = weights.view(
                list(weights.shape) + [1] * (len(unweighted.shape) - 2))
        else:
            raise NotImplementedError

        weighted = (unweighted * weights).sum(dim=[0, 1])
        return weighted

    def _export_single(self, domain_idx: int) -> Tuple[str, int]:
        """Performs a discretization of alpha of a specific domain

        :param domain_idx: domain index
        :return: tuple of name, previous node index
        """
        W = self.alpha[domain_idx].detach().softmax(-1).cpu().numpy()
        none_idx = [i for i, name in enumerate(
            self.op_choices.keys()) if name == 'none'][0]
        W[:, none_idx] = -float('inf')
        best_prev_node = W.max(-1).argmax()
        best_op_idx = W[best_prev_node].argmax()
        return (list(self.op_choices.keys())[best_op_idx], best_prev_node)

    def export(self) -> List[Tuple[str, int]]:
        """Performs discretization of all alphas

        :return: discretized alphas: list of (name, prev_index) for each domain
        """
        return [self._export_single(domain_idx) for domain_idx in range(len(self.alpha))]


class MdOneHotRnnLayerChoice(nn.Module):
    def __init__(self, layer_choice: LayerChoice, genotype: List[List[Tuple[str, int]]]):
        """Discretized layer choice

        :param layer_choice: layer choice
        :param genotype: discretized alphas
        """
        super().__init__()
        self.label = layer_choice.label
        # save all candidate operations with no memory overhead
        self.op_choices = nn.ModuleDict(OrderedDict(
            [(name, layer_choice[name]) for name in layer_choice.names]))
        self.node_idx = int(self.label[5:])  # in [1, 8]
        self.genotype = genotype

    def forward(self, c: torch.Tensor, h: torch.Tensor, states: torch.Tensor, domain_idx: int = 0) -> torch.Tensor:
        """Performs forward pass

        :param c: tensor of shape (node_idx, batch_size, nhid)
        :param h: tensor of shape (node_idx, batch_size, nhid)
        :param states: tensor of shape (node_idx, batch_size, nhid)
        :param domain_idx: domain index, defaults to 1
        :return: current state
        """
        name, prev_node_idx = self.genotype[domain_idx][self.node_idx - 1]
        act_fn = self.op_choices[name]
        out = states[prev_node_idx] + c[prev_node_idx] * (act_fn(h[prev_node_idx]) - states[prev_node_idx])
        return out

    def export(self):
        """Performs export of the architecture

        :return: genotype
        """
        return self.genotype


class MdRnnCell(nn.Module):
    def __init__(self, ninp: int, nhid: int, dropouth: float, dropoutx: float, genotype=None,
                 n_domains: int = 1):
        """Multidomain RNN cell. Supports one-hot layer choice and DARTS layer choice

        :param ninp: input dim
        :param nhid: hidden dim
        :param dropouth: dropout for x
        :param dropoutx: dropout for h
        :param genotype: architecture if not performing search phase, defaults to None
        :param n_domains: number of domains, defaults to 1
        """
        super(MdRnnCell, self).__init__()
        self.nhid = nhid
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.genotype = genotype
        self.n_domains = n_domains

        self._W0 = nn.Parameter(torch.Tensor(
            ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.Tensor(nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for _ in range(STEPS)
        ])
        self.ops = nn.ModuleList()
        for node_idx in range(1, STEPS + 1):
            self.ops.append(LayerChoice(
                OrderedDict([(name, Operation(name)) for name in PRIMITIVES]),
                label=f'node_{node_idx}'
            ))

        # replace layer choices
        if self.genotype:  # perform fine-tuning phase
            replace_layer_choice(
                self, lambda lc: MdOneHotRnnLayerChoice(lc, self.genotype))
        else:  # perform architecture search phase
            replace_layer_choice(self, lambda lc: MdDartsRnnLayerChoice(lc, n_domains=n_domains,
                                                                        sampling_mode='softmax'))

        self.bn = nn.ModuleList(
            [nn.BatchNorm1d(nhid, affine=False) for _ in range(n_domains)])

    def _compute_init_state(self, x: torch.Tensor, h_prev: torch.Tensor,
                            x_mask: torch.Tensor, h_mask: torch.Tensor) -> torch.Tensor:
        """Computes initial state s0

        :param x: tensor of shape (bs, ninp)
        :param h_prev: tensor of shape (bs, nhid)
        :param x_mask: tensor of shape (bs, ninp)
        :param h_mask: tensor of shape (bs, nhid)
        :return: tensor of shape (bs, nhid')
        """
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def cell(self, x: torch.Tensor, h_prev: torch.Tensor, x_mask: torch.Tensor,
             h_mask: torch.Tensor, domain_idx: int = 0) -> torch.Tensor:
        """Performs forward cell pass

        :param x: tensor of shape (bs, ninp)
        :param h_prev: tensor of shape (bs, nhid)
        :param x_mask: tensor of shape (bs, ninp)
        :param h_mask: tensor of shape (bs, nhid)
        :param domain_idx: domain index
        :return: mean of selected states
        """
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        if self.genotype is None:  # disable batch norm when fine-tuning
            s0 = self.bn[domain_idx](s0)
        # TODO: note that we can speed up the forward pass during one-hot training like
        # in the original implementation
        states = s0[None]
        for i in range(STEPS):
            if self.training:
                masked_states = states * h_mask[None]
            else:
                masked_states = states
            ch = masked_states.view(-1, self.nhid).mm(
                self._Ws[i]).view(i + 1, -1, 2 * self.nhid)
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            s = self.ops[i](c, h, states, domain_idx)
            if self.genotype is None:
                s = self.bn[domain_idx](s)

            states = torch.cat([states, s[None]], dim=0)

        return torch.mean(states[-CONCAT:], dim=0)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor,
                domain_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs forward RNN cell pass

        :param inputs: tensor of shape (seq_len, batch_size, ninp)
        :param hidden: tensor of shape (1, batch_size, nhid)
        :param domain_idx: domain index
        :return: tuple of all hiiden states, last hidden state
        """
        seq_len, batch_size = inputs.size(0), inputs.size(1)
        if self.training:
            x_mask = mask2d(batch_size, inputs.size(2),
                            keep_prob=1. - self.dropoutx).to(next(self.parameters()).device)
            h_mask = mask2d(batch_size, hidden.size(2),
                            keep_prob=1. - self.dropouth).to(next(self.parameters()).device)
        else:
            x_mask = h_mask = None

        hidden = hidden[0]
        hiddens = []
        for t in range(seq_len):
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask, domain_idx)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        return hiddens, hiddens[-1].unsqueeze(0)

    def export(self) -> List[List[Tuple[str, int]]]:
        """Performs architecture discretization

        :return: architectures for each domain
        """
        if self.genotype:
            return self.genotype
        nodes_genotypes = [node.export() for node in self.ops]
        return [[g[domain_idx] for g in nodes_genotypes] for domain_idx in range(self.n_domains)]


class MdRnnModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nhidlast,
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1, genotype=None,
                 n_domains: int = 1):
        super(MdRnnModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)

        assert ninp == nhid == nhidlast
        self.genotype = genotype
        self.n_domains = n_domains
        self.rnn = MdRnnCell(ninp, nhid, dropouth,
                             dropoutx, genotype, n_domains)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        # extra line, because Enc and Dec are tied
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input: torch.Tensor, hidden: List[torch.Tensor], domain_idx: int = 0,
                return_h: bool = False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input,
                               dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        raw_output, new_h = self.rnn(raw_output, hidden[0], domain_idx)
        new_hidden.append(new_h)
        raw_outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def _loss(self, hidden: List[torch.Tensor], input: torch.Tensor, target: torch.LongTensor,
              domain_idx: int = 0) -> torch.Tensor:
        log_prob, hidden_next = self(
            input, hidden, domain_idx=domain_idx, return_h=False)
        loss = nn.functional.nll_loss(
            log_prob.view(-1, log_prob.size(2)), target)
        return loss, hidden_next

    def init_hidden(self, batch_size: int):
        device = next(self.parameters()).device
        return [torch.zeros(1, batch_size, self.nhid).to(device)]

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(MdRnnModel, self).named_parameters():
            if 'alpha' in name:
                continue
            yield name, p

    def struct_named_parameters(self):
        for name, p in super(MdRnnModel, self).named_parameters():
            if 'alpha' in name:
                yield name, p

    def struct_parameters(self):
        for _, p in self.struct_named_parameters():
            yield p

    def export(self):
        return self.rnn.export()

