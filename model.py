import flax
import jax
import jax.numpy as jnp
import functools
from jax import tree_util
from jax.tree_util import register_pytree_node_class
import functools


import jax
from typing import Any, Callable, Dict, List, Sequence, Tuple
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
import time
import numpy as np

from genotypes import STEPS, CONCAT, INITRANGE
from utils import locked_dropout, mask2d


class MdDartsRnnLayerChoice(nn.Module):
    # n_domains: int
    num_prev_nodes: int
    op_choices = {'relu': nn.relu,
                  'tanh': nn.tanh,
                  'none': lambda x: jnp.zeros_like(x),
                  'sigmoid': nn.sigmoid,
                  'identity': lambda x: x,
                  }

    def setup(self):
        self.alpha = self.param('alpha', lambda rng, shape: jax.random.normal(rng, shape) * 1e-3,
                           (self.num_prev_nodes, len(self.op_choices)))

    def __call__(self, c, h, states):
        """Performs forward pass

        :param c: tensor of shape (node_idx, batch_size, nhid)
        :param h: tensor of shape (node_idx, batch_size, nhid)
        :param states: tensor of shape (node_idx, batch_size, nhid)
        :return: weightted output
        """

        unweighted = jnp.stack([states + c * (op(h) - states) for op in self.op_choices.values()],
                                 axis=0)  # (num_ops, num_prev, *)
        alpha = self.alpha
        weights = nn.activation.softmax(alpha, axis=-1).transpose()
        weights = weights.reshape(
            list(weights.shape) + [1] * (len(unweighted.shape) - 2))

        weighted = (unweighted * weights).sum(axis=[0, 1])
        return weighted


class MdOnehotRnnLayerChoice(nn.Module):
    prev_node_idx: int  # from [0, 7]
    op_name: str
    op_choices = {'relu': nn.relu,
                  'tanh': nn.tanh,
                  'none': lambda x: jnp.zeros_like(x),
                  'sigmoid': nn.sigmoid,
                  'identity': lambda x: x,
                  }
    
    def __call__(self, c, h, states):
        """Performs forward pass

        :param c: tensor of shape (node_idx, batch_size, nhid)
        :param h: tensor of shape (node_idx, batch_size, nhid)
        :param states: tensor of shape (node_idx, batch_size, nhid)
        :return: weightted output
        """
        act_fn = self.op_choices[self.op_name]
        return states[self.prev_node_idx] + c[self.prev_node_idx] * (act_fn(h[self.prev_node_idx]) - states[self.prev_node_idx])


class MdRnnCell(nn.Module):
    ninp: int
    nhid: int
    dropouth: float
    dropoutx: float
    training: bool
    genotype: Any = None

    def setup(self):
        self._W0 = self.param('_W0', lambda key, shape: jax.random.uniform(key, shape) * 2 * INITRANGE - INITRANGE,
                              (self.ninp + self.nhid, 2 * self.nhid)
                              )
        self._Ws = self.param('_Ws', lambda key, shape: jax.random.uniform(key, shape) * 2 * INITRANGE - INITRANGE,
                              (STEPS, self.nhid, 2 * self.nhid)
                              )
        if self.genotype is None:
            self.ops = [MdDartsRnnLayerChoice(num_prev_nodes=i + 1) for i in range(STEPS)]
        else:
            self.ops = [MdOnehotRnnLayerChoice(prev_node_idx=p, op_name=label) for label, p in self.genotype]
        if self.genotype is None:
            # affine = False
            self.bn = nn.BatchNorm(use_bias=False, use_scale=False)

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        """Computes initial state s0

        :param x: tensor of shape (bs, ninp)
        :param h_prev: tensor of shape (bs, nhid)
        :param x_mask: tensor of shape (bs, ninp)
        :param h_mask: tensor of shape (bs, nhid)
        :return: tensor of shape (bs, nhid)
        """
        if self.training:
            xh_prev = jnp.concatenate([x * x_mask, h_prev * h_mask], axis=-1)
        else:
            xh_prev = jnp.concatenate([x, h_prev], axis=-1)
        c0, h0 = jnp.split(xh_prev @ self._W0, [self.nhid], axis=-1)
        c0 = nn.activation.sigmoid(c0)
        h0 = nn.activation.tanh(h0)
        s0 = h_prev + c0 * (h0 - h_prev)
        return s0

    def __call__(self, carry, x):
        """carry = (h_prev, x_mask, h_mask)"""
        """Performs forward cell pass

        :param x: tensor of shape (bs, ninp)
        :param h_prev: tensor of shape (bs, nhid)
        :param x_mask: tensor of shape (bs, ninp)
        :param h_mask: tensor of shape (bs, nhid)
        :param domain_idx: domain index
        :return: mean of selected states
        """
        h_prev, x_mask, h_mask = carry
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        if self.genotype is None:  # disable batch norm when fine-tuning
            s0 = self.bn(s0, use_running_average=not self.training)
        states = s0[None]
        for i in range(STEPS):
            if self.training:
                masked_states = states * h_mask[None]
            else:
                masked_states = states
  
            ch = masked_states.reshape(-1, self.nhid) @ self._Ws[i]
            ch = ch.reshape(i + 1, -1, 2 * self.nhid)

            c, h = jnp.split(ch, [self.nhid], axis=-1)
            c = nn.activation.sigmoid(c)

            s = self.ops[i](c, h, states)
            if self.genotype is None:
                s = self.bn(s, use_running_average=not self.training)

            states = jnp.concatenate([states, s[None]], axis=0)
        new_hidden = jnp.mean(states[-CONCAT:], axis=0)
        return (new_hidden, x_mask, h_mask), new_hidden


class ScannedCell(nn.Module):
    ninp: int
    nhid: int
    dropouth: float
    dropoutx: float
    training: bool
    genotype: Any = None

    @nn.compact
    def __call__(self, carry, x):
        is_init = 'batch_stats' not in self.variables

        if is_init:
            return MdRnnCell(self.ninp, self.nhid, self.dropouth,
                              self.dropoutx, self.training, name='cell')(carry, x[0])
        return nn.scan(MdRnnCell, variable_carry='batch_stats',
                   variable_broadcast='params',
                    in_axes=0, out_axes=0,
                   split_rngs={'params': False})(self.ninp, self.nhid, self.dropouth,
                              self.dropoutx, self.training, name='cell')(carry, x)
    

class MdRnn(nn.Module):
    ninp: int
    nhid: int
    dropouth: float
    dropoutx: float
    training: bool
    genotype: Any = None
    rng_collection: str = 'mask_2d'

    def setup(self):
        self.cell = ScannedCell(self.ninp, self.nhid, self.dropouth, self.dropoutx,
                                self.training,
                              self.genotype)
        
    def __call__(self, inputs, hidden):
        """Performs forward RNN cell pass

        :param inputs: tensor of shape (seq_len, batch_size, ninp)
        :param hidden: tensor of shape (batch_size, nhid)
        :param domain_idx: domain index
        :return: tuple of all hiiden states, last hidden state
        """
        seq_len, batch_size = inputs.shape[0], inputs.shape[1]
        x_mask = mask2d(batch_size, inputs.shape[-1],
                        1. - self.dropoutx, self.make_rng(self.rng_collection))
        h_mask = mask2d(batch_size, hidden.shape[-1],
                            1. - self.dropouth, self.make_rng(self.rng_collection))
        
        return self.cell((hidden, x_mask, h_mask), inputs)
    
            
class MdRnnModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    ntoken: int
    ninp: int
    nhid: int
    nhidlast: int
    training: bool
    dropout: float = 0.5
    dropouth: float = 0.5
    dropoutx: float = 0.5
    dropouti: float = 0.5
    dropoute: float = 0.1
    genotype: Any = None
    rng_collection_emb: str = 'locked_dropout_emb'
    rng_collection_out: str = 'locked_dropout_out'

    def setup(self):
        self.encoder = nn.Embed(self.ntoken, self.ninp,
                                embedding_init=lambda key, shape, dtype:
                                    jax.random.uniform(key, shape, dtype) * INITRANGE * 2 - INITRANGE)
        assert self.ninp == self.nhid == self.nhidlast

        self.rnn = MdRnn(self.ninp, self.nhid, self.dropouth, self.dropoutx,
                         self.training, self.genotype)
        
        self.decoder = nn.Dense(self.ntoken, use_bias=False)  # TODO: note that we do not use bias
        self.embedded_dropout = nn.Dropout(rate=self.dropoute)

    def __call__(self, input, hidden):
        """
        returns: (seq_len, bs, ntokens), (bs, nhid), [(seq_len, bs, nhid)], same
        """
        batch_size = input.shape[1]

        emb = self.encoder(input)
        emb = self.embedded_dropout(emb, deterministic=not self.training)
        emb = locked_dropout(emb, self.training, self.make_rng(self.rng_collection_emb),
                             self.dropouti)


        raw_output = emb
        raw_outputs = []
        outputs = []
        raw_output = self.rnn(raw_output, hidden)[1]
        raw_outputs.append(raw_output)

        output = locked_dropout(raw_output, self.training, self.make_rng(self.rng_collection_out),
                                self.dropout)
        outputs.append(output)

        # weight sharing between encoder and decoder
        dec_kernel = self.variables['params']['encoder']['embedding'].T
        logit = self.decoder.apply({'params': {'kernel': dec_kernel}},
                                   output.reshape(-1, self.ninp))
        log_prob = nn.activation.log_softmax(logit, axis=-1)
        model_output = log_prob
        model_output = model_output.reshape(-1, batch_size, self.ntoken)

        return model_output, hidden, raw_outputs, outputs

    def _loss(self, input, hidden, target):
        log_prob, hidden_next, raw_outs, _ = self(input, hidden)
        loss = -jnp.take_along_axis(log_prob.reshape(-1, log_prob.shape[-1]),
                                   target.reshape(-1)[..., None], axis=-1).mean()
        return loss, hidden_next, raw_outs
    
    def init_hidden(self, batch_size: int):
        return jnp.zeros((batch_size, self.nhid))


def export_single_architectute(params):
    alphas_dict = {k: v for k, v in params['params']['rnn']['cell']['cell'].items() if 'ops' in k}
    op_names = ['relu', 'tanh', 'none', 'sigmoid', 'identity']
    node_exports = []
    for _, d in alphas_dict.items():
        alpha = d['alpha']
        W = jax.nn.softmax(alpha, axis=-1)
        W = np.array(W)
        none_idx = [i for i, name in enumerate(op_names) if name == 'none'][0]
        W[:, none_idx] = -float('inf')
        best_prev_node = W.max(-1).argmax()
        best_op_idx = W[best_prev_node].argmax()
        node_exports.append((op_names[best_op_idx], int(best_prev_node)))

    return node_exports
