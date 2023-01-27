import argparse
import glob
import math
from utils import create_exp_dir, get_batch
import time
import sys
import os

import jax
import flax
import jax.numpy as jnp

import data
import gc
import logging

from model import MdRnnModel
from utils import batchify

from flax.training import train_state
import optax

from typing import Any
from genotypes import STEPS

from typing import Any, Callable

from flax import core, struct
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(
    description='Flax PennTreeBank Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
# parser.add_argument('--log-interval', type=int, default=50, metavar='N',
#                     help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# steup logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size)
search_data = batchify(corpus.valid, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, test_batch_size)


ntokens = len(corpus.dictionary)
model = MdRnnModel(ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, nhidlast=args.nhidlast,
                   training=True, dropout=args.dropout, dropouth=args.dropouth, dropoutx=args.dropoutx,
                   dropouti=args.dropouti, dropoute=args.dropoute, genotype=None)

# initialize a model
key = jax.random.PRNGKey(0)

seq_len = 10
bs = 4

inp = jnp.ones((seq_len, bs), dtype=jnp.int32)
hidden = jax.random.normal(key, (bs, args.emsize))

params = model.init({'locked_dropout': key, 'dropout': key, 'params': key,
                     'mask_2d': key}, inp, hidden)
print(jax.tree_map(lambda x: x.shape, params))
print(model.apply(params, inp, hidden, rngs={'locked_dropout': key, 'dropout': key, 'params': key,
                     'mask_2d': key}, mutable='batch_stats')[0][0].shape)

logging.info('initial genotype:')
# logging.info(model.export())

opt_weights = optax.masked(optax.chain(optax.clip_by_global_norm(args.clip),
                                       optax.add_decayed_weights(args.wdecay), optax.sgd(args.lr)),
                                       jax.tree_util.tree_map(lambda x: x.shape[-1] != STEPS, params['params']))

# TODO: add unroll (note that it give minor performance gain)
opt_alpha = optax.masked(optax.chain(optax.add_decayed_weights(args.arch_wdecay), optax.adam(args.arch_lr)),
                         jax.tree_util.tree_map(lambda x: x.shape[-1] == STEPS, params['params']))

writer = SummaryWriter(args.save)

class RnnNasTrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    opt_weights: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_weights_state: optax.OptState
    opt_alpha: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_alpha_state: optax.OptState

    batch_stats: Any
    mask_2d: jax.random.KeyArray
    dropout: jax.random.KeyArray
    locked_dropout: jax.random.KeyArray
    params_key: jax.random.KeyArray

    def apply_gradients_weights(self, *, grads, **kwargs):
        updates, new_weights_opt_state = self.opt_weights.update(grads,
                                                                 self.opt_weights_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_weights_state=new_weights_opt_state,
                            **kwargs)

    def apply_gradients_alpha(self, *, grads, **kwargs):
        updates, new_opt_alpha_state = self.opt_alpha.update(grads,
                                                             self.opt_alpha_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(params=new_params, opt_alpha_state=new_opt_alpha_state, **kwargs)

    @classmethod
    def create(cls, *, apply_fn, params, opt_weights, opt_alpha, **kwargs):
        opt_weihts_state = opt_weights.init(params)
        opt_alpha_state = opt_alpha.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            opt_weights=opt_weights,
            opt_weights_state=opt_weihts_state,
            opt_alpha=opt_alpha,
            opt_alpha_state=opt_alpha_state,
            **kwargs,
        )


state = RnnNasTrainState.create(apply_fn=model.apply,
                                params=params['params'], batch_stats=params['batch_stats'],
                                opt_weights=opt_weights,
                                opt_alpha=opt_alpha,
                                mask_2d=jax.random.PRNGKey(0),
                                dropout=jax.random.PRNGKey(0),
                                locked_dropout=jax.random.PRNGKey(0),
                                params_key=jax.random.PRNGKey(0),
                                )


@jax.jit
def train_step(state: RnnNasTrainState, batch):
    """Performs training step

    :param state: NasRnnState
    :param batch: dict with keys: src_train, trg_train, src_val, trg_val, hidden_train, hidden_val
    """
    def loss_weights(params):
        out, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
            batch['src_train'], batch['hidden_train'], batch['trg_train'], mutable=['batch_stats'],
            rngs={'dropout': state.dropout, 'mask_2d': state.mask_2d, 'params': state.params_key,
            'locked_dropout': state.locked_dropout}, method=model._loss)
        return out[0], updates
    
    # repeat yourself to avoid re-compilation
    def loss_alpha(params):
        out, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
            batch['src_val'], batch['hidden_val'], batch['trg_val'], mutable=['batch_stats'],
            rngs={'dropout': state.dropout, 'mask_2d': state.mask_2d, 'params': state.params_key,
            'locked_dropout': state.locked_dropout}, method=model._loss)
        return out[0], updates
    
    # update weights
    grad_weights_fn = jax.value_and_grad(loss_weights, has_aux=True)
    (loss_train, updates), grads = grad_weights_fn(state.params)
    state = state.apply_gradients_weights(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    # update alphas
    grad_alpha_fn = jax.value_and_grad(loss_alpha, has_aux=True)
    (loss_val, updates), grads = grad_alpha_fn(state.params)
    state = state.apply_gradients_alpha(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    return state, {'loss_train': loss_train, 'loss_val': loss_val}

# TODO: eval step and eval model for that. Same for test

def train_epoch(state, epoch: int):
    hidden_train = model.init_hidden(args.batch_size)
    hidden_valid = model.init_hidden(eval_batch_size)
    for i in range(train_data.shape[0] - 1 - args.bptt + 1):
        # TODO: check that we no not need seq_len=... in get_batch
        src_valid, trg_valid = get_batch(val_data, i, args)
        src_train, trg_train = get_batch(train_data, i, args)
        batch = {'src_train': src_train, 'trg_train': trg_train, 'src_val': src_valid, 'trg_val': trg_valid,
            'hidden_train': hidden_train, 'hidden_val': hidden_valid}
        start = time.time()
        state, losses = train_step(state, batch)
        elapsed = time.time() - start
        logging.info('Train | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, i, len(train_data) // args.bptt,
            elapsed * 1000, losses['loss_train'].item(), math.exp(losses['loss_train'].item())))
        writer.add_scalar('train/ppl_step', math.exp(losses['loss_train'].item()), state.step)
        writer.add_scalar('val/ppl_step', math.exp(losses['loss_val'].item()), state.step)
    return state

for epoch in range(1, args.epochs + 1):
    state = train_epoch(state, epoch)
