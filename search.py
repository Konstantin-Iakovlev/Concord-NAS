import argparse
import glob
import json
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
from model import export_single_architectute
import numpy as np
from tqdm import tqdm


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

models = [MdRnnModel(ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, nhidlast=args.nhidlast,
                   training=training, dropout=args.dropout, dropouth=args.dropouth, dropoutx=args.dropoutx,
                   dropouti=args.dropouti, dropoute=args.dropoute, genotype=None) for training in [True, False]]
model, eval_model = models
# initialize a models
key = jax.random.PRNGKey(0)

seq_len = 10
bs = 4

inp = jnp.ones((seq_len, bs), dtype=jnp.int32)
hidden = jax.random.normal(key, (bs, args.emsize))

params = model.init({'locked_dropout_emb': key, 'locked_dropout_out': key, 'dropout': key, 'params': key,
                     'mask_2d': key}, inp, hidden)
_ = eval_model.init({'locked_dropout_emb': key, 'locked_dropout_out': key, 'dropout': key, 'params': key,
                     'mask_2d': key}, inp, hidden)
# print(jax.tree_map(lambda x: x.shape, params))
print(model.apply(params, inp, hidden, rngs={'locked_dropout_emb': key, 'locked_dropout_out': key,
                                             'dropout': key, 'params': key,
                                             'mask_2d': key}, mutable='batch_stats')[0][0].shape)

logging.info('Args: {}'.format(args))
logging.info('initial genotype:')
logging.info(export_single_architectute(params))

opt_weights = optax.masked(optax.chain(optax.clip_by_global_norm(args.clip),
                                       optax.add_decayed_weights(args.wdecay), optax.sgd(args.lr)),
                           jax.tree_util.tree_map(lambda x: x.shape[-1] != 5, params['params']))

# TODO: add unroll (note that it give minor performance gain)
opt_alpha = optax.masked(optax.chain(optax.add_decayed_weights(args.arch_wdecay), optax.adam(args.arch_lr)),
                         jax.tree_util.tree_map(lambda x: x.shape[-1] == 5, params['params']))

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
    locked_dropout_emb: jax.random.KeyArray
    locked_dropout_out: jax.random.KeyArray
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
                                mask_2d=jax.random.PRNGKey(args.seed),
                                dropout=jax.random.PRNGKey(args.seed + 1),
                                locked_dropout_emb=jax.random.PRNGKey(
                                    args.seed + 2),
                                locked_dropout_out=jax.random.PRNGKey(
                                    args.seed + 3),
                                params_key=jax.random.PRNGKey(args.seed + 4),
                                )


@jax.jit
def train_step(state: RnnNasTrainState, batch):
    """Performs training step

    :param state: NasRnnState
    :param batch: dict with keys: src_train, trg_train, src_val, trg_val, hidden_train, hidden_val
    """
    dropout = jax.random.fold_in(key=state.dropout, data=state.step)
    mask_2d = jax.random.fold_in(key=state.mask_2d, data=state.step)
    locked_dropout_emb = jax.random.fold_in(
        key=state.locked_dropout_emb, data=state.step)
    locked_dropout_out = jax.random.fold_in(
        key=state.locked_dropout_out, data=state.step)
    # TODO: reomove "params" from apply function
    # TODO: decouple alphas and weights for significant speedup

    def loss_weights(params):
        out, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                      batch['src_train'], batch['hidden_train'], batch['trg_train'], mutable=[
                                          'batch_stats'],
                                      rngs={'dropout': dropout, 'mask_2d': mask_2d, 'params': state.params_key,
                                            'locked_dropout_emb': locked_dropout_emb, 'locked_dropout_out': locked_dropout_out}, method=model._loss)
        # add regularizer. Note that raw_outputs is [single_tensor]
        raw_outs = out[-1]
        reg_loss = jnp.stack([jnp.power(rnn_h[1:] - rnn_h[:-1], 2).mean() for rnn_h in raw_outs]).sum()
        return out[0] + args.beta * reg_loss, updates

    # repeat yourself to avoid re-compilation
    dropout = jax.random.fold_in(key=dropout, data=state.step)
    mask_2d = jax.random.fold_in(key=mask_2d, data=state.step)
    locked_dropout_emb = jax.random.fold_in(
        key=locked_dropout_emb, data=state.step)
    locked_dropout_out = jax.random.fold_in(
        key=locked_dropout_out, data=state.step)

    def loss_alpha(params):
        out, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                      batch['src_val'], batch['hidden_val'], batch['trg_val'], mutable=[
                                          'batch_stats'],
                                      rngs={'dropout': dropout, 'mask_2d': mask_2d, 'params': state.params_key,
                                            'locked_dropout_emb': locked_dropout_emb, 'locked_dropout_out': locked_dropout_out},
                                      method=model._loss)
        return out[0], updates

    # update weights
    grad_weights_fn = jax.value_and_grad(loss_weights, has_aux=True)
    (loss_train, updates), grads = grad_weights_fn(state.params)  # loss w/ regularizer
    # manually zero alphas grads
    masked_grads = jax.tree_util.tree_map(lambda x: 0 if x.shape[-1] == 5 else x, grads)
    state = state.apply_gradients_weights(grads=masked_grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    # update alphas
    grad_alpha_fn = jax.value_and_grad(loss_alpha, has_aux=True)
    (loss_val, updates), grads = grad_alpha_fn(state.params)
    # manually mask weights grads
    masked_grads = jax.tree_util.tree_map(lambda x: 0 if x.shape[-1] != 5 else x, grads)
    state = state.apply_gradients_alpha(grads=masked_grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    return state, {'loss_train': loss_train, 'loss_val': loss_val}

# TODO: eval step and eval model for that. Same for test


@jax.jit
def val_step(state: RnnNasTrainState, batch):
    out, _ = eval_model.apply({'params': state.params, 'batch_stats': state.batch_stats},
                                    batch['src_val'], batch['hidden_val'], batch['trg_val'], mutable=[
                                        'batch_stats'],
                                    rngs={'dropout': state.dropout, 'mask_2d': state.mask_2d, 'params': state.params_key,
                                        'locked_dropout_emb': state.locked_dropout_emb, 'locked_dropout_out': state.locked_dropout_out},
                                        method=eval_model._loss)
    return out[0]


def train_epoch(state: RnnNasTrainState, epoch: int):
    hidden_train = model.init_hidden(args.batch_size)
    hidden_valid = model.init_hidden(args.batch_size)
    for i in range(0, train_data.shape[0] - 1 - args.bptt + 1, args.bptt):
        # TODO: check that we no not need seq_len=... in get_batch
        src_valid, trg_valid = get_batch(search_data, i, args)
        src_train, trg_train = get_batch(train_data, i, args)
        batch = {'src_train': src_train, 'trg_train': trg_train, 'src_val': src_valid, 'trg_val': trg_valid,
                 'hidden_train': hidden_train, 'hidden_val': hidden_valid}
        start = time.time()
        state, losses = train_step(state, batch)
        elapsed = time.time() - start
        loss_train = min(losses['loss_train'].item(), 50)
        loss_val = min(losses['loss_val'].item(), 50)
        logging.info('Train | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                     'loss {:5.2f} | ppl {:8.2f}'.format(
                         epoch, i // args.bptt + 1, len(train_data) // args.bptt,
                         elapsed * 1000, loss_train, math.exp(loss_train)))
        logging.info(export_single_architectute({'params': state.params}))
        writer.add_scalar('train/ppl_step', math.exp(loss_train), state.step)
        writer.add_scalar('val/ppl_step', math.exp(loss_val), state.step)
    return state


def val_epoch(state: RnnNasTrainState, epoch: int):
    hidden_valid = model.init_hidden(eval_batch_size)
    total_loss = 0.0
    # Note that here we keep the last small batch
    for i in tqdm(range(0, val_data.shape[0] - 1, args.bptt), leave=False):
        src_valid, trg_valid = get_batch(val_data, i, args)
        batch = {'src_val': src_valid, 'hidden_val': hidden_valid, 'trg_val': trg_valid}
        loss_val = val_step(state, batch)
        total_loss += loss_val.item() * src_valid.shape[0]
    
    mean_loss = min(total_loss / val_data.shape[0], 50)
    logging.info('| end of epoch {:3d} | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, mean_loss, math.exp(mean_loss)))
    writer.add_scalar('val/ppl_epoch', math.exp(mean_loss), state.step)
    return mean_loss


for epoch in range(1, args.epochs + 1):
    state = train_epoch(state, epoch)
    val_epoch(state, epoch)

# export architecture
with open(os.path.join(args.save, 'final_architecture.json'), 'w') as fout:
    fout.write(json.dumps(export_single_architectute({'params': state.params})))
