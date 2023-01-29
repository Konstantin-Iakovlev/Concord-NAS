import os
import gc
import sys
import glob
import time
import math
from typing import Any
import numpy as np
import logging
import argparse
import data
import model
import json
from model import MdRnnModel
import optax
import jax
import flax
from flax.training import train_state
import jax.numpy as jnp
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import batchify, get_batch, create_exp_dir


parser = argparse.ArgumentParser(
    description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--arch_path', type=str, required=True,
                    help='architecture path')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--device', type=str, default='cuda',
                    help='device: cpu, cuda')
parser.add_argument('--emsize', type=int, default=850,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=850,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=850,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1267,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=8e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

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
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, test_batch_size)

ntokens = len(corpus.dictionary)
genotype = json.load(open(args.arch_path))

ntokens = len(corpus.dictionary)
genotype = json.load(open(args.arch_path))

models = [MdRnnModel(ntoken=ntokens, ninp=args.emsize, nhid=args.nhid, nhidlast=args.nhidlast,
                     training=training, dropout=args.dropout, dropouth=args.dropouth, dropoutx=args.dropoutx,
                     dropouti=args.dropouti, dropoute=args.dropoute, genotype=genotype) for training in [True, False]]
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
logging.info('Genotype: {}'.format(genotype))


# TODO: implement ASGD with optax
opt_weights = optax.chain(optax.clip_by_global_norm(args.clip), optax.add_decayed_weights(args.wdecay),
                          optax.sgd(args.lr))


writer = SummaryWriter(args.save)

class RnnRetrainTrainState(train_state.TrainState):
    batch_stats: Any
    mask_2d: jax.random.KeyArray
    dropout: jax.random.KeyArray
    locked_dropout_emb: jax.random.KeyArray
    locked_dropout_out: jax.random.KeyArray
    params_key: jax.random.KeyArray


state = RnnRetrainTrainState.create(apply_fn=model.apply,
                                    params=params['params'],
                                    batch_stats=params['batch_stats'],
                                    tx=opt_weights,
                                    mask_2d=jax.random.PRNGKey(args.seed),
                                    dropout=jax.random.PRNGKey(args.seed + 1),
                                    locked_dropout_emb=jax.random.PRNGKey(args.seed + 2),
                                    locked_dropout_out=jax.random.PRNGKey(args.seed + 3),
                                    params_key=jax.random.PRNGKey(args.seed + 4),
                                    )


@jax.jit
def train_step(state: RnnRetrainTrainState, batch):
    """Performs training step

    :param state: NasRnnState
    :param batch: dict with keys: src_train, trg_train, hidden_train
    """
    dropout = jax.random.fold_in(key=state.dropout, data=state.step)
    mask_2d = jax.random.fold_in(key=state.mask_2d, data=state.step)
    locked_dropout_emb = jax.random.fold_in(
        key=state.locked_dropout_emb, data=state.step)
    locked_dropout_out = jax.random.fold_in(
        key=state.locked_dropout_out, data=state.step)
    def loss(params):
        out, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                      batch['src_train'], batch['hidden_train'], batch['trg_train'], mutable=[
                                          'batch_stats'],
                                      rngs={'dropout': dropout, 'mask_2d': mask_2d, 'params': state.params_key,
                                            'locked_dropout_emb': locked_dropout_emb, 'locked_dropout_out': locked_dropout_out}, method=model._loss)
        # add regularizer. Note that raw_outputs is [single_tensor]
        raw_outs = out[-1]
        reg_loss = jnp.stack([jnp.power(rnn_h[1:] - rnn_h[:-1], 2).mean() for rnn_h in raw_outs]).sum()
        return out[0] + args.beta * reg_loss, updates

    grad_fn = jax.value_and_grad(loss, has_aux=True)
    (loss_train, updates), grads = grad_fn(state.params)  # loss w/ regularizer
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    return state, {'loss_train': loss_train}

@jax.jit
def val_step(state: RnnRetrainTrainState, batch):
    out, _ = eval_model.apply({'params': state.params, 'batch_stats': state.batch_stats},
                                    batch['src_val'], batch['hidden_val'], batch['trg_val'], mutable=[
                                        'batch_stats'],
                                    rngs={'dropout': state.dropout, 'mask_2d': state.mask_2d, 'params': state.params_key,
                                        'locked_dropout_emb': state.locked_dropout_emb, 'locked_dropout_out': state.locked_dropout_out},
                                        method=eval_model._loss)
    return out[0]


def train_epoch(state: RnnRetrainTrainState, epoch: int):
    hidden_train = model.init_hidden(args.batch_size)
    for i in range(0, train_data.shape[0] - 1 - args.bptt + 1, args.bptt):
        # TODO: check that we no not need seq_len=... in get_batch
        src_train, trg_train = get_batch(train_data, i, args)
        batch = {'src_train': src_train, 'trg_train': trg_train, 'hidden_train': hidden_train}
        start = time.time()
        state, losses = train_step(state, batch)
        elapsed = time.time() - start
        loss_train = min(losses['loss_train'].item(), 50)
        logging.info('Train | epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                     'loss {:5.2f} | ppl {:8.2f}'.format(
                         epoch, i // args.bptt + 1, len(train_data) // args.bptt,
                         elapsed * 1000, loss_train, math.exp(loss_train)))
        writer.add_scalar('train/ppl_step', math.exp(loss_train), state.step)
    return state


def val_epoch(state: RnnRetrainTrainState, epoch: int):
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
