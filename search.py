import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import gc

import data
# import model_search as model
from model import MdRnnModel

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--device', type=str, default='cuda',
                    help='device: cpu, cuda')
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
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
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
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
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

log_format = '%(asctime)s %(message)s'
formatter = logging.Formatter(log_format)
logger = logging.getLogger('search')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(formatter)
fh.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)

logger.addHandler(fh)
logger.addHandler(sh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)
    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.manual_seed_all(args.seed)

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size, args)
search_data = batchify(corpus.valid, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


ntokens = len(corpus.dictionary)
model = MdRnnModel(ntokens, args.emsize, args.nhid, args.nhidlast, 
                    args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute,
                    genotype=None, n_domains=1).to(args.device)
# TODO: add domains to argparser, domain-agnostic dataset and add domain_idx to forward pass

size = 0
for p in model.parameters():
    size += p.nelement()
logger.info('param size: {}'.format(size))
logger.info('initial genotype:')
logger.info(model.export())

arch_optimizer = torch.optim.Adam(model.struct_parameters(), lr=args.arch_lr,
    weight_decay=args.arch_wdecay)
# TODO: add architecture optimizer (w/o unroll) and weight optimizer

total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))


@torch.no_grad()
def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        data = data.to(args.device)
        targets = targets.to(args.device)

        targets = targets.view(-1)

        log_prob, hidden = model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    hidden_valid = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        data_valid, targets_valid = get_batch(search_data, i % (search_data.size(0) - 1), args)
        data_valid = data_valid.to(args.device)
        targets_valid = targets_valid.to(args.device)

        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        data = data.to(args.device)
        targets = targets.to(args.device)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
            cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])
            hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])

            # architecture step
            arch_optimizer.zero_grad()
            arch_loss, _ = model._loss(hidden_valid[s_id], cur_data_valid, cur_targets_valid, domain_idx=0)
            arch_loss.backward()
            arch_optimizer.step()



            # assuming small_batch_size = batch_size so we don't accumulate gradients
            optimizer.zero_grad()
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
              loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            logger.info(model.export())
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            global global_step
            writer.add_scalar('train/ppl', math.exp(cur_loss), global_step)
            global_step += 1
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
writer = SummaryWriter(log_dir=args.save)
global_step = 0

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()

    val_loss = evaluate(val_data, eval_batch_size)
    logger.info('-' * 89)
    logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    writer.add_scalar('val/ppl', math.exp(val_loss))
    logger.info('-' * 89)

    if val_loss < stored_loss:
        save_checkpoint(model, optimizer, epoch, args.save)
        logger.info('Saving Normal!')
        stored_loss = val_loss

    best_val_loss.append(val_loss)
