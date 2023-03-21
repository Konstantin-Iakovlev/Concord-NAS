import os
import gc
import sys
import glob
import time
import math
import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import data
import model
import json
from torch.utils.tensorboard import SummaryWriter

from utils import (batchify, get_batch, repackage_hidden,
                   create_exp_dir, save_checkpoint, to_device)

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
parser.add_argument('--nonmono', type=int, default=5,
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
formatter = logging.Formatter(log_format)
logger = logging.getLogger('retrain')
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
    cudnn.enabled = True
    torch.cuda.manual_seed_all(args.seed)

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


ntokens = len(corpus.dictionary)
genotype = json.load(open(args.arch_path))
model = model.MdRnnModel(ntoken=ntokens, ninp=args.emsize, nhid=args.nhid,
                         nhidlast=args.nhid, dropout=args.dropout, dropouth=args.dropouth, dropoutx=args.dropoutx,
                         dropouti=args.dropouti, dropoute=args.dropoute, genotype=genotype, n_domains=1).to(args.device)  # TODO: n_domains to argparser

total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))
logger.info('Genotype: {}'.format(genotype))


@torch.no_grad()
def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = model(to_device(data, args.device), to_device(hidden, args.device))
        loss = nn.functional.nll_loss(
            log_prob.view(-1, log_prob.size(2)), targets.to(args.device)).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(
        args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous(
            ).view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = model(
                to_device(cur_data, args.device), to_device(hidden[s_id], args.device), return_h=True)
            raw_loss = nn.functional.nll_loss(
                log_prob.view(-1, log_prob.size(2)), cur_targets.to(args.device))

            loss = raw_loss
            # Activiation Regularization
            if args.alpha > 0:
                loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean()
                                  for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + \
                sum(args.beta * (rnn_h[1:] - rnn_h[:-1]
                                 ).pow(2).mean() for rnn_h in rnn_hs[-1:])
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

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2

        if np.isnan(total_loss.item()):
            raise

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                         'loss {:5.2f} | ppl {:8.2f}'.format(
                             epoch, batch, len(
                                 train_data) // args.bptt, optimizer.param_groups[0]['lr'],
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

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    epoch = 1
    while epoch < args.epochs + 1:
        epoch_start_time = time.time()
        try:
            train()
        except:
            logger.info('rolling back to the previous best model ...')
            model.load_state_dict(torch.load(os.path.join(args.save, 'model.pt')))
            model = model.to(args.device)

            optimizer_state = torch.load(
                os.path.join(args.save, 'optimizer.pt'))
            if 't0' in optimizer_state['param_groups'][0]:
                optimizer = torch.optim.ASGD(
                    model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=args.lr, weight_decay=args.wdecay)
            optimizer.load_state_dict(optimizer_state)

            epoch = torch.load(os.path.join(args.save, 'misc.pt'))['epoch']
            continue

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                    val_loss2, math.exp(val_loss2)))
            writer.add_scalar('val/ppl', math.exp(val_loss2), global_step)
            logger.info('-' * 89)

            if val_loss2 < stored_loss:
                save_checkpoint(model, optimizer, epoch, args.save)
                logger.info('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                         'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                    val_loss, math.exp(val_loss)))
            writer.add_scalar('val/ppl', math.exp(val_loss), global_step)
            logger.info('-' * 89)

            if val_loss < stored_loss:
                save_checkpoint(model, optimizer, epoch, args.save)
                logger.info('Saving Normal!')
                stored_loss = val_loss

            if 't0' not in optimizer.param_groups[0] and (len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logger.info('Switching!')
                optimizer = torch.optim.ASGD(
                    model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            best_val_loss.append(val_loss)

        epoch += 1

except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.info('Exiting from training early')

# Load the best saved model.
model.load_state_dict(torch.load(os.path.join(args.save, 'model.pt')))
model = model.to(args.device)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
logger.info('=' * 89)
logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logger.info('=' * 89)
