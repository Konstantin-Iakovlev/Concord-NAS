import argparse
import os
import sys
import glob
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

from data import BatchParallelLoader, ParallelSentenceCorpus
from model import MdRnnModel, triplet_loss, nll_lm_loss, get_eos_embeds, struct_reg_loss, struct_intersect_loss

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

parser = argparse.ArgumentParser(
    description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data', type=str, default='data/iwslt14/en_de_parallel',
                    help='location of the data corpus')
parser.add_argument('--max_len', type=int, default=500,
                    help='maximal sentence length for each language')
parser.add_argument('--min_len', type=int, default=0,
                    help='minimal sentence length for each language')
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
parser.add_argument('--n_tokens', type=int, default=256, metavar='N',
                    help='number of tokens per update')
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
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--beta_contr', type=float, default=1.0,
                    help='contrastive regularizer coefficient')
parser.add_argument('--beta_struct', type=float, default=1.0,
                    help='structural regularizer coefficient')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
formatter = logging.Formatter(log_format)
logger = logging.getLogger('search-parallel')
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

eval_n_tokens = 2000
test_n_tokens = 2000

par_corpus = ParallelSentenceCorpus(args.data)
train_loader = BatchParallelLoader(
    par_corpus.train_parallel, args.n_tokens, device=args.device, max_len=args.max_len, min_len=args.min_len)
search_loader = BatchParallelLoader(
    par_corpus.valid_parallel, args.n_tokens, device=args.device, max_len=args.max_len, min_len=args.min_len)
valid_loader = BatchParallelLoader(
    par_corpus.valid_parallel, eval_n_tokens, device=args.device)
test_loader = BatchParallelLoader(
    par_corpus.valid_parallel, test_n_tokens, device=args.device)

ntokens = len(par_corpus.dictionary)
model = MdRnnModel(ntokens, args.emsize, args.nhid, args.nhidlast,
                   args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute,
                   genotype=None, n_domains=2).to(args.device)
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
def evaluate(data_source: BatchParallelLoader):
    model.eval()
    total_loss_en = total_loss_de = 0
    n_total_en = n_total_de = 0
    for _, (en_batch, de_batch) in enumerate(data_source):
        hidden = model.init_hidden(en_batch.shape[0])
        log_en, _ = model(en_batch.t(), hidden, 0)
        n_en_tokens = (en_batch != data_source.pad_id).sum().item()
        total_loss_en += nll_lm_loss(log_en.transpose(0, 1),
                                     en_batch) * n_en_tokens

        log_de, _ = model(de_batch.t(), hidden, 1)
        n_de_tokens = (de_batch != data_source.pad_id).sum().item()
        total_loss_de += nll_lm_loss(log_de.transpose(0, 1),
                                     de_batch) * n_de_tokens
        n_total_en += n_en_tokens
        n_total_de += n_de_tokens

    return {'en_loss': total_loss_en.item() / n_total_en, 'de_loss': total_loss_de.item() / n_total_de}


def train():
    model.train()
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    valid_iterator = iter(search_loader)
    for i, (en_train, de_train) in enumerate(train_loader):
        hidden = model.init_hidden(en_train.shape[0])

        # architecture step; TODO: unroll
        arch_optimizer.zero_grad()
        try:
            en_val, de_val = next(valid_iterator)
        except:
            valid_iterator = iter(valid_iterator)
            en_val, de_val = next(valid_iterator)

        hidden_valid = model.init_hidden(en_val.shape[0])
        log_en, _, raw_outputs_en, _ = model(
            en_val.t(), hidden_valid, 0, return_h=True)
        arch_loss = nll_lm_loss(log_en.transpose(0, 1), en_val)
        log_de, _, raw_outputs_de, _ = model(
            de_val.t(), hidden_valid, 1, return_h=True)
        arch_loss += nll_lm_loss(log_de.transpose(0, 1), de_val)
        contr_loss = triplet_loss(get_eos_embeds(raw_outputs_en[0].transpose(0, 1), en_val),
                                  get_eos_embeds(raw_outputs_de[0].transpose(0, 1), de_val))
        # structural regularization
        alphas = [[t for key, t in model.struct_named_parameters() if key.split(
            '.')[-1] == d and 'alpha' in key] for d in ['0', '1']]
        betas = [[t for key, t in model.struct_named_parameters() if key.split(
            '.')[-1] == d and 'beta' in key] for d in ['0', '1']]
        struct_loss = struct_intersect_loss(alphas[0], betas[0], alphas[1], betas[1])
        (arch_loss / 2 + contr_loss * args.beta_contr + struct_loss * args.beta_struct).backward()
        arch_optimizer.step()

        optimizer.zero_grad()

        # we do not update hidden, because the sentence is not splitted
        log_en, _, hs_en, dropped_hs_en = model(
            en_train.t(), hidden, 0, return_h=True)
        raw_loss = nll_lm_loss(log_en.transpose(0, 1), en_train)
        log_de, _, hs_de, dropped_hs_de = model(
            de_train.t(), hidden, 1, return_h=True)
        raw_loss += nll_lm_loss(log_de.transpose(0, 1), de_train)
        raw_loss /= 2

        # activation regularization
        reg_loss = torch.tensor(0.0).to(en_train.device)
        if args.alpha > 0:
            reg_loss += sum(args.alpha * dropped_rnn_h.pow(2).mean()
                            for dropped_rnn_h in dropped_hs_en[-1:])
            reg_loss += sum(args.alpha * dropped_rnn_h.pow(2).mean()
                            for dropped_rnn_h in dropped_hs_de[-1:])

        # Temporal Activation Regularization (slowness)
        reg_loss += sum(args.beta *
                        (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in hs_en[-1:])
        reg_loss += sum(args.beta *
                        (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in hs_de[-1:])
        reg_loss /= 2

        # contrastive regularization
        contr_loss = triplet_loss(get_eos_embeds(hs_en[0].transpose(0, 1), en_train),
                                  get_eos_embeds(hs_de[0].transpose(0, 1), de_train))

        total_loss = raw_loss + reg_loss + contr_loss * args.beta_contr
        total_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        gc.collect()

        if i % args.log_interval == 0 and i > 0:
            logger.info(model.export())
            cur_loss = raw_loss.item()
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, i, len(
                                train_loader), optimizer.param_groups[0]['lr'],
                            elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            global global_step
            writer.add_scalar('train/ppl', math.exp(cur_loss), global_step)
            writer.add_scalar('train/contr', contr_loss.item(), global_step)
            writer.add_scalar('train/struct', struct_loss.item(), global_step)
            global_step += 1
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
writer = SummaryWriter(log_dir=args.save)
global_step = 0

optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr, weight_decay=args.wdecay)

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()

    val_loss_dict = evaluate(valid_loader)
    logger.info('-' * 89)
    for key, val_loss in val_loss_dict.items():
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), key,
                                               val_loss, math.exp(val_loss)))
        writer.add_scalar('val/ppl', math.exp(val_loss))
    logger.info('-' * 89)

    alphas = [[t for key, t in model.struct_named_parameters() if key.split(
        '.')[-1] == d and 'alpha' in key] for d in ['0', '1']]
    betas = [[t for key, t in model.struct_named_parameters() if key.split(
        '.')[-1] == d and 'beta' in key] for d in ['0', '1']]
    struct_loss = struct_intersect_loss(alphas[0], betas[0], alphas[1], betas[1])
    if val_loss + struct_loss * args.beta_struct < stored_loss:
        save_checkpoint(model, optimizer, epoch, args.save)
        logger.info('Saving Normal!')
        stored_loss = val_loss

    best_val_loss.append(val_loss)
