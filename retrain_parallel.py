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
import model
import json
from torch.utils.tensorboard import SummaryWriter

from model import nll_lm_loss, triplet_loss, get_eos_embeds
from data import BatchParallelLoader, ParallelSentenceCorpus
from utils import create_exp_dir, save_checkpoint
from utils import repackage_hidden, create_exp_dir, save_checkpoint


parser = argparse.ArgumentParser(
    description='PyTorch IWSLT14 Language Model')
parser.add_argument('--arch_path', type=str, required=True,
                    default='final_architecture.json',
                    help='architecture path')
parser.add_argument('--data', type=str, default='data/iwslt14/en_de_parallel',
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
parser.add_argument('--n_tokens', type=int, default=2_000, metavar='N',
                    help='number of tokens per update')
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
parser.add_argument('--beta_contr', type=float, default=1.0,
                    help='contrastive regularizer coefficient')
parser.add_argument('--wdecay', type=float, default=8e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize

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

par_corpus = ParallelSentenceCorpus(args.data)

eval_n_tokens = 2000
test_n_tokens = 2000
train_loader = BatchParallelLoader(
    par_corpus.train_parallel, args.n_tokens, device=args.device, max_len=70)
valid_loader = BatchParallelLoader(
    par_corpus.valid_parallel, eval_n_tokens, device=args.device)
test_loader = BatchParallelLoader(
    par_corpus.valid_parallel, test_n_tokens, device=args.device)


ntokens = len(par_corpus.dictionary)
genotype = json.load(open(args.arch_path))
model = model.MdRnnModel(ntoken=ntokens, ninp=args.emsize, nhid=args.nhid,
                         nhidlast=args.nhid, dropout=args.dropout, dropouth=args.dropouth, dropoutx=args.dropoutx,
                         dropouti=args.dropouti, dropoute=args.dropoute, genotype=genotype, n_domains=1).to(args.device)

total_params = sum(x.data.nelement() for x in model.parameters())
logger.info('Args: {}'.format(args))
logger.info('Model total parameters: {}'.format(total_params))
logger.info('Genotype: {}'.format(genotype))


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
    total_loss = 0
    start_time = time.time()
    model.train()
    # TODO: think of the last small batch
    for i, (en_train, de_train) in enumerate(train_loader):
        if i == len(train_loader) - 1:
            continue
        hidden = model.init_hidden(en_train.shape[0])
        hidden = repackage_hidden(hidden)

        log_en, _, hs_en, dropped_hs_en = model(
            en_train.t(), hidden, 0, return_h=True)
        log_de, _, hs_de, dropped_hs_de = model(
            de_train.t(), hidden, 1, return_h=True)

        raw_loss = nll_lm_loss(log_en.transpose(0, 1), en_train)
        raw_loss += nll_lm_loss(log_de.transpose(0, 1), de_train)
        raw_loss /= 2

        # contrastive regularization
        contr_loss = triplet_loss(get_eos_embeds(hs_en[0].transpose(0, 1), en_train),
                                  get_eos_embeds(hs_de[0].transpose(0, 1), de_train)) * args.beta_contr

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

        optimizer.zero_grad()
        total_loss = raw_loss + reg_loss + contr_loss
        total_loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if np.isnan(total_loss.item()):
            raise

        gc.collect()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = raw_loss.item()
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, i, len(
                                train_loader) // args.bptt, optimizer.param_groups[0]['lr'],
                            elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            global global_step
            writer.add_scalar('train/ppl', math.exp(cur_loss), global_step)
            writer.add_scalar('train/contr', contr_loss.item(), global_step)
            global_step += 1
            total_loss = 0
            start_time = time.time()


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
            model.load_state_dict(torch.load(
                os.path.join(args.save, 'model.pt')))
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

            val_loss2_dict = evaluate(valid_loader)
            logger.info('-' * 89)
            for key, val_loss2 in val_loss2_dict.items():
                logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} {:5.2f} | '
                            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                       key,
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
            val_loss_dict = evaluate(valid_loader)
            logger.info('-' * 89)
            for key, val_loss in val_loss_dict.items():
                logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {} {:5.2f} | '
                            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                       key,
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
test_loss_dict = evaluate(test_loader)
logger.info('=' * 89)
for key, test_loss in test_loss_dict.items():
    logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
logger.info('=' * 89)
