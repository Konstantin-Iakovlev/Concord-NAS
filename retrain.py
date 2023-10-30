import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from argparse import ArgumentParser
from dataset import RteDataset
from transformers import AutoTokenizer, AutoModel
from model import AdaBertStudent
import numpy as np
from tqdm.auto import tqdm
import json


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def collate_fn(data_points, tok: AutoTokenizer, max_length=128, ds_name='qnli'):  # pair = True
    k1, k2 = task_to_keys[ds_name]
    if k2 is not None:
        tok_out = tok([d[k1] for d in data_points], [d[k2] for d in data_points], return_tensors='pt',
                    padding=True, max_length=max_length, truncation=True)
        inp_ids = torch.stack([tok_out['input_ids'], tok_out['input_ids']], dim=0)
        type_ids = torch.stack([tok_out['token_type_ids'], tok_out['token_type_ids']], dim=0)
    else:
        tok_out = tok([d[k1] for d in data_points], return_tensors='pt',padding=True,
                      max_length=max_length, truncation=True)
        inp_ids = tok_out['input_ids']
        type_ids = tok_out['token_type_ids']
    logits = torch.tensor(np.stack([b['logits'] for b in data_points], axis=0))
    return {'labels': torch.LongTensor([d['label'] for d in data_points]), 'inp_ids': inp_ids,
            'att': (inp_ids != tok.pad_token_id).long(), 'logits': logits, 'type_ids': type_ids}


def distil_loss(pi_logits: torch.Tensor, p_scores: torch.Tensor):
    pi_probs = pi_logits.softmax(-1)
    return -(pi_probs * torch.log_softmax(p_scores, -1)).sum(-1).mean()


def evaluate(model, dl, device):
    model.eval()
    n_total = 0
    n_corr = 0
    for batch in dl:
        batch = {k: batch[k].to(device) for k in batch}
        pi_logits = batch['logits']
        inp_ids = batch['inp_ids']
        type_ids = batch['type_ids']
        msk = batch['att'].max(0).values
        with torch.no_grad():
            p_logits = model(inp_ids, type_ids, msk)
        n_total += p_logits.shape[0]
        n_corr += (pi_logits.argmax(-1) == p_logits.argmax(-1)).sum().item()
    return n_corr / n_total


def main():
    parser = ArgumentParser()
    parser.add_argument('--arch_path', required=True)
    parser.add_argument('--epochs', required=False, type=int, default=20)
    parser.add_argument('--valid_freq', required=False, type=int, default=200)
    parser.add_argument('--device', required=False, type=str, default='cuda')
    parser.add_argument('--ds_name', required=False, type=str, default='qnli')
    args = parser.parse_args()
    with open(args.arch_path) as f:
        genotype = json.load(f)

    max_length = 128
    batch_size = 128
    lr = 0.025
    clip_value = 1.0
    device = args.device
    epochs = args.epochs
    log_freq = 20
    valid_freq = args.valid_freq
    seed = 2

    ### RTE dataset
    # teacher val acc: 0.6498194945848376
    #### 30 epochs
    # random(0) => 2: 0.5704
    # random(1) => 2: 0.5957
    # random(2) => 2: 0.574
    # random(3) => 2: 0.6137
    # random(4) => 2: 0.6029 
    # => random 0.5913 +- 0.0167


    torch.manual_seed(seed)
    np.random.seed(seed)


    train_ds = RteDataset(args.ds_name, split='train')
    val_ds = RteDataset(args.ds_name, split='validation')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, tokenizer,
                                                                                           max_length, args.ds_name), shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=lambda b: collate_fn(b, tokenizer,
                                                                                       max_length, args.ds_name), shuffle=False)

    # genotype = [[('conv7x7', 0), ('maxpool', 1)],
    #             [('maxpool', 1), ('maxpool', 2)],
    #             [('conv3x3', 1), ('dilconv3x3', 3)]]
    if args.ds_name == 'qnli':
        m = AutoModel.from_pretrained('gchhablani/bert-base-cased-finetuned-qnli', cache_dir='.')
    elif args.ds_name == 'rte':
        m = AutoModel.from_pretrained('gchhablani/bert-base-cased-finetuned-rte', cache_dir='.')
    elif args.ds_name == 'sst2':
        m = AutoModel.from_pretrained('gchhablani/bert-base-cased-finetuned-sst2', cache_dir='.')
    else:
        raise ValueError(f'Unknown dataset {args.ds_name}')
    pretrained_token_embeddigns = m.embeddings.word_embeddings.weight
    pretrained_pos_embeddigns = m.embeddings.position_embeddings.weight
    model = AdaBertStudent(tokenizer.vocab_size, task_to_keys[args.ds_name][-1] is not None,
                           2, pretrained_token_embeddigns,
                           pretrained_pos_embeddigns,
                           genotype=genotype, dropout_p=0.0).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_dl), eta_min=1e-3)
    criterion = nn.CrossEntropyLoss()

    
    total_steps = 0
    val_accs = []
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dl)):
            batch = {k: batch[k].to(device) for k in batch}
            pi_logits = batch['logits']
            inp_ids = batch['inp_ids']
            type_ids = batch['type_ids']
            msk = batch['att'].max(0).values
            p_logits = model(inp_ids, type_ids, msk)
            optimizer.zero_grad()
            loss = 0.2 * criterion(p_logits, batch['labels']) + 0.8 * distil_loss(pi_logits, p_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            lr_scheduler.step()

            total_steps += 1
            if i % log_freq == 0 and i > 0:
                print('Train acc', round((pi_logits.argmax(-1) == p_logits.argmax(-1)).float().mean().item(), 4))
        
            if total_steps % valid_freq == 0:
                val_acc = evaluate(model, val_dl, device)
                val_accs.append(val_acc)
                print(f'Step: {total_steps}, val acc: {round(val_acc, 4)}, best: {round(max(val_accs), 4)}')

    print('Finished with', round(max(val_accs), 4))


if __name__ == '__main__':
    main()
