import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from argparse import ArgumentParser
from dataset import NliDataset
from transformers import AutoTokenizer, AutoModel
from model import AdaBertStudent, AdaBertStudentMLM, evaluate, distil_loss
import numpy as np
from tqdm.auto import tqdm
import json
from transformers import get_cosine_schedule_with_warmup


def main():
    parser = ArgumentParser()
    parser.add_argument('--arch_path', required=True)
    parser.add_argument('--epochs', required=False, type=int, default=20)
    parser.add_argument('--pretrain_steps', required=False, type=int, default=10_000)
    parser.add_argument('--seed', required=False, type=int, default=0)
    parser.add_argument('--valid_freq', required=False, type=int, default=200)
    parser.add_argument('--device', required=False, type=str, default='cuda')
    parser.add_argument('--ds_name', required=False, type=str, default='xnli.en')
    args = parser.parse_args()
    with open(args.arch_path) as f:
        genotype = json.load(f)

    max_length = 128
    batch_size = 128
    num_cells = 1
    lr = 1e-1
    clip_value = 1.0
    device = args.device
    epochs = args.epochs
    log_freq = 20
    valid_freq = args.valid_freq
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    ds_name, *ds_configs = args.ds_name.split('.')
    num_domains = len(ds_configs)
    train_ds = [NliDataset(tokenizer, ds_name, ds_config=cfg, split='train', max_length=max_length) for cfg in ds_configs]
    val_ds = [NliDataset(tokenizer, ds_name, ds_config=cfg, split='validation', max_length=max_length) for cfg in ds_configs]
    train_dl = [DataLoader(ds, batch_size=batch_size, collate_fn=train_ds[0].collate_fn, shuffle=True) for ds in train_ds]
    val_dl = [DataLoader(ds, batch_size=batch_size, collate_fn=val_ds[0].collate_fn, shuffle=False) for ds in val_ds]

    m = AutoModel.from_pretrained('bert-base-multilingual-cased', cache_dir='.')
    pretrained_token_embeddigns = m.embeddings.word_embeddings.weight
    pretrained_pos_embeddigns = m.embeddings.position_embeddings.weight
    
    ### Pretrain ###
    model = AdaBertStudentMLM(tokenizer.vocab_size, False,
                        len(tokenizer), num_domains, pretrained_token_embeddigns,
                        pretrained_pos_embeddigns, num_cells=num_cells,
                        genotype=genotype, dropout_p=0.05).to(device)
    print(sum([p.numel() for p in model.parameters()]) // 1000 / 1000, 'M')

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.pretrain_steps)
    criterion = nn.CrossEntropyLoss()

    total_steps = 0
    val_accs = []
    losses = []
    model.train()
    for epoch in range(10):
        if total_steps >= args.pretrain_steps:
            break
        for i, batches in enumerate(tqdm(zip(*train_dl), desc=f'epoch {epoch + 1}/{epochs}', total=len(train_dl[0]))):
            for domain_idx, batch in enumerate(batches):
                batch = {k: batch[k].to(device) for k in batch}
                inp_ids = batch['inp_ids']
                inp_ids_orig = inp_ids.clone()
                # add mask tokens to premise
                is_mask = (torch.rand_like(inp_ids[0].float()) < 0.15)
                inp_ids[0] = torch.where(is_mask, tokenizer.mask_token_id, inp_ids[0])
                type_ids = batch['type_ids']
                msk = batch['att'].max(0).values

                out = model(inp_ids, type_ids, msk, domain_idx)

                logits = model.lm_head(out.reshape(-1, out.shape[-1])[is_mask.reshape(-1)])
                labels = inp_ids_orig[0].reshape(-1)[is_mask.reshape(-1)]
                optimizer.zero_grad()

                loss = criterion(logits, labels)

                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                lr_scheduler.step()

                total_steps += 1
            if total_steps >= args.pretrain_steps:
                break
    print('Finished pretrain', losses[0], losses[-1])
    torch.save(f=f'pretrain_{seed}.ckpt', obj=model.state_dict())

    model = AdaBertStudent(tokenizer.vocab_size, train_ds[0].task_to_keys[ds_name][-1] is not None,
                           3, num_domains, pretrained_token_embeddigns,
                           pretrained_pos_embeddigns, num_cells=num_cells,
                           genotype=genotype, dropout_p=0.05).to(device)
    model.load_state_dict(torch.load(f'pretrain_{seed}.ckpt', map_location='cpu'), strict=False)
    print(sum([p.numel() for p in model.parameters()]) // 1000 / 1000, 'M')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs * len(train_dl[0]) * len(ds_configs))
    criterion = nn.CrossEntropyLoss()
    
    total_steps = 0
    val_accs = []
    model.train()
    for epoch in range(epochs):
        for i, batches in enumerate(tqdm(zip(*train_dl), desc=f'epoch {epoch + 1}/{epochs}', total=len(train_dl[0]))):
            for domain_idx, batch in enumerate(batches):
                batch = {k: batch[k].to(device) for k in batch}
                pi_logits = batch['logits']
                inp_ids = batch['inp_ids']
                type_ids = batch['type_ids']
                msk = batch['att'].max(0).values
                p_logits = model(inp_ids, type_ids, msk, domain_idx)
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
                    val_acc_arr = evaluate(model, val_dl, device)
                    model.train()
                    val_accs.append(np.mean(val_acc_arr))
                    for val_acc, cfg in zip(val_acc_arr, ds_configs):
                        print(f'Cfg: {cfg}, step: {total_steps}, val acc: {round(val_acc, 4)}, best avg: {round(max(val_accs), 4)}')

    print('Finished with', round(max(val_accs), 4))


if __name__ == '__main__':
    main()
