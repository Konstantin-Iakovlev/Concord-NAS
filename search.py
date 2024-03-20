import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from argparse import ArgumentParser
from dataset import NliDataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from model import AdaBertStudent, evaluate, distil_loss
import numpy as np
from tqdm.auto import tqdm
import json


def struct_regul(model: AdaBertStudent, eps=1e-1):
    reg = 0.0
    for node in model.cells[0].nodes:
        # decode using the edge norm params
        soft_beta = node.input_switch.alpha.softmax(-1)
        ids = torch.topk(soft_beta, 2, dim=-1).indices
        hard_beta = torch.scatter(torch.zeros_like(soft_beta), -1, ids, 1.0)
        hard_beta = torch.where(hard_beta == 0, eps, 1)
        beta_hat = soft_beta * hard_beta + (hard_beta - soft_beta * hard_beta).detach()
        for prev_idx_str, edge in node.edges.items():
            reg += torch.prod(edge.alpha.softmax(-1), dim=0).sum() * torch.prod(beta_hat[:, int(prev_idx_str)])
    return reg


def main():
    parser = ArgumentParser()
    parser.add_argument('--epochs', required=False, type=int, default=20)
    parser.add_argument('--seed', required=False, type=int, default=0)
    parser.add_argument('--valid_freq', required=False, type=int, default=200)
    parser.add_argument('--device', required=False, type=str, default='cuda')
    parser.add_argument('--ds_name', required=False, type=str, default='xnli;en')
    parser.add_argument('--lambda_reg', required=False, type=float, default=0.0)
    parser.add_argument('--curr_steps', required=False, type=int, default=3000)
    args = parser.parse_args()

    max_length = 128
    batch_size = 128
    lr = 1e-1
    clip_value = 1.0
    num_cells = 1
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
    search_dl = [DataLoader(ds, batch_size=batch_size, collate_fn=val_ds[0].collate_fn, shuffle=True) for ds in val_ds]

    m = AutoModel.from_pretrained('bert-base-multilingual-cased', cache_dir='.')
    pretrained_token_embeddigns = m.embeddings.word_embeddings.weight
    pretrained_pos_embeddigns = m.embeddings.position_embeddings.weight
    model = AdaBertStudent(tokenizer.vocab_size, train_ds[0].task_to_keys[ds_name][-1] is not None,
                           3, num_domains, pretrained_token_embeddigns,
                           pretrained_pos_embeddigns, num_cells=num_cells,
                           genotype=None, dropout_p=0.05).to(device)
    optimizer = torch.optim.SGD([p for name, p in model.named_parameters() if 'alpha' not in name], lr=lr, momentum=0.9,
                                weight_decay=1e-4)
    sch = get_cosine_schedule_with_warmup(optimizer, 0, epochs * len(train_dl[0]) * len(val_ds))
    optimizer_struct = torch.optim.Adam([p for name, p in model.named_parameters() if 'alpha' in name], lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    
    total_steps = 0
    val_accs = []
    best_arch = model.export()
    model.train()
    for epoch in range(epochs):
        for i, batches in enumerate(tqdm(zip(*train_dl), desc=f'epoch {epoch + 1}/{epochs}', total=len(train_dl[0]))):
            for domain_idx, batch in enumerate(batches):
                batch = {k: batch[k].to(device) for k in batch}
                pi_logits = batch['logits']
                inp_ids = batch['inp_ids']
                type_ids = batch['type_ids']
                msk = batch['att'].max(0).values
                p_logits = model(inp_ids, type_ids, msk, domain_idx if total_steps >= args.curr_steps else 0)

                # weights update
                optimizer.zero_grad()
                loss = 0.2 * criterion(p_logits, batch['labels']) + 0.8 * distil_loss(pi_logits, p_logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for name, p in model.named_parameters() if 'alpha' not in name], clip_value)
                optimizer.step()
                sch.step()

                # structure update
                for val_batch in search_dl[domain_idx]:
                    val_batch = {k: val_batch[k].to(device) for k in val_batch}
                    pi_logits = val_batch['logits']
                    inp_ids = val_batch['inp_ids']
                    type_ids = val_batch['type_ids']
                    msk = val_batch['att'].max(0).values
                    p_logits = model(inp_ids, type_ids, msk, domain_idx if total_steps >= args.curr_steps else 0)
                    optimizer_struct.zero_grad()
                    loss = 0.2 * criterion(p_logits, val_batch['labels']) + 0.8 * distil_loss(pi_logits, p_logits)
                    loss += args.lambda_reg * struct_regul(model)

                    loss.backward()
                    g_val_w = [p.grad.data if p.grad is not None else torch.zeros_like(p.data) \
                               for name, p in model.named_parameters() if 'alpha' not in name]
                    g_val_alpha = [p.grad.data if p.grad is not None else torch.zeros_like(p.data) \
                                   for name, p in model.named_parameters() if 'alpha' in name]
                    eps = 0.01 / np.sqrt(sum([(g ** 2).sum().item() for g in g_val_w]))
                    old_w = [p.data.clone() for name, p in model.named_parameters() if 'alpha' not in name] 

                    w_plus = [w + eps * g for w, g in zip(old_w, g_val_w)]
                    for p, w_p in zip([p for n, p in model.named_parameters() if 'alpha' not in n], w_plus):
                        p.data = w_p.data
                    p_logits = model(batch['inp_ids'], batch['type_ids'],
                                     batch['att'].max(0).values, domain_idx if total_steps >= args.curr_steps else 0)
                    loss = 0.2 * criterion(p_logits, batch['labels']) + 0.8 * distil_loss(batch['logits'], p_logits)
                    loss.backward()
                    g_alpha_plus = [p.grad.data for n, p in model.named_parameters() if 'alpha' in n]
                    del w_plus

                    w_minus = [w - eps * g for w, g in zip(old_w, g_val_w)]
                    for p, w_m in zip([p for n, p in model.named_parameters() if 'alpha' not in n], w_minus):
                        p.data = w_m.data
                    p_logits = model(batch['inp_ids'], batch['type_ids'],
                                     batch['att'].max(0).values, domain_idx if total_steps >= args.curr_steps else 0)
                    loss = 0.2 * criterion(p_logits, batch['labels']) + 0.8 * distil_loss(batch['logits'], p_logits)
                    loss.backward()
                    g_alpha_minus = [p.grad.data for n, p in model.named_parameters() if 'alpha' in n]

                    curr_lr = optimizer.param_groups[0]['lr']
                    g_so = [g_fo - curr_lr * (g_p - g_m) / (2 * eps) for g_fo, g_p, g_m in \
                            zip(g_val_alpha, g_alpha_plus, g_alpha_minus)]
                    for p, g in zip([p for n, p in model.named_parameters() if 'alpha' in n], g_so):
                        p.grad.data = g.data

                    optimizer_struct.step()
                    break

                total_steps += 1
                if total_steps <= args.curr_steps:
                    for cell in model.cells:
                        for node in cell.nodes:
                            for edge in node.edges.values():
                                edge.alpha.data[1:] = edge.alpha.data[0][None]
                            node.input_switch.alpha.data[1:] = node.input_switch.alpha.data[0][None]

                temp = max(1e-3, 0.01 ** (total_steps / len(train_dl[0]) / len(ds_configs)))
                model.set_temperature(temp)
                if i % log_freq == 0 and i > 0:
                    print('Train acc', round((pi_logits.argmax(-1) == p_logits.argmax(-1)).float().mean().item(), 4))
                    print(model.export())
            
                if total_steps % valid_freq == 0:
                    val_acc_arr = evaluate(model, val_dl, device, total_steps < args.curr_steps)
                    model.train()
                    val_accs.append(np.mean(val_acc_arr))
                    if np.mean(val_acc_arr) >= max(val_accs):
                        best_arch = model.export()
                    for val_acc, cfg in zip(val_acc_arr, ds_configs):
                        print(f'Cfg: {cfg}, step: {total_steps}, val acc: {round(val_acc, 4)}, best avg: {round(max(val_accs), 4)}')

    print('Finished with', round(max(val_accs), 4))
    print('Final architecture', best_arch)
    with open(f'final_arch_{seed}.json', 'w') as f:
        f.write(json.dumps(best_arch))


if __name__ == '__main__':
    main()
