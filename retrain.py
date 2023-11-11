import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from argparse import ArgumentParser
from dataset import NliDataset
from transformers import AutoTokenizer, AutoModel
from model import AdaBertStudent, evaluate, distil_loss
import numpy as np
from tqdm.auto import tqdm
import json
from transformers import get_cosine_schedule_with_warmup


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
    num_cells = 2
    lr = 1e-3
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


    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    train_ds = NliDataset(tokenizer, args.ds_name, ds_config='en', split='train', max_length=max_length)
    val_ds = NliDataset(tokenizer, args.ds_name, ds_config='en', split='validation', max_length=max_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=val_ds.collate_fn, shuffle=False)

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
        m = AutoModel.from_pretrained('bert-base-cased', cache_dir='.')
        # raise ValueError(f'Unknown dataset {args.ds_name}')
    pretrained_token_embeddigns = m.embeddings.word_embeddings.weight
    pretrained_pos_embeddigns = m.embeddings.position_embeddings.weight
    model = AdaBertStudent(tokenizer.vocab_size, train_ds.task_to_keys[args.ds_name][-1] is not None,
                           2, pretrained_token_embeddigns,
                           pretrained_pos_embeddigns, num_cells=num_cells,
                           genotype=genotype, dropout_p=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, epochs * len(train_dl))
    criterion = nn.CrossEntropyLoss()

    
    total_steps = 0
    val_accs = []
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_dl, desc=f'epoch {epoch}/{epochs}')):
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
