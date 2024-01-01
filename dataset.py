from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np


class NliDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, ds_name, ds_config=None,
                 split='train', max_length=128) -> None:
        super().__init__()
        self.task_to_keys = {
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
        self.glue_datasets = set(self.task_to_keys.keys())
        self.task_to_keys["paws-x"] = ("sentence1", "sentence2")
        self.task_to_keys["xnli"] = ("premise", "hypothesis")
        if ds_name in self.glue_datasets:
            self.raw_ds = load_dataset("glue", ds_name, data_dir='.')[split]
            self.distil_logits = np.load(f'analysis/{ds_name}_logits_{split}.npy')
        else:
            self.raw_ds = load_dataset(ds_name, ds_config, data_dir='.')[split]
            self.distil_logits = np.load(f'analysis/{ds_name}_{ds_config}_logits_{split}.npy')
        assert len(self.raw_ds) == self.distil_logits.shape[0]
        self.max_length = max_length
        self.tok = tokenizer
        self.ds_name = ds_name
    
    def __getitem__(self, index):
        dp = self.raw_ds[index]
        dp['logits'] = self.distil_logits[index]
        return dp

    def __len__(self):
        return len(self.raw_ds)
    
    def collate_fn(self, data_points):
        k1, k2 = self.task_to_keys[self.ds_name]
        if k2 is not None:
            tok_out = self.tok([d[k1] for d in data_points], [d[k2] for d in data_points], return_tensors='pt',
                        padding=True, max_length=self.max_length, truncation=True)
            inp_ids = torch.stack([tok_out['input_ids'], tok_out['input_ids']], dim=0)
            type_ids = torch.stack([tok_out['token_type_ids'], tok_out['token_type_ids']], dim=0)
        else:
            tok_out = self.tok([d[k1] for d in data_points], return_tensors='pt',padding=True,
                        max_length=self.max_length, truncation=True)
            inp_ids = tok_out['input_ids'][None]
            type_ids = tok_out['token_type_ids'][None]
        logits = torch.tensor(np.stack([b['logits'] for b in data_points], axis=0))
        return {'labels': torch.LongTensor([d['label'] for d in data_points]), 'inp_ids': inp_ids,
                'att': (inp_ids != self.tok.pad_token_id).long(), 'logits': logits, 'type_ids': type_ids}
