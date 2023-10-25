from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np


class RteDataset(Dataset):
    def __init__(self, split='train') -> None:
        super().__init__()
        self.raw_ds = load_dataset("glue", 'rte', cache_dir='.')[split]
        self.distil_logits = np.load(f'analysis/{split}_logits.npy')
        assert len(self.raw_ds) == self.distil_logits.shape[0]
    
    def __getitem__(self, index):
        dp = self.raw_ds[index]
        dp['logits'] = self.distil_logits[index]
        return dp

    def __len__(self):
        return len(self.raw_ds)
