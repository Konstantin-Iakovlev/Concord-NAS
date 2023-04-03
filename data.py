import os
import torch
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from typing import List, Tuple

PAD_TOKEN = '<pad>'


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class SentCorpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), path
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                words = line.split() + ['<eos>']
                sent = torch.LongTensor(len(words))
                for i, word in enumerate(words):
                    sent[i] = self.dictionary.word2idx[word]
                sents.append(sent)

        return sents


class ParallelSentenceCorpus:
    def __init__(self, path: str) -> None:
        self.dictionary = Dictionary()
        self.train_parallel = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid_parallel = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test_parallel = self.tokenize(os.path.join(path, 'test.txt'))
    
    def tokenize(self, path: str) -> Tuple[List[torch.LongTensor], List[torch.LongTensor]]:
        """Tokenizes a parallel file with \t separator"""
        assert os.path.exists(path)
        self.dictionary.add_word(PAD_TOKEN)
        # update dictionary
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # we do not add <eos>, because we already have it in the file
                words = line.strip().split()
                for word in words:
                    self.dictionary.add_word(word)
        # tokenize the content
        en_sents = []
        de_sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                en_text, de_text = line.strip().split('\t')
                en_sents.append(torch.LongTensor(self.tokenize_sent(en_text.split())))
                de_sents.append(torch.LongTensor(self.tokenize_sent(de_text.split())))
        return en_sents, de_sents     
    
    def tokenize_sent(self, words: List[str]) -> List[int]:
        return [self.dictionary.word2idx[w] for w in words]


class BatchSentLoader(object):
    def __init__(self, sents, batch_size, pad_id=0, cuda=False, volatile=False):
        self.sents = sents
        self.batch_size = batch_size
        self.sort_sents = sorted(sents, key=lambda x: x.size(0))
        self.cuda = cuda
        self.volatile = volatile
        self.pad_id = pad_id

    def __next__(self):
        if self.idx >= len(self.sort_sents):
            raise StopIteration

        batch_size = min(self.batch_size, len(self.sort_sents)-self.idx)
        batch = self.sort_sents[self.idx:self.idx+batch_size]
        max_len = max([s.size(0) for s in batch])
        tensor = torch.LongTensor(max_len, batch_size).fill_(self.pad_id)
        for i in range(len(batch)):
            s = batch[i]
            tensor[:s.size(0),i].copy_(s)
        if self.cuda:
            tensor = tensor

        self.idx += batch_size

        return tensor
    
    next = __next__

    def __iter__(self):
        self.idx = 0
        return self


class BatchParallelLoader:
    def __init__(self, sents: Tuple[List[torch.LongTensor], List[torch.LongTensor]],
                 batch_size, pad_id=0, device='cpu', max_len=500) -> None:
        self.sents = sorted(zip(*sents), key=lambda s: s[0].shape[0] + s[1].shape[0],
                            reverse=True)
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.device = device
        self.max_len = max_len

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.sents):
            raise StopIteration
        batch_size = min(self.batch_size, len(self.sents) - self.idx)
        batch_par = self.sents[self.idx: self.idx + batch_size]
        en_padded = pad_sequence([b[0][:self.max_len] for b in batch_par], batch_first=True, padding_value=self.pad_id)
        de_padded = pad_sequence([b[1][:self.max_len] for b in batch_par], batch_first=True, padding_value=self.pad_id)
        self.idx += batch_size
        return en_padded.to(self.device), de_padded.to(self.device)
    
    def __len__(self):
        return len(self.sents)
        