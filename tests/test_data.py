import sys
sys.path.append('.')
from data import BatchParallelLoader, ParallelSentenceCorpus


def test_corpus():
    c = ParallelSentenceCorpus('data/iwslt14/en_de_parallel')
    assert True

def test_loader():
    c = ParallelSentenceCorpus('data/iwslt14/en_de_parallel')
    b = BatchParallelLoader(c.test_parallel, 256 * 35)
    for en_b, de_b in b:
        assert en_b.shape[0] == de_b.shape[0]
        pad_mask_en = en_b != b.pad_id
        pad_mask_de = de_b != b.pad_id
        assert (pad_mask_en.sum() + pad_mask_de.sum()).item() <= 256 * 35
