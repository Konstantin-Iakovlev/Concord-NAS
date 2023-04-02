import sys
sys.path.append('.')
from data import BatchParallelLoader, ParallelSentenceCorpus


def test_corpus():
    c = ParallelSentenceCorpus('data/iwslt14/en_de_parallel')
    assert True

def test_loader():
    c = ParallelSentenceCorpus('data/iwslt14/en_de_parallel')
    b = BatchParallelLoader(c.test_parallel, 6)
    en_b, de_b = next(iter(b))
    assert en_b.shape[0] == de_b.shape[0]
    assert en_b.shape[0] == 6
