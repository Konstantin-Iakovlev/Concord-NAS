from model import MdRnnCell
from genotypes import DARTS_V2
import torch

def test_darts_single():
    ninp = nhid = 50
    rnn = MdRnnCell(ninp, nhid, 0, 0, None, 1)
    bs = 16
    seq_len = 5
    inp = torch.randn(seq_len, bs, ninp)
    hidden = torch.randn(1, bs, ninp)
    _, last_h = rnn(inp, hidden, 0)
    assert True

def test_one_hot_single():
    ninp = nhid = 50
    rnn = MdRnnCell(ninp, nhid, 0, 0, [DARTS_V2.recurrent], 1)
    bs = 16
    seq_len = 5
    inp = torch.randn(seq_len, bs, ninp)
    hidden = torch.randn(1, bs, ninp)
    _, last_h = rnn(inp, hidden, 0)
    assert True

def test_darts_single_export():
    ninp = nhid = 50
    rnn = MdRnnCell(ninp, nhid, 0, 0, None, 1)
    arch = rnn.export()
    assert True
