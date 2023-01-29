from model import MdRnnModel
import jax
import jax.numpy as jnp
from genotypes import DARTS_V1

def build_model(genotype=None):
    nhid = ninp = 20
    key = jax.random.PRNGKey(0)

    md_model = MdRnnModel(100, ninp, ninp, ninp, True, genotype=genotype)
    seq_len = 30
    bs = 4

    inp = jnp.ones((seq_len, bs), dtype=jnp.int32)
    hidden = jax.random.normal(key, (bs, ninp))

    params = md_model.init({'locked_dropout_emb': key, 'locked_dropout_out': key, 'dropout': key, 'params': key,
                            'mask_2d': key}, inp, hidden)

    md_model.apply(params, inp, hidden,
        rngs={'locked_dropout_emb': key, 'locked_dropout_out': key, 'dropout': key, 'params': key,
        'mask_2d': key}, mutable='batch_stats')

    assert True

def test_rnn_darts():
    build_model()

def test_rnn_one_hot():
    build_model(DARTS_V1)
