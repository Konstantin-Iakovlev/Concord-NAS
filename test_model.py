from model import MdRnnModel
import jax
import jax.numpy as jnp

def test_rnn_model():
    nhid = ninp = 20
    key = jax.random.PRNGKey(0)

    md_model = MdRnnModel(100, ninp, ninp, ninp, True)
    seq_len = 30
    bs = 4

    inp = jnp.ones((seq_len, bs), dtype=jnp.int32)
    hidden = jax.random.normal(key, (bs, ninp))

    params = md_model.init({'locked_dropout': key, 'dropout': key, 'params': key,
                            'mask_2d': key}, inp, hidden)

    md_model.apply(params, inp, hidden,
        rngs={'locked_dropout': key, 'dropout': key, 'params': key,
        'mask_2d': key}, mutable='batch_stats')

    assert True