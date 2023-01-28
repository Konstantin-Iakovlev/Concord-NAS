import jax
import jax.numpy as jnp
import numpy as np
import os, shutil


def mask2d(B, D, keep_prob, key):
    m = jax.random.bernoulli(key, p=keep_prob, shape=(B, D)) / keep_prob
    return m

def locked_dropout(x: jnp.array, training: bool, key, dropout: float=0.5):
    if not training:
        return x
    mask = jax.random.bernoulli(key, p=1 - dropout, shape=x.shape[1:])[None] / (1 - dropout)
    return x * mask


def batchify(data: np.array, bsz: int) -> jnp.array:
    nbatch = data.shape[0] // bsz
    data = data[:nbatch * bsz]
    data = data.reshape(bsz, -1).T
    print(data.shape)
    return jnp.asarray(data)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def get_batch(source, i, args, seq_len=None):
    i = i % (len(source) - 1)
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1: i + 1 + seq_len]
    return data, target
