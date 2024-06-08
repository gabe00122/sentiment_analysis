from jax import numpy as jnp
import numpy as np


def get_positional_embeddings(seq_length, features, n=10000):
    output = np.zeros((seq_length, features))
    for k in range(seq_length):
        for i in jnp.arange(int(features/2)):
            denominator = np.power(n, 2*i/features)
            output[k, 2*i] = np.sin(k/denominator)
            output[k, 2*i+1] = np.cos(k/denominator)
    return jnp.asarray(output)
