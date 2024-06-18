from pathlib import Path

import jax
from flax import linen as nn
from jax import numpy as jnp, random
import numpy as np

from sentiment_analysis.eval import get_abstract_tree, load_model
from sentiment_analysis.network import Network
from sentiment_analysis.positional_embeddings import get_positional_embeddings
from sentiment_analysis.transformer import Transformer


def main():
    total_steps = 25_000
    epochs = 10

    rng_key = random.PRNGKey(42)
    vocab_size = 2000
    embedding_features = 128

    batch_size = 128

    sequence_length = 128
    num_heads = 8
    num_layers = 6

    data_size = total_steps * sequence_length

    training_data = jnp.load("./data/training_data.npz")
    print(training_data)
    print(training_data["tokens"].shape)

    test_set_size = 10000
    labels = training_data["stars"][data_size:]  # should be stars or labels
    tokens = training_data["tokens"][data_size:]

    print(tokens.shape)

    transformer = Transformer(
        num_heads=num_heads,
        token_features=embedding_features,
        num_layers=num_layers,
    )

    network = Network(
        transformer=transformer,
        vocab_size=vocab_size,
        embedding_features=embedding_features,
        position_embeddings=get_positional_embeddings(
            sequence_length * 2, embedding_features
        ),
    )

    abstract_tree = get_abstract_tree(network, sequence_length)
    params = load_model(Path("./metrics_friday/1"), abstract_tree)
    # param_count = sum(x.size for x in jax.tree_leaves(params))

    # print(param_count)

    # unique, counts = jnp.unique(labels, return_counts=True)
    # print(unique)
    # print(counts)

    def score(labels, tokens):
        mask = tokens != -1
        logits = network.apply(params, tokens, mask)
        predictions = jnp.argmax(nn.softmax(logits), axis=1)
        return jnp.mean(predictions == labels - 1)

    score = jax.jit(score)

    batch_size = 1000
    accuracy = np.zeros(870, jnp.float32)
    for i in range(870):
        accuracy[i] = score(
            labels[i * batch_size : i * batch_size + batch_size],
            tokens[i * batch_size : i * batch_size + batch_size],
        )
    print(jnp.mean(accuracy))


if __name__ == "__main__":
    main()
