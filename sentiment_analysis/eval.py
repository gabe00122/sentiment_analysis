from sentiment_analysis.model.network import Transformer, Network
from sentiment_analysis.positional_embeddings import get_positional_embeddings
import orbax.checkpoint as ocp
from sentencepiece import SentencePieceProcessor


from pathlib import Path
import jax
from jax import numpy as jnp, random
from flax import linen as nn


def main():
    vocab_size = 2000
    embedding_features = 128
    sequence_length = 128
    num_heads = 8
    context_size = 128

    transformer = Transformer(
        num_heads=num_heads,
        token_features=embedding_features,
        num_layers=6,
    )

    network = Network(
        transformer=transformer,
        vocab_size=vocab_size,
        embedding_features=embedding_features,
        position_embeddings=get_positional_embeddings(
            sequence_length * 2, embedding_features
        ),
    )

    abstract_tree = get_abstract_tree(network, context_size)
    params = load_model(Path("./metrics_friday/1"), abstract_tree)
    # print(params)

    processor = SentencePieceProcessor(model_file="tokenizer.model")

    def inference(tokens):
        mask = tokens != -1
        logits = network.apply(params, tokens, mask)
        return nn.softmax(logits)

    inference = jax.jit(inference)

    while True:
        text = input("input: ")
        tokens = processor.Encode(text.lower())
        tokens += [-1] * (context_size - len(tokens))
        tokens = jnp.array(tokens, jnp.int16)

        softmax = inference(tokens)
        prediction = jnp.argmax(softmax) + 1
        print(softmax)
        print("⭐" * prediction)


def get_abstract_tree(network, context_size):
    dummy_tokens = jnp.zeros(context_size, dtype=jnp.int16)
    dummy_mask = jnp.zeros(context_size, dtype=jnp.bool)
    dummy_params = network.init(random.PRNGKey(0), dummy_tokens, dummy_mask)
    return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, dummy_params)


def load_model(path: Path, abstract_tree):
    checkpointer = ocp.StandardCheckpointer()
    # 'checkpoint_name' must not already exist.
    return checkpointer.restore(
        path.absolute(), args=ocp.args.StandardRestore(abstract_tree)
    )


if __name__ == "__main__":
    main()
