from sentiment_analysis.transformer import Transformer
from sentiment_analysis.network import Network
from sentiment_analysis.positional_embeddings import get_positional_embeddings
from sentiment_analysis.eval import get_abstract_tree, load_model
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from tensorboardX import SummaryWriter


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

    embeddings = params['params']['Embed_0']['embedding']
    print(embeddings)
    #embeddings = params[]

    sp = SentencePieceProcessor(model_file="tokenizer.model")
    vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

    writer = SummaryWriter("./tensorboard")
    writer.add_embedding(embeddings, metadata=vocabs)

    writer.flush()


if __name__ == '__main__':
    main()
