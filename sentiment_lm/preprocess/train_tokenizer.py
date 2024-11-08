import sentencepiece as spm


def train_tokenizer(vocab_size: int = 16000):
    spm.SentencePieceTrainer.train(
        input="./data/corpus.txt",
        model_prefix=f"./vocab/yelp-{vocab_size}",
        vocab_size=vocab_size,
    )
