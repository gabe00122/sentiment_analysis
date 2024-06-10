import json
from pathlib import Path
from sentencepiece import SentencePieceTrainer


def generate_corpus(output_path: Path, size: int):
    data_file = open(
        "./data/yelp_academic_dataset_review.json",
        "r",
    )

    with open(output_path, "w") as out:
        i = 0
        for line in data_file:
            data = json.loads(line)
            text = data["text"].lower()
            out.write(text + "\n")

            i += 1
            if i >= size:
                break


def main():
    corpus_path = Path("./data/tokenizer_corpus.txt")
    generate_corpus(Path("./data/tokenizer_corpus.txt"), 1_000_000)
    SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix="tokenizer",
        vocab_size=2000,
    )


if __name__ == "__main__":
    main()
