import random
from pathlib import Path
import json

data_path = "./data/training.json"
corpus_path = "./data/corpus.txt"


def create_tokenizer_corpus(
    corpus_ratio: float = 0.1,
    seed: int = 42,
):
    random.seed(seed)
    sentence_count = 0

    def print_stats():
        print(f"sentence_count = {sentence_count}")

    with open(data_path, "r") as data_file, open(corpus_path, "w") as corpus:
        for i, line in enumerate(data_file):
            data = json.loads(line)
            text = data["text"]
            sample = random.uniform(0, 1)

            if sample <= corpus_ratio:
                corpus.write(text)
                sentence_count += 1

            if i % 10_000 == 9_999:
                print_stats()

    print("done!")


def main():
    create_tokenizer_corpus()


if __name__ == "__main__":
    main()
