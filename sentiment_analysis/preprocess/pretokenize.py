import tokenmonster
import numpy as np
from pathlib import Path
import json


def pretokenize(path: str | Path, vocab: tokenmonster.Vocab, max_length: int):
    input_path = Path(path).absolute()
    output_path = input_path.with_suffix(".npz")

    output_tokens = []
    output_labels = []

    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data['text']
            label = data['label']

            tokens = vocab.tokenize(text)
            if len(tokens) <= max_length:
                tokens = list(tokens) + ([-1] * (max_length - len(tokens)))
                output_tokens.append(tokens)
                output_labels.append(label - 1)

            if i % 10_000 == 9_999:
                total_reviews = len(output_tokens)
                percentage = (total_reviews / i) * 100
                print(f"{output_path.name} - {total_reviews} - {percentage:.0f}%")

    print(f"Saving {input_path.name}")
    np_tokens = np.array(output_tokens, np.int16)
    np_labels = np.array(output_labels, np.int8)
    np.savez_compressed(output_path, tokens=np_tokens, labels=np_labels)


def main():
    vocab = tokenmonster.load("./vocab/yelp-32000-consistent-oneword-v1.vocab")
    paths = [
        "./data/test.json",
        "./data/training.json",
        "./data/validation.json"
    ]

    for p in paths:
        pretokenize(p, vocab, 115)


if __name__ == '__main__':
    main()
