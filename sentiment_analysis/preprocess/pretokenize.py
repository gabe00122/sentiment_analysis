import tokenmonster
import numpy as np
from pathlib import Path
import json

from sentiment_analysis.constants import CONTEXT_SIZE, SPECIAL_TOKENS, EMPTY_TOKEN, END_TOKEN


def pretokenize(path: str | Path, vocab: tokenmonster.Vocab, max_length: int):
    input_path = Path(path).absolute()
    output_path = input_path.with_suffix(".npz")

    output_tokens = []
    output_length = []

    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data["text"]
            label = data["label"]

            tokens = list(vocab.tokenize(text))
            if len(tokens) <= max_length - 2:
                tokens = [token + SPECIAL_TOKENS for token in tokens] + [END_TOKEN, label]
                token_length = len(tokens)

                tokens = tokens + ([EMPTY_TOKEN] * (max_length - token_length))
                output_tokens.append(tokens)
                output_length.append(token_length)

            if i % 10_000 == 9_999:
                total_reviews = len(output_tokens)
                percentage = (total_reviews / i) * 100
                print(f"{output_path.name} - {total_reviews} - {percentage:.0f}%")

    print(f"Saving {input_path.name}")
    np_tokens = np.array(output_tokens, np.int16)
    np_length = np.array(output_length, np.uint8)
    np.savez_compressed(output_path, tokens=np_tokens, length=np_length)


def main():
    vocab = tokenmonster.load("./vocab/yelp-32000-consistent-oneword-v1.vocab")
    paths = ["./data/test.json", "./data/training.json", "./data/validation.json"]

    for p in paths:
        pretokenize(p, vocab, CONTEXT_SIZE)


if __name__ == "__main__":
    main()
