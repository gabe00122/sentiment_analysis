import tokenmonster
import numpy as np
from pathlib import Path
import json

from sentiment_lm.constants import SPECIAL_TOKENS, EMPTY_TOKEN, START_TOKEN, STAR_TOKENS
from sentiment_lm.tokenizer import Tokenizer


def pretokenize(path: str, vocab_file: str, context_size: int = 128):
    input_path = Path(path).absolute()
    output_path = input_path.with_suffix(".npz")

    output_tokens = []
    output_length = []

    tokenizer = Tokenizer(vocab_file, context_size)

    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data["text"]
            label = data["label"]

            tokens, length = tokenizer.encode(text)
            if length <= context_size:
                output_tokens.append(tokens)
                output_length.append(length)

            if i % 10_000 == 9_999:
                total_reviews = len(output_tokens)
                percentage = (total_reviews / i) * 100
                print(f"{output_path.name} - {total_reviews} - {percentage:.0f}%")

    print(f"Saving {input_path.name}")
    np_tokens = np.array(output_tokens, np.uint16)
    np_length = np.array(output_length, np.uint8)
    np.savez_compressed(output_path, tokens=np_tokens, length=np_length)


def main():
    paths = ["./data/test.json", "./data/training.json", "./data/validation.json"]

    for p in paths:
        pretokenize(p, "./vocab/yelp-16000.model", CONTEXT_SIZE)


if __name__ == "__main__":
    main()
