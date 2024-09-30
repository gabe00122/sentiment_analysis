import random
from pydantic import BaseModel
from pathlib import Path
import json


class TokenizerCorpusConfig(BaseModel):
    corpus_ratio: float
    seed: int


def create_tokenizer_corpus(data_path: Path, config: TokenizerCorpusConfig):
    random.seed(config.seed)
    sentence_count = 0

    def print_stats():
        print(f"sentence_count = {sentence_count}")

    with open(data_path, "r") as data_file, open("./data/corpus.txt", "w") as corpus:
        for i, line in enumerate(data_file):
            data = json.loads(line)
            text = data["text"]
            sample = random.uniform(0, 1)

            if sample <= config.corpus_ratio:
                corpus.write(text)
                sentence_count += 1

            if i % 10_000 == 9_999:
                print_stats()

    print("done!")


def main():
    config_text = Path("./experiment_settings/corpus.json").read_text()
    config = TokenizerCorpusConfig.model_validate_json(config_text)
    create_tokenizer_corpus(Path("./data/training.json"), config)


if __name__ == "__main__":
    main()
