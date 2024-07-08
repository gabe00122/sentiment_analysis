import random

import argparse
from dataclasses import replace

from pydantic import TypeAdapter

from sentiment_analysis.types import ExperimentSettings
from sentiment_analysis.train import train


def load_settings(file: str) -> ExperimentSettings:
    with open(file, 'r') as f:
        text = f.read()

    settings = TypeAdapter(ExperimentSettings).validate_json(text)

    if settings.seed == 'random':
        settings = replace(settings, seed=random.getrandbits(32))

    return settings


def main():
    parser = argparse.ArgumentParser(
        prog="Train",
        description=""
    )

    parser.add_argument("-s", "--settings", required=True)
    args = parser.parse_args()

    settings = load_settings(args.settings)
    train(settings)


if __name__ == '__main__':
    main()
