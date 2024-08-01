import argparse

from sentiment_analysis.experiment import Experiment
from sentiment_analysis.train import train


def main():
    parser = argparse.ArgumentParser(prog="Train", description="")

    parser.add_argument("-s", "--settings", required=True)
    args = parser.parse_args()

    settings = Experiment.create_experiment(args.settings)
    train(settings)


if __name__ == "__main__":
    main()
