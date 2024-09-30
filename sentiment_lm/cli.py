from argparse import ArgumentParser
from pathlib import Path
from rich.traceback import install


def inference_parser(subparsers):
    parser = subparsers.add_parser("inference")
    parser.set_defaults(func=run_inference)

    parser.add_argument("-m", "--model", required=True)

    return parser


def run_inference(args):
    from sentiment_lm.inference import inference_cli
    inference_cli(Path(args.model))


def train_parser(subparsers):
    parser = subparsers.add_parser("train")
    parser.set_defaults(func=run_train)

    parser.add_argument("-s", "--settings", required=True)

    return parser


def run_train(args):
    from sentiment_lm.experiment import Experiment
    from sentiment_lm.train import train

    experiment = Experiment.create_experiment(Path(args.settings))
    train(experiment)


def main():
    install(show_locals=True)

    parser = ArgumentParser("sentiment_analysis")

    subparsers = parser.add_subparsers(required=True)
    train_parser(subparsers)
    inference_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
