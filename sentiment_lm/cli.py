from argparse import ArgumentParser


def inference_parser(subparsers):
    parser = subparsers.add_parser("inference")
    parser.set_defaults(func=run_inference)

    parser.add_argument("-m", "--model", required=True)

    return parser


def run_inference(args):
    from sentiment_lm.inference import inference_cli
    inference_cli(args.model)


def train_parser(subparsers):
    parser = subparsers.add_parser("train")
    parser.set_defaults(func=run_train)

    parser.add_argument("-s", "--settings", required=True)

    return parser


def run_train(args):
    from sentiment_lm.experiment import Experiment
    from sentiment_lm.train import train

    experiment = Experiment.create_experiment(args.settings)
    train(experiment)


def main():
    parser = ArgumentParser("sentiment_analysis")

    subparsers = parser.add_subparsers(required=True)
    train_parser(subparsers)
    inference_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
