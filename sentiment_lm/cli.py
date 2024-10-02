from typing import Annotated, Optional
import typer

from pathlib import Path
from rich.traceback import install


app = typer.Typer()

@app.command()
def inference(
    model: Annotated[Path, typer.Option()],
    temp: Annotated[float, typer.Option()] = 0.7,
    top_k: Annotated[int, typer.Option()] = 50,
    top_p: Annotated[float, typer.Option()] = 0.9,
):
    from sentiment_lm.inference import inference_cli

    inference_cli(model, temp, top_k, top_p)


@app.command()
def train(settings: str):
    pass


# def inference_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
#     parser = subparsers.add_parser("inference")
#     parser.set_defaults(func=run_inference)

#     parser.add_argument("-m", "--model", required=True)
#     #parser.add_argument()

#     return parser


# def run_inference(args):
#     from sentiment_lm.inference import inference_cli
#     inference_cli(Path(args.model))


# def train_parser(subparsers):
#     parser = subparsers.add_parser("train")
#     parser.set_defaults(func=run_train)

#     parser.add_argument("-s", "--settings", required=True)

#     return parser


# def run_train(args):
#     from sentiment_lm.experiment import Experiment
#     from sentiment_lm.train import train

#     experiment = Experiment.create_experiment(Path(args.settings))
#     train(experiment)


def main():
    install()
    app()


if __name__ == '__main__':
    main()
