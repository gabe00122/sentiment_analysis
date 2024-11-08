from typing import Annotated
import typer

from pathlib import Path

from sentiment_lm import preprocess
from sentiment_lm.train import train as train_fn
from sentiment_lm.experiment import Experiment

app = typer.Typer()
app.add_typer(preprocess.app, name="preprocess")


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
    settings_path = Path(settings)
    experiment = Experiment.create_experiment(settings_path)
    train_fn(experiment)


def main():
    # install()
    app()


if __name__ == "__main__":
    main()
