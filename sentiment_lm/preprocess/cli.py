import typer
from .yelp_test_split import train_test_split
from .create_tokenizer_corpus import create_tokenizer_corpus
from .train_tokenizer import train_tokenizer
from .pretokenize import pretokenize


app = typer.Typer()

app.command()(train_test_split)
app.command()(create_tokenizer_corpus)
app.command()(train_tokenizer)
app.command()(pretokenize)
app.command()(train_tokenizer)
