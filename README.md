# Sentiment Language Model

A 109 million parameter generative language trained on yelp data for next token prediction and review rating prediction.

## Demo
![Test](/assets/demo.gif)

## Features

* RoPE
* Tied embeddings
* GLU layers
* Post layer norm
* Attention soft caps

## Installation

- Install python 3.12
- Install poetry

```bash
cd sentiment_analysis
poetry install
```

## Inference

```bash
poetry run sentiment_lm inference --model ./results/small_generative_2024-08-04_23-50-54 --tem
p 1.0 --top-p 0.95 --top-k 16000
```

## Training

TODO

## Data

https://www.yelp.com/dataset
