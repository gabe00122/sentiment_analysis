# Sentiment Language Model

A 109 million parameter generative language trained on Yelp data for next token prediction and review rating prediction.

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

### Data prepreation
1)
Download the json data from https://www.yelp.com/dataset

2)
Create a folder in the sentiment_lm project called `./data`
Extract the yelp data to `data/yelp_academic_dataset_review.json`

3)
Create a training/validation/test split
```bash
sentiment_lm preprocess train-test-split
```

4)
Create the vocab
```bash
sentiment_lm preprocess create-tokenizer-corpus
sentiment_lm preprocess train-tokenizer
```

5)
Tokenize the data
```bash
sentiment_lm preprocess pretokenize
```

## Train


## Data

https://www.yelp.com/dataset
