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
poetry shell

sentiment_lm inference \
    --model ./results/small_generative_2024-08-04_23-50-54 \
    --temperature 0.7 \
    --top-p 0.9 \
    --top-k 50
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
sentiment_lm preprocess train-test-split \
    --test-ratio 0.2 \
    --validation-ratio 0.05 \
    --seed 1
```

4)
Create the vocab
```bash
sentiment_lm preprocess create-tokenizer-corpus \
    --corpus-ratio 0.4 \
    --seed 2
sentiment_lm preprocess train-tokenizer --vocab-size 32000
```

5)
Tokenize the data
```bash
sentiment_lm preprocess pretokenize \
    --vocab-file ./vocab/yelp-32000.model \
    --context-size 128
```

## Train
```bash
sentiment_lm train ./config/small_generative.json
```

## Data

https://www.yelp.com/dataset
