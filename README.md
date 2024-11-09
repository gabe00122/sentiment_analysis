# Sentiment Language Model

A small language model (109M parameters) that can both generate Yelp-style reviews and predict their star ratings. Built with JAX and Flax.

## Demo
![Demo of the model generating a restaurant review and predicting its rating](/assets/demo.gif)

## Overview

This model combines two tasks:
1. Next-token prediction for generating coherent restaurant reviews
2. Star rating prediction (1-5 stars) based on the review text

Key technical features:
* Rotary Position Embeddings (RoPE)
* Weight-tied input/output embeddings
* Gated Linear Units (GLU)
* Post-layer normalization
* Attention soft caps

## Quick Start

### Prerequisites
- Python 3.12
- Poetry for dependency management

### Installation

```bash
cd sentiment_analysis
poetry install
```

### Inference

```bash
poetry shell

sentiment_lm inference \
    --model ./results/small_generative_2024-08-04_23-50-54 \
    --temperature 0.7 \
    --top-p 0.9 \
    --top-k 50
```

### Training
All my training was done on a single RTX 3070 GPU and the vram requirements were tuned to that. Training 3 epocs took about 12 hours.

### Data prepreation

Download the json data from https://www.yelp.com/dataset


Create a folder in the sentiment_lm project called `./data`
Extract the yelp data to `data/yelp_academic_dataset_review.json`


#### Create a training/validation/test split
```bash
sentiment_lm preprocess train-test-split \
    --test-ratio 0.2 \
    --validation-ratio 0.05 \
    --seed 1
```


#### Create the vocab
```bash
sentiment_lm preprocess create-tokenizer-corpus \
    --corpus-ratio 0.4 \
    --seed 2
sentiment_lm preprocess train-tokenizer --vocab-size 32000
```


#### Tokenize the data
```bash
sentiment_lm preprocess pretokenize \
    --vocab-file ./vocab/yelp-32000.model \
    --context-size 128
```

### Train
```bash
sentiment_lm train ./config/small_generative.json
```

Example config file:
```json
{
    "seed": "random",
    "training_file": "data/training.npz",
    "validation_file": "data/validation.npz",
    "epochs": 3,
    "batch_size": 1536,
    "accumulation_steps": 48,
    "context_size": 128,
    "optimizer": {
        "type": "adamw",
        "learning_rate": 0.0012,
        "warmup_steps": 200,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-12,
        "weight_decay": 0.1
    },
    "vocab": {
        "path": "vocab/yelp-16000.model",
        "size": 16000
    },
    "model": {
        "num_layers": 12,
        "num_heads": 12,
        "d_model": 768,
        "ffn_size": 2048,
        "glu": true,
        "activation_name": "silu",
        "dtype": "bfloat16",
        "param_dtype": "float32"
    },
    "logger": {
        "use_tb": true,
        "use_csv": false,
        "use_neptune": false,
        "use_wandb": false
    }
}
```

### Data

https://www.yelp.com/dataset
