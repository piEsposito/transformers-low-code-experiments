# Transformers for Data Scientists in a rush
Low-code pre-built pipelines for experiments with huggingface/transformers for Data Scientists in a rush.

---
This repository contains low-code, easy to understand, pre-built pipelines for fast experimentation on NLP tasks using [huggingface/transformers](https://github.com/huggingface/transformers) pre-trained language models, which are explained and explored in a post series on Medium about the theme. 

This was inspired by a LinkedIn post of Thomas Wolf, HuggingFace's CSO in which there was an image of a low-code pipeline for fast experimentation on their Transformers repo. As I could not see anything like it implemented on the internet, I've decided to do it myself. 

# Index 
As of now, we have:
 * [classification](#Classification), with a classification example. 
 
 ## Classification
 On the classification example, we use a [dataset for email spam classification](https://www.kaggle.com/team-ai/spam-text-message-classification)  from Kaggle, and use [optuna](https://optuna.org/) for hyperparameter tuning.
 
 You might run it, on the classification directory, with:
 
```bash
python classification-experiment.py --model-name bert-base-multilingual-cased ---metric f1_score --train-data-path train.csv --test-data-path test.csv --max-sequence-length 25 --label-nbr 2
```

It should yield a f1_score higher than 0.9.

---
###### Made by Pi Esposito