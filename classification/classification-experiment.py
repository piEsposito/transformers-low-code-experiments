from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
import nlp

import optuna

from sklearn.metrics import f1_score
import numpy as np

from tqdm import tqdm

import argparse


class MultilabeledSequenceModel(nn.Module):
    def __init__(self,
                 pretrained_model_name,
                 label_nbr):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(list(self.transformer.modules())[-2].out_features,
                      label_nbr)
        )

    def forward(self, x):
        x = self.transformer(x)[1]
        return self.classifier(x)


def encode_dataset(tokenizer,
                   example,
                   max_length
                   ):
    return tokenizer(example['sentence'],
                     padding='max_length',
                     truncation=True,
                     max_length=max_length)


def train_epoch(model,
                optimizer,
                dataset,
                batch_size,
                device):
    dataloader_train = DataLoader(dataset['train'],
                                  shuffle=True,
                                  batch_size=batch_size)

    for batch in tqdm(dataloader_train, total=len(dataloader_train)):
        optimizer.zero_grad()
        preds = model(batch['input_ids'].long().to(device))
        loss = F.cross_entropy(preds, batch['label'].to(device))
        loss.backward()
        optimizer.step()


def evaluate(model,
             dataset,
             batch_size,
             device):
    dataloader_test = DataLoader(dataset['test'],
                                 shuffle=True,
                                 batch_size=batch_size)

    with torch.no_grad():
        eval_preds = []
        eval_labels = []

        for batch in tqdm(dataloader_test, total=len(dataloader_test)):
            preds = model(batch['input_ids'].long().to(device))
            preds = preds.argmax(dim=-1)
            eval_preds.append(preds.cpu().numpy())
            eval_labels.append(batch['label'].cpu().numpy())

    return np.concatenate(eval_labels), np.concatenate(eval_preds)


def calculate_metric(metric_name,
                     labels,
                     preds,
                     reference_class):
    if metric_name == "accuracy":
        return (labels == preds).mean()

    # binarize labels referent to class
    labels = (labels == reference_class).astype(np.float32)
    preds = (preds == reference_class).astype(np.float32)

    # get the metrics as boss
    if metric_name == "f1_score":
        return f1_score(labels, preds)

    # if metric is not put here, just return the accuracy for binarized labels
    return (labels == preds).mean()


def hp_search(trial: optuna.Trial,
              model_name: str,
              dataset,
              label_nbr,
              metric_name,
              reference_class,
              device):

    lr = trial.suggest_float("lr", 1e-7, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 6])
    seed = trial.suggest_int("seed", 1, 40)
    epochs = trial.suggest_int("epochs", 1, 5)

    model = MultilabeledSequenceModel(pretrained_model_name=model_name,
                                      label_nbr=label_nbr).to(device)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_epoch(model,
                    optimizer,
                    dataset,
                    batch_size,
                    device)

        labels, preds = evaluate(model,
                                  dataset,
                                  batch_size,
                                  device)

        metric = calculate_metric(metric_name,
                                  labels,
                                  preds,
                                  reference_class)

        trial.report(metric, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", default="bert-base-multilingual-cased", type=str,
                        help="Pretrained model name to be fetched from HuggingFace repo")
    parser.add_argument("--train-data-path", default="train.csv", type=str,
                        help="Path of the train .csv file")
    parser.add_argument("--test-data-path", default="test.csv", type=str,
                        help="Path of the test .csv file")
    parser.add_argument("--max-sequence-length", type=int, default=25, help="Max length for the sequence to be padded "
                                                                            "with")
    parser.add_argument("--metric", type=str, default="accuracy", help="metric for optune to optimize the "
                                                                       "hyperparameters for")
    parser.add_argument("--reference-class", type=int, default=1, help="reference class index for the metric, in case "
                                                                       "of binary-specific metrics to optimize the "
                                                                       "model for")
    parser.add_argument("--label-nbr", type=int, default=2, help="Number of labels for the dataset")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = nlp.load_dataset('csv', data_files={
        'train': f"{args.test_data_path}",
        'test': f"{args.test_data_path}"
    })

    dataset = dataset.map(lambda i: encode_dataset(tokenizer, i, args.max_sequence_length))
    dataset.set_format(type='torch', columns=['input_ids', 'label'])

    objective = lambda trial: hp_search(trial,
                                        model_name=args.model_name,
                                        dataset=dataset,
                                        label_nbr=args.label_nbr,
                                        metric_name=args.metric,
                                        reference_class=1,
                                        device=device)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=1800)