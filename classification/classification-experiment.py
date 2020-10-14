from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
import nlp

from sklearn.metrics import confusion_matrix
import numpy as np

from tqdm import tqdm

import argparse


class MultilabeledSequenceModel(nn.Module):
    def __init__(self,
                 pretrained_model_name,
                 label_nbr):
        super().__init__()
        self.transformer = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(list(self.transformer.modules())[-2].out_features,
                      label_nbr)
        )

    def forward(self, x):
        x = self.transformer(x)[1]
        return self.classifier(x)


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


def encode_dataset(tokenizer,
                   example,
                   max_length
                   ):
    return tokenizer(example['sentence'],
                     padding='max_length',
                     truncation=True,
                     max_length=max_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", default="bert-base-multilingual-cased", type=str, required=True,
                        help="Pretrained model name to be fetched from HuggingFace repo")
    parser.add_argument("--train-data-path", default="train.csv", type=str, required=True,
                        help="Path of the train .csv file")
    parser.add_argument("--test-data-path", default="test.csv", type=str, required=True,
                        help="Path of the test .csv file")
    parser.add_argument("--max-sequence-length", type=int, default=25, help="Max length for the sequence to be padded "
                                                                            "with")
    parser.add_argument("--metric", type=str, default="accuracy", help="metric for optune to optimize the "
                                                                       "hyperparameters for")
    parser.add_argument("--reference-class", type=int, default=1, help="reference class index for the metric, in case "
                                                                       "of binary-specific metrics to optimize the "
                                                                       "model for")

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = nlp.load_dataset('csv', data_files={
        'train': f"{args.test_data_path}",
        'test': f"{args.test_data_path}"
    })

    dataset = dataset.map(encode_dataset)
    dataset.set_format(type='torch', columns=['input_ids', 'label'])

