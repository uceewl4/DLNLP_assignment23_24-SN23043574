"""
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2024-02-08 21:17:17
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2024-02-08 21:26:11
FilePath: /DLNLP_assignment23_24-SN23043574/A/sentiment_analysis/Pretrained.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    LongformerForSequenceClassification,
)
from torch.optim import Adam
from tqdm.auto import tqdm
import numpy as np
import torch

from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    LongformerForSequenceClassification,
)
from torch.optim import Adam
from tqdm.auto import tqdm
import numpy as np
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm_notebook
import torch.nn.functional as F


class NER(nn.Module):
    def __init__(
        self,
        method,
        device,
        input_dim,
        output_dim,
        n_tags,
        epochs=10,
        lr=1e-4,
    ):
        super(NER, self).__init__()
        self.method = method
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_tags = n_tags
        self.embedding = nn.Embedding(input_dim, output_dim)
        # input表示词典中词的总数，output表示每个词用多少维表示，直接作用于句子的词典表达

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=output_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
        )  # input_size 每个词的维度，hidden_size 神经元的个数

        self.linear = nn.Linear(output_dim, n_tags)  # 分类成12个tag  64--12
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(lr=lr, weight_decay=1e-4)
        self.epochs = epochs
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1 = self.embedding(x)
        x_2, _ = self.lstm(x_1)
        seq_len = x_2.size(1)  # 104
        x_3 = x_2.contiguous().view(-1, x_2.size(2))
        x_4 = self.relu(self.linear(x_3))
        x_end = x_4.view(-1, seq_len, self.n_tags)
        return x_end

    def train(self, model, x_train, y_train, x_val, y_val):
        print("Start training......")
        for epoch in range(self.epochs):
            model.train()  # switch into training mode
            loss_total = 0  # calculate the average loss each time

            # Assuming 'data_loader' is your DataLoader for training data
            for inputs, labels in zip(x_train, y_train):  # already divide into batches
                # Reshape inputs and labels to match the network structure
                inputs = inputs.view(-1, 104)  # Reshape inputs if necessary  # 64 104
                labels = labels.view(
                    -1, 17
                )  # Labels need to be in the shape [batch_size*104, 17]  # ground truth for each word

                # Forward pass
                outputs = model(inputs)
                outputs = outputs.view(
                    -1, 17
                )  # Reshape outputs to match labels shape  64*104,17

                # Compute loss
                loss = self.criterion(outputs, labels)
                loss_total += loss.item()
                # print (loss)

                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # evaluation
        size = (
            x_val.shape[0] * x_val.shape[1] * x_val.shape[2]
        )  # total number of words  batch_num, batch_size, sentence length
        num_batches = x_val.shape[0]
        val_loss, correct = 0, 0

        with torch.no_grad():
            for i in range(x_val.shape[0]):  # for each batch
                X, y = x_val[i].reshape(-1, 104), y_val[i].reshape(
                    -1, 17
                )  # (64,104), (64,104,17), every word one predicted output
                pred = model(X).reshape(-1, 17)
                val_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(axis=-1) == y.argmax(axis=-1)).sum()

        val_loss /= num_batches  # average of the batch
        accuracy = correct / size
        print(f"Validation Error: Avg loss: {val_loss:>8f}")
        print(f"Validation Accuracy: {accuracy:>8f}")

    def test(self, model, x_test, y_test):
        print("Start testing......")
        size = (
            x_test.shape[0] * x_test.shape[1] * x_test.shape[2]
        )  # total number of words  batch_num, batch_size, sentence length
        num_batches = x_test.shape[0]
        test_loss, correct = 0, 0

        with torch.no_grad():
            for i in range(x_test.shape[0]):  # for each batch
                X, y = x_test[i].reshape(-1, 104), y_test[i].reshape(
                    -1, 17
                )  # (64,104), (64,104,17), every word one predicted output
                pred = model(X).reshape(-1, 17)
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(axis=-1) == y.argmax(axis=-1)).sum()

        test_loss /= num_batches  # average of the batch
        accuracy = correct / size
        print(f"Test Error: Avg loss: {test_loss:>8f}")
        print(f"Test Accuracy: {accuracy:>8f}")
        print("Finish testing.")
