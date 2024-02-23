# -*- encoding: utf-8 -*-
"""
@File    :   NER.py
@Time    :   2024/02/23 18:58:07
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0141: Deep Learning for Natural Language Processing
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all process and attributes of name entity recognition.
The code refers to DLNLP lab 1.
"""

# here put the import lib

import torch.nn as nn
import torch
from torch import optim


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
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=output_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
        )
        self.linear = nn.Linear(output_dim, n_tags)  # 17 classes
        self.relu = nn.ReLU()

        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.lr = lr

    def forward(self, x):
        x_1 = self.embedding(x)
        x_2, _ = self.lstm(x_1)
        seq_len = x_2.size(1)
        x_3 = x_2.contiguous().view(-1, x_2.size(2))
        x_4 = self.relu(self.linear(x_3))
        x_end = x_4.view(-1, seq_len, self.n_tags)
        return x_end

    def train(self, model, x_train, y_train, x_val, y_val):
        """
        description: This is the process for training and validation.
        param {*} self
        param {*} model
        param {*} x_train: training data
        param {*} y_train: train label
        param {*} x_val: validation data
        param {*} y_val: validation label
        """
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        print("Start training......")
        for epoch in range(self.epochs):
            loss_total = 0
            for inputs, labels in zip(x_train, y_train):  # batches
                inputs = inputs.view(-1, 104)  # reshape
                labels = labels.view(-1, 17)

                outputs = model(inputs)
                outputs = outputs.view(-1, 17)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item()
                self.optimizer.zero_grad()
                loss.backward()  # back propagate
                self.optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # validation
        size = x_val.shape[0] * x_val.shape[1] * x_val.shape[2]
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
        """
        description: This method is for the testing process.
        param {*} self
        param {*} model
        param {*} x_test: testing data
        param {*} y_test: testing labels
        """
        print("Start testing......")
        size = (
            x_test.shape[0] * x_test.shape[1] * x_test.shape[2]
        )  # total number of words  batch_num, batch_size, sentence length
        num_batches = x_test.shape[0]
        test_loss, correct = 0, 0

        with torch.no_grad():
            for i in range(x_test.shape[0]):
                X, y = x_test[i].reshape(-1, 104), y_test[i].reshape(-1, 17)
                pred = model(X).reshape(-1, 17)
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(axis=-1) == y.argmax(axis=-1)).sum()

        test_loss /= num_batches  # average of the batch
        accuracy = correct / size
        print(f"Test Error: Avg loss: {test_loss:>8f}")
        print(f"Test Accuracy: {accuracy:>8f}")
        print("Finish testing.")
