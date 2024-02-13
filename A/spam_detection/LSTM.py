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


class LSTM(nn.Module):
    def __init__(
        self,
        method,
        device,
        embeddings,
        output_dim,
        bidirectional=False,
        epochs=10,
        lr=1e-5,
    ):
        super(LSTM, self).__init__()
        self.method = method
        self.device = device

        self.embedding = torch.nn.Embedding.from_pretrained(torch.Tensor(embeddings))
        self.lstm = nn.LSTM(
            input_size=embeddings.shape[1],  # dimension of word
            hidden_size=output_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
            bidirectional=bidirectional,
        )  # input_size 每个词的维度，hidden_size 神经元的个数
        if bidirectional:
            self.linear = nn.Linear(
                in_features=output_dim * 2, out_features=2, bias=True
            )
        else:
            self.linear = nn.Linear(in_features=output_dim, out_features=2, bias=True)
        self.output = nn.Softmax()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.lr = lr
        self.epochs = epochs

    def forward(self, x):
        x = self.embedding(x)
        out, (h_n, c_n) = self.lstm(x)
        x = self.linear(h_n)  # take the last hidden state
        x = self.output(x)  # logits
        return x

    def train(self, model, train_dataloader, val_dataloader):
        self.optimizer = Adam(model.parameters(), lr=self.lr)
        model.to(self.device)
        print("Start training......")
        # model.train()
        train_epoch_losses, train_epoch_accs = [], []
        val_epoch_losses, val_epoch_accs = [], []

        for epoch in range(self.epochs):
            progress_bar = tqdm(range(len(train_dataloader)))
            train_losses, train_pred, train_labels = [], [], []
            for step, train_batch in enumerate(train_dataloader):
                train_input_ids = train_batch[0].to(self.device)  # id tokens  8,195
                train_label = train_batch[1].to(self.device)  # 8

                self.optimizer.zero_grad()  # Zero the gradients
                train_output = model(train_input_ids)
                train_loss = self.loss_fn(
                    train_output[0], train_label
                )  # not from logits
                train_loss.backward()  # Compute the gradient of the loss
                self.optimizer.step()  # Update model parameters
                progress_bar.update(1)
                train_losses.append(train_loss.item())

                train_pred += torch.argmax(
                    train_output[0], dim=-1
                ).tolist()  # from logits argmaxput[0], dim=-1)
                train_labels += train_label.tolist()

            train_pred = np.array(train_pred)
            train_epoch_loss = np.mean(train_losses)
            train_epoch_losses.append(train_epoch_loss)
            train_epoch_acc = round(
                accuracy_score(
                    np.array(train_labels).astype(int), train_pred.astype(int)
                )
                * 100,
                4,
            )
            train_epoch_accs.append(train_epoch_acc)
            print(
                f"\nEpoch {epoch} complete, train loss: {round(train_epoch_loss,4)}, acc: {train_epoch_acc}"
            )

            val_pred, val_labels = [], []
            progress_bar_val = tqdm(range(len(val_dataloader)))
            for val_batch in val_dataloader:
                val_losses = []
                # Move batch data to the same device as the model
                val_input_ids = val_batch[0].to(self.device)  # id tokens
                val_label = val_batch[1].to(self.device)

                self.optimizer.zero_grad()
                val_output = model(val_input_ids)
                val_loss = self.loss_fn(val_output[0], val_label)
                val_loss.backward()
                self.optimizer.step()  # Update model parameters
                progress_bar_val.update(1)

                val_pred += torch.argmax(
                    val_output[0], dim=-1
                ).tolist()  # from logits argmaxput[0], dim=-1)
                val_labels += val_label.tolist()
                val_losses.append(val_loss.item())

            val_pred = np.array(val_pred)
            val_epoch_acc = round(
                accuracy_score(np.array(val_labels).astype(int), val_pred.astype(int))
                * 100,
                4,
            )
            val_epoch_loss = np.array(val_losses).mean()  # stack value together
            print(f"\nval loss: {val_epoch_loss}, acc: {val_epoch_acc}")
            val_epoch_losses.append(val_epoch_loss)
            val_epoch_accs.append(val_epoch_acc)
            print(val_epoch_losses)
            print(val_epoch_accs)

        print("Finish training.")

        return (
            train_epoch_losses,
            train_epoch_accs,
            val_epoch_losses,
            val_epoch_accs,
            train_pred,
            val_pred,
            train_labels,
            val_labels,
        )

    def test(self, model, test_dataloader):
        print("Start testing......")
        # model.eval()
        # self.model.to(device)
        test_losses, test_pred, test_labels = [], [], []
        progress_bar_test = tqdm(range(len(test_dataloader)))

        with torch.no_grad():
            for test_batch in test_dataloader:
                # Move batch data to the same device as the model
                test_input_ids = test_batch[0].to(self.device)  # id tokens
                test_label = test_batch[1].to(self.device)

                test_output = model(test_input_ids)
                test_loss = self.loss_fn(test_output[0], test_label).item()
                progress_bar_test.update(1)

                test_pred += torch.argmax(
                    test_output[0], dim=-1
                ).tolist()  # from logits argmax
                test_labels += test_label.tolist()
                test_losses.append(test_loss)
            test_pred = np.array(test_pred)
            print(f"Finish testing. Test loss: {np.array(test_losses).mean()}")

        return test_pred, test_labels
