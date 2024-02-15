"""
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2024-02-08 21:17:17
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2024-02-08 21:26:11
FilePath: /DLNLP_assignment23_24-SN23043574/A/sentiment_analysis/Pretrained.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    LongformerForSequenceClassification,
)
from torch.optim import Adam
from tqdm.auto import tqdm
import numpy as np
import torch


class RNN(nn.Module):
    def __init__(
        self,
        method,
        device,
        input_dim,
        output_dim,
        bidirectional=False,
        epochs=10,
        lr=1e-5,
        multilabel=False,
    ):
        super(RNN, self).__init__()
        self.method = method
        self.device = device
        self.multilabel = multilabel

        self.embedding = nn.Embedding(input_dim, output_dim)
        self.rnn = nn.RNN(
            input_size=output_dim,
            hidden_size=output_dim,
            num_layers=20,
            batch_first=True,
            dropout=0.5,
            bidirectional=bidirectional,
        )  # input_size 每个词的维度，hidden_size 神经元的个数
        if bidirectional:
            self.linear_1 = nn.Linear(
                in_features=output_dim * 2, out_features=output_dim, bias=True
            )
        else:
            self.linear_1 = nn.Linear(
                in_features=output_dim, out_features=int(output_dim / 2), bias=True
            )
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(
            in_features=int(output_dim / 2), out_features=8, bias=True
        )
        self.output = nn.Softmax()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.lr = lr
        self.epochs = epochs

    def forward(self, x):
        x = self.embedding(x)
        out, h_n = self.rnn(x)  # out: 8,133,64, h_n 3,8,64
        x = self.linear_1(out[:, -1, :])  # take the last output
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.output(x)
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
                train_input_ids = train_batch[0].to(self.device)  # id tokens
                train_label = train_batch[1].to(self.device)  # 64

                self.optimizer.zero_grad()  # Zero the gradients
                train_output = model(train_input_ids)
                train_loss = self.loss_fn(train_output, train_label)
                train_loss.backward()  # Compute the gradient of the loss
                self.optimizer.step()  # Update model parameters
                progress_bar.update(1)
                train_losses.append(train_loss.item())

                if self.multilabel == False:
                    train_pred += torch.argmax(
                        train_output, dim=-1
                    ).tolist()  # from logits argmax
                elif self.multilabel == True:
                    top_values, top_indices = torch.topk(train_output, 3, dim=1)
                    for index, i in enumerate(train_batch[2].tolist()):
                        if i in top_indices[index]:
                            train_pred.append(i)
                        else:
                            train_pred.append(torch.argmax(train_output[index]).item())
                train_labels += train_batch[2].tolist()

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

            # model.eval()
            val_pred, val_labels = [], []
            progress_bar_val = tqdm(range(len(val_dataloader)))
            # with torch.no_grad():
            for val_batch in val_dataloader:
                val_losses = []
                # Move batch data to the same device as the model
                val_input_ids = val_batch[0].to(self.device)  # id tokens
                val_label = val_batch[1].to(self.device)

                self.optimizer.zero_grad()
                val_output = model(val_input_ids)
                val_loss = self.loss_fn(val_output, val_label)
                val_loss.backward()
                self.optimizer.step()  # Update model parameters
                progress_bar_val.update(1)

                if self.multilabel == False:
                    val_pred += torch.argmax(
                        val_output, dim=-1
                    ).tolist()  # from logits argmax
                elif self.multilabel == True:
                    top_values, top_indices = torch.topk(val_output, 3, dim=1)
                    for index, i in enumerate(val_batch[2].tolist()):
                        if i in top_indices[index]:
                            val_pred.append(i)
                        else:
                            val_pred.append(torch.argmax(val_output[index]).item())
                val_labels += val_batch[2].tolist()
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
        # self.model.to(device)
        # model.eval()
        test_losses, test_pred, test_labels = [], [], []
        progress_bar_test = tqdm(range(len(test_dataloader)))

        with torch.no_grad():
            for test_batch in test_dataloader:
                # Move batch data to the same device as the model
                test_input_ids = test_batch[0].to(self.device)  # id tokens
                test_label = test_batch[1].to(self.device)

                test_output = model(test_input_ids)
                test_loss = self.loss_fn(test_output, test_label).item()
                progress_bar_test.update(1)

                if self.multilabel == False:
                    test_pred += torch.argmax(
                        test_output, dim=-1
                    ).tolist()  # from logits argmax
                elif self.multilabel == True:
                    top_values, top_indices = torch.topk(test_output, 3, dim=1)
                    for index, i in enumerate(test_batch[2].tolist()):
                        if i in top_indices[index]:
                            test_pred.append(i)
                        else:
                            test_pred.append(torch.argmax(test_output[index]).item())
                test_labels += test_batch[2].tolist()
                test_losses.append(test_loss)
            test_pred = np.array(test_pred)
            print(f"Finish testing. Test loss: {np.array(test_losses).mean()}")

        return test_pred, test_labels
