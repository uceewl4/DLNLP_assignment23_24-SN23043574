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


class LSTM(nn.Module):
    def __init__(
        self,
        method,
        device,
        embeddings,
        output_dim,
        bidrectional=False,
        epochs=10,
        lr=1e-5,
        grained="fine",
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
            bidirectional=bidrectional,
        )  # input_size 每个词的维度，hidden_size 神经元的个数
        self.out_features = 6 if grained == "fine" else 2
        if bidrectional:
            self.linear = nn.Linear(
                in_features=output_dim * 2, out_features=self.out_features, bias=True
            )
        else:
            self.linear = nn.Linear(
                in_features=output_dim, out_features=self.out_features, bias=True
            )
        self.output = nn.Softmax()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.lr = lr
        self.epochs = epochs
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.embedding(x)
        out, (h_n, c_n) = self.lstm(x)
        x = self.linear(h_n)  # take the last hidden state
        x = self.output(x)  # logits
        return x

    def train(self, model, train_dataloader, val_dataloader):
        print("Start training......")
        progress_bar = tqdm(range(self.epochs * len(train_dataloader)))
        train_epoch_losses = []

        for epoch in range(self.epochs):
            train_losses, train_pred, train_labels = [], [], []
            for step, train_batch in enumerate(train_dataloader):
                train_input_ids = train_batch[0].to(self.device)  # id tokens
                train_label = train_batch[1].to(self.device)  # 64

                self.optimizer.zero_grad()  # Zero the gradients
                train_output = model(train_input_ids)
                train_loss = self.loss_fn(
                    train_output[0], train_label
                ).item()  # not from logits
                train_loss.backward()  # Compute the gradient of the loss
                self.optimizer.step()  # Update model parameters
                progress_bar.update(1)
                train_losses.append(train_loss)

                train_pred.append(
                    torch.argmax(train_output[0], dim=-1)
                )  # from logits argmax
                train_labels.append(train_label)

                if step % 10 == 0:  # last time result
                    model.eval()
                    # self.model.to(device)

                    val_step_losses, val_pred, val_labels = [], [], []

                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_losses = []
                            # Move batch data to the same device as the model
                            val_input_ids = val_batch[0].to(self.device)  # id tokens
                            val_label = val_batch[1].to(self.device)

                            val_output = model(val_input_ids)
                            val_loss = self.loss_fn(val_output[0], val_label).item()

                            val_pred.append(
                                torch.argmax(val_output[0], dim=-1)
                            )  # from logits argmax
                            val_labels.append(val_label)
                            val_losses.append(val_loss)
                        val_step_loss = torch.stack(
                            val_losses
                        ).mean()  # stack value together
                        print(f"Step {step} complete, val loss: {val_step_loss}")
                        val_step_losses.append(val_step_loss)

            train_epoch_loss = np.mean(train_losses)
            print(f"Epoch {epoch} complete, train loss: {train_epoch_losses}")
            train_epoch_losses.append(train_epoch_loss)
            print("Finish training.")

            return (
                train_epoch_losses,
                val_step_losses,
                train_pred,
                val_pred,
                train_labels,
                val_labels,
            )

    def test(self, model, test_dataloader):
        print("Start testing......")
        model.eval()
        # self.model.to(device)
        test_losses, test_pred, test_labels = [], [], []

        with torch.no_grad():
            for test_batch in test_dataloader:
                # Move batch data to the same device as the model
                test_input_ids = test_batch[0].to(self.device)  # id tokens
                test_label = test_batch[1].to(self.device)

                test_output = model(test_input_ids)
                test_loss = self.loss_fn(test_output[0], test_label).item()

                test_pred.append(
                    torch.argmax(test_output[0], dim=-1)
                )  # from logits argmax
                test_labels.append(test_label)
                test_losses.append(test_loss)
            print(f"Finish testing. Test loss: {torch.stack(test_losses).mean()}")

        return test_pred, test_labels
