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

from A.sentiment_analysis.Pretrained import Pretrained
from A.sentiment_analysis.RNN import RNN


class Ensemble(nn.Module):
    def __init__(
        self,
        method,
        device,
        input_dim,
        output_dim,
        bidirectional=False,
        epochs=10,
        lr=1e-5,
        alpha=0.5,
        grained="fine"
    ):
        super(Ensemble, self).__init__()
        self.method = method
        self.device = device
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     "bert-base-uncased", num_labels=4
        # )  # 4 class
        self.pretrained = Pretrained(method=method, device=device, lr=lr, epochs=epochs)
        self.rnn = RNN(
            method=method,
            device=device,
            input_dim=input_dim,
            output_dim=output_dim,
            bidrectional=bidirectional,
        )
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_dataloader, val_dataloader):
        print("Start training......")
        progress_bar = tqdm(range(self.epochs * len(train_dataloader)))
        train_epoch_losses = []

        for epoch in range(self.epochs):
            train_losses, train_pred, train_labels = [], [], []
            for step, train_batch in enumerate(train_dataloader):
                train_input_ids = train_batch[0].to(self.device)  # id tokens
                train_attention_mask = train_batch[1].to(self.device)
                train_label = train_batch[2].to(self.device)  # 64

                self.optimizer.zero_grad()  # Zero the gradients
                pretrained_train_output = self.pretrained(
                    train_input_ids,
                    attention_mask=train_attention_mask,
                    labels=train_label,
                )
                rnn_train_output = self.rnn(train_input_ids)

                # pretrained_train_prob = torch.nn.functional.log_softmax(pretrained_train_output.logits)  # from logits to log softmax
                pretrained_train_logits = pretrained_train_output.logits
                total_train_prob = (
                    self.alpha * pretrained_train_logits
                    + (1 - self.alpha) * rnn_train_output[0]
                )
                total_train_loss = self.loss_fn(total_train_prob, train_label)

                total_train_loss.backward()  # Compute the gradient of the loss
                self.optimizer.step()  # Update model parameters
                progress_bar.update(1)
                train_losses.append(total_train_loss)

                train_pred.append(
                    torch.argmax(total_train_prob, dim=-1)
                )  # from logits argmax
                train_labels.append(train_label)

                if step % 10 == 0:  # last time result
                    self.pretrained.eval()
                    self.rnn.eval()
                    min_average_loss = float("inf")
                    val_min_pred, val_min_labels = [], []

                    for alpha in np.arange(0.1, 1, 0.1):
                        # self.model.to(device)
                        val_step_losses, val_pred, val_labels = [], [], []

                        with torch.no_grad():
                            for val_batch in val_dataloader:
                                val_losses = []
                                # Move batch data to the same device as the model
                                val_input_ids = val_batch[0].to(
                                    self.device
                                )  # id tokens
                                val_attention_mask = val_batch[1].to(self.device)
                                val_label = val_batch[2].to(self.device)

                                pretrained_val_output = self.pretrained(
                                    val_input_ids,
                                    attention_mask=val_attention_mask,
                                    labels=val_label,
                                )
                                rnn_val_output = self.rnn(val_input_ids)

                                # pretrained_train_prob = torch.nn.functional.log_softmax(pretrained_train_output.logits)  # from logits to log softmax
                                pretrained_val_logits = pretrained_val_output.logits
                                total_val_prob = (
                                    alpha * pretrained_val_logits
                                    + (1 - alpha) * rnn_val_output[0]
                                )
                                total_val_loss = self.loss_fn(total_val_prob, val_label)
                                val_pred.append(
                                    torch.argmax(total_val_prob, dim=-1)
                                )  # from logits argmax
                                val_labels.append(val_label)
                                val_losses.append(total_val_loss)
                            val_step_loss = torch.stack(
                                val_losses
                            ).mean()  # stack value together

                            if val_step_loss <= min_average_loss:
                                self.alpha = alpha
                                print(
                                    f"Step {step} complete, val loss: {val_step_loss}, bset alpha: {self.alpha}"
                                )
                                val_step_losses.append(val_step_loss)
                                min_average_loss = val_step_loss
                                val_min_labels = val_labels
                                val_min_pred = val_pred

            train_epoch_loss = np.mean(train_losses)
            print(f"Epoch {epoch} complete, train loss: {train_epoch_losses}")
            train_epoch_losses.append(train_epoch_loss)
            print("Finish training.")

            return (
                train_epoch_losses,
                min_average_loss,
                train_pred,
                val_min_pred,
                train_labels,
                val_min_labels,
            )

    def test(self, test_dataloader):
        print("Start testing......")
        self.model.eval()
        # self.model.to(device)
        test_losses, test_pred, test_labels = [], [], []

        with torch.no_grad():
            for test_batch in test_dataloader:
                # Move batch data to the same device as the model
                test_input_ids = test_batch[0].to(self.device)  # id tokens
                test_attention_mask = test_batch[1].to(self.device)
                test_label = test_batch[2].to(self.device)

                pretrained_test_output = self.pretrained(
                    test_input_ids,
                    attention_mask=test_attention_mask,
                    labels=test_label,
                )
                rnn_test_output = self.rnn(test_input_ids)

                # pretrained_train_prob = torch.nn.functional.log_softmax(pretrained_train_output.logits)  # from logits to log softmax
                pretrained_val_logits = pretrained_test_output.logits
                total_test_prob = (
                    self.alpha * pretrained_val_logits
                    + (1 - self.alpha) * rnn_test_output[0]
                )
                total_test_loss = self.loss_fn(total_test_prob, test_label)
                test_pred.append(
                    torch.argmax(total_test_prob, dim=-1)
                )  # from logits argmax
                test_labels.append(test_label)
                test_losses.append(total_test_loss)
            print(f"Finish testing. Test loss: {torch.stack(test_losses).mean()}")

        return test_pred, test_labels
