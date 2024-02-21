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
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
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
        multilabel=False,
    ):
        super(Ensemble, self).__init__()
        self.method = method
        self.device = device
        self.multilabel = multilabel
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     "bert-base-uncased", num_labels=4
        # )  # 4 class
        self.pretrained = Pretrained(method=method, device=device, lr=lr, epochs=epochs)
        self.rnn = RNN(
            method=method,
            device=device,
            input_dim=input_dim,
            output_dim=output_dim,
            bidirectional=bidirectional,
        )
        self.num_class = 4
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.loss_fn = torch.nn.CrossEntropyLoss()

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
                train_attention_mask = train_batch[1].to(self.device)
                train_label = (
                    F.one_hot(train_batch[2], num_classes=self.num_class)
                    .type(torch.float)
                    .to(self.device)
                )  # (8,8)

                self.optimizer.zero_grad()  # Zero the gradients
                pretrained_train_output = self.pretrained.model(
                    train_input_ids,
                    attention_mask=train_attention_mask,
                    labels=train_label,
                )
                rnn_train_output = self.rnn(train_input_ids)

                # pretrained_train_prob = torch.nn.functional.log_softmax(pretrained_train_output.logits)  # from logits to log softmax
                pretrained_train_logits = pretrained_train_output.logits
                total_train_prob = (
                    self.alpha * pretrained_train_logits
                    + (1 - self.alpha) * rnn_train_output
                )
                total_train_loss = self.loss_fn(total_train_prob, train_label)

                total_train_loss.backward()  # Compute the gradient of the loss
                self.optimizer.step()  # Update model parameters
                progress_bar.update(1)
                train_losses.append(total_train_loss.item())

                if self.multilabel == False:
                    train_pred += torch.argmax(
                        total_train_prob, dim=-1
                    ).tolist()  # from logits argmax
                elif self.multilabel == True:
                    top_values, top_indices = torch.topk(total_train_prob, 3, dim=1)
                    for index, i in enumerate(train_batch[2].tolist()):
                        if i in top_indices[index]:
                            train_pred.append(i)
                        else:
                            train_pred.append(
                                torch.argmax(total_train_prob[index]).item()
                            )
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

            min_average_loss = float("inf")
            val_min_pred, val_min_labels = [], []
            val_epoch_losses, val_epoch_accs = [], []
            progress_bar_val = tqdm(range(9 * len(val_dataloader)))
            for alpha in np.arange(0.1, 1, 0.25):
                # self.model.to(device)
                val_pred, val_labels = [], []
                for val_batch in val_dataloader:
                    val_losses = []
                    # Move batch data to the same device as the model
                    val_input_ids = val_batch[0].to(self.device)  # id tokens
                    val_attention_mask = val_batch[1].to(self.device)
                    val_label = (
                        F.one_hot(val_batch[2], num_classes=self.num_class)
                        .type(torch.float)
                        .to(self.device)
                    )  # (8,20)

                    self.optimizer.zero_grad()
                    pretrained_val_output = self.pretrained.model(
                        val_input_ids,
                        attention_mask=val_attention_mask,
                        labels=val_label,
                    )
                    rnn_val_output = self.rnn(val_input_ids)

                    # pretrained_train_prob = torch.nn.functional.log_softmax(pretrained_train_output.logits)  # from logits to log softmax
                    pretrained_val_logits = pretrained_val_output.logits
                    total_val_prob = (
                        alpha * pretrained_val_logits + (1 - alpha) * rnn_val_output[0]
                    )
                    total_val_loss = self.loss_fn(total_val_prob, val_label)
                    total_val_loss.backward()
                    self.optimizer.step()  # Update model parameters
                    progress_bar_val.update(1)
                    if self.multilabel == False:
                        val_pred += torch.argmax(
                            total_val_prob, dim=-1
                        ).tolist()  # from logits argmax
                    elif self.multilabel == True:
                        top_values, top_indices = torch.topk(total_val_prob, 3, dim=1)
                        for index, i in enumerate(val_batch[2].tolist()):
                            if i in top_indices[index]:
                                val_pred.append(i)
                            else:
                                val_pred.append(
                                    torch.argmax(total_val_prob[index]).item()
                                )
                    val_labels += val_batch[2].tolist()
                    val_losses.append(total_val_loss.item())

                val_pred = np.array(val_pred)
                val_epoch_loss = np.array(val_losses).mean()
                val_epoch_acc = round(
                    accuracy_score(
                        np.array(val_labels).astype(int), val_pred.astype(int)
                    )
                    * 100,
                    4,
                )
                if val_epoch_loss <= min_average_loss:
                    self.alpha = alpha
                    print(
                        print(
                            f"\nval loss: {val_epoch_loss}, acc: {val_epoch_acc}, best alpha: {self.alpha}"
                        )
                    )
                    val_epoch_losses.append(val_epoch_loss)
                    val_epoch_accs.append(val_epoch_acc)
                    min_average_loss = val_epoch_loss
                    val_min_labels = val_labels
                    val_min_pred = val_pred

        print("Finish training.")

        return (
            train_epoch_losses,
            train_epoch_accs,
            val_epoch_losses,
            val_epoch_accs,
            train_pred,
            val_min_pred,
            train_labels,
            val_min_labels,
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
                test_attention_mask = test_batch[1].to(self.device)
                test_label = (
                    F.one_hot(test_batch[2], num_classes=self.num_class)
                    .type(torch.float)
                    .to(self.device)
                )

                pretrained_test_output = self.pretrained.model(
                    test_input_ids,
                    attention_mask=test_attention_mask,
                    labels=test_label,
                )
                rnn_test_output = self.rnn(test_input_ids)

                # pretrained_train_prob = torch.nn.functional.log_softmax(pretrained_train_output.logits)  # from logits to log softmax
                pretrained_test_logits = pretrained_test_output.logits
                total_test_prob = (
                    self.alpha * pretrained_test_logits
                    + (1 - self.alpha) * rnn_test_output[0]
                )
                total_test_loss = self.loss_fn(total_test_prob, test_label)
                progress_bar_test.update(1)
                if self.multilabel == False:
                    test_pred += torch.argmax(
                        total_test_prob, dim=-1
                    ).tolist()  # from logits argmax
                elif self.multilabel == True:
                    top_values, top_indices = torch.topk(total_test_prob, 3, dim=1)
                    for index, i in enumerate(test_batch[2].tolist()):
                        if i in top_indices[index]:
                            test_pred.append(i)
                        else:
                            test_pred.append(
                                torch.argmax(total_test_prob[index]).item()
                            )
                test_labels += test_batch[2].tolist()
                test_losses.append(total_test_loss.item())
            test_pred = np.array(test_pred)
            print(f"\nFinish testing. Test loss: {np.array(test_losses).mean()}")

        return test_pred, test_labels
