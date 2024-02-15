"""
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2024-02-08 21:17:17
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2024-02-08 21:26:11
FilePath: /DLNLP_assignment23_24-SN23043574/A/sentiment_analysis/Pretrained.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

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
import torch.nn.functional as F


class Pretrained(nn.Module):
    def __init__(
        self, method, device, epochs=10, lr=1e-5, grained="fine", multilabel=False
    ):
        super(Pretrained, self).__init__()
        self.method = method
        self.device = device
        self.multilabel = multilabel
        self.num_class = 20 if grained == "fine" else 2
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     "bert-base-uncased", num_labels=4
        # )  # 4 class
        if grained == "fine":
            self.model = LongformerForSequenceClassification.from_pretrained(
                "jpwahle/longformer-base-plagiarism-detection",
                num_labels=self.num_class,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True,
            )
        else:
            self.model = LongformerForSequenceClassification.from_pretrained(
                "jpwahle/longformer-base-plagiarism-detection"
            )
        self.model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_dataloader, val_dataloader):
        print("Start training......")
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
                )  # (8,20)

                self.optimizer.zero_grad()  # Zero the gradients
                train_output = self.model(
                    train_input_ids,
                    attention_mask=train_attention_mask,
                    labels=train_label,
                )
                train_loss = train_output.loss
                train_loss.backward()  # Compute the gradient of the loss
                self.optimizer.step()  # Update model parameters
                progress_bar.update(1)
                train_losses.append(train_loss.cpu().detach())

                train_logits = train_output.logits
                if self.multilabel == False:
                    train_pred += torch.argmax(
                        train_logits, dim=-1
                    ).tolist()  # from logits argmax
                elif self.multilabel == True:
                    top_values, top_indices = torch.topk(train_logits, 3, dim=1)
                    for index, i in enumerate(train_batch[2].tolist()):
                        if i in top_indices[index]:
                            train_pred.append(i)
                        else:
                            train_pred.append(torch.argmax(train_logits[index]).item())
                train_labels += train_batch[2].tolist()

            train_pred = np.array(train_pred)
            train_epoch_loss = np.mean(train_losses)
            train_epoch_acc = round(
                accuracy_score(
                    np.array(train_labels).astype(int), train_pred.astype(int)
                )
                * 100,
                4,
            )
            print(
                f"\nEpoch {epoch} complete, train loss: {round(train_epoch_loss,4)}, acc: {train_epoch_acc}"
            )
            train_epoch_accs.append(train_epoch_acc)
            self.model.eval()
            # self.model.to(device)

            val_pred, val_labels = [], []
            progress_bar_val = tqdm(range(len(val_dataloader)))

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_losses = []
                    # Move batch data to the same device as the model
                    val_input_ids = val_batch[0].to(self.device)  # id tokens
                    val_attention_mask = val_batch[1].to(self.device)
                    val_label = (
                        F.one_hot(val_batch[2], num_classes=self.num_class)
                        .type(torch.float)
                        .to(self.device)
                    )

                    val_output = self.model(
                        val_input_ids,
                        attention_mask=val_attention_mask,
                        labels=val_label,
                    )
                    val_loss = val_output.loss
                    val_logits = val_output.logits
                    if self.multilabel == False:
                        val_pred += torch.argmax(
                            val_logits, dim=-1
                        ).tolist()  # from logits argmax
                    elif self.multilabel == True:
                        top_values, top_indices = torch.topk(val_logits, 3, dim=1)
                        for index, i in enumerate(val_batch[2].tolist()):
                            if i in top_indices[index]:
                                val_pred.append(i)
                            else:
                                val_pred.append(torch.argmax(val_logits[index]).item())
                    val_labels += val_batch[2].tolist()

                    val_losses.append(val_loss)
                    progress_bar_val.update(1)

                val_pred = np.array(val_pred)
                val_epoch_acc = round(
                    accuracy_score(
                        np.array(val_labels).astype(int), val_pred.astype(int)
                    )
                    * 100,
                    4,
                )
                val_epoch_loss = torch.stack(val_losses).mean()  # stack value together
                print(f"\nval loss: {val_epoch_loss}, acc: {val_epoch_acc}")
                val_epoch_losses.append(val_epoch_loss.item())
                val_epoch_accs.append(val_epoch_acc)

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

    def test(self, test_dataloader):
        print("Start testing......")
        self.model.eval()
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

                test_output = self.model(
                    test_input_ids,
                    attention_mask=test_attention_mask,
                    labels=test_label,
                )
                test_loss = test_output.loss
                test_logits = test_output.logits
                progress_bar_test.update(1)

                if self.multilabel == False:
                    test_pred += torch.argmax(
                        test_logits, dim=-1
                    ).tolist()  # from logits argmax
                elif self.multilabel == True:
                    top_values, top_indices = torch.topk(test_logits, 3, dim=1)
                    for index, i in enumerate(test_batch[2].tolist()):
                        if i in top_indices[index]:
                            test_pred.append(i)
                        else:
                            test_pred.append(torch.argmax(test_logits[index]).item())
                test_labels += test_batch[2].tolist()
                test_losses.append(test_loss)
            test_pred = np.array(test_pred)
            print(f"\nFinish testing. Test loss: {torch.stack(test_losses).mean()}")

        return test_pred, test_labels
