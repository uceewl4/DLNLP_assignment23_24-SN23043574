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


class Pretrained(nn.Module):
    def __init__(self, method, device, epochs=10, lr=1e-5):
        super(Pretrained, self).__init__()
        self.method = method
        self.device = device
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     "bert-base-uncased", num_labels=4
        # )  # 4 class
        self.model = LongformerForSequenceClassification.from_pretrained(
            "jpwahle/longformer-base-plagiarism-detection",
            num_labels=4,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        self.model.to(device)
        self.lr = lr
        self.epochs = epochs
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
                train_pred.append(
                    torch.argmax(train_logits, dim=-1)
                )  # from logits argmax
                train_labels.append(train_label)

                if step % 10 == 0:  # last time result
                    self.model.eval()
                    # self.model.to(device)

                    val_step_losses, val_pred, val_labels = [], [], []

                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_losses = []
                            # Move batch data to the same device as the model
                            val_input_ids = val_batch[0].to(self.device)  # id tokens
                            val_attention_mask = val_batch[1].to(self.device)
                            val_label = val_batch[2].to(self.device)

                            val_output = self.model(
                                val_input_ids,
                                attention_mask=val_attention_mask,
                                labels=val_label,
                            )
                            val_loss = val_output.loss
                            val_logits = val_output.logits

                            val_pred.append(
                                torch.argmax(val_logits, dim=-1)
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

                test_output = self.model(
                    test_input_ids,
                    attention_mask=test_attention_mask,
                    labels=test_label,
                )
                test_loss = test_output.loss
                test_logits = test_output.logits

                test_pred.append(
                    torch.argmax(test_logits, dim=-1)
                )  # from logits argmax
                test_labels.append(test_label)
                test_losses.append(test_loss)
            print(f"Finish testing. Test loss: {torch.stack(test_losses).mean()}")

        return test_pred, test_labels
