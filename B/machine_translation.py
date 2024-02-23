# -*- encoding: utf-8 -*-
"""
@File    :   machine_translation.py
@Time    :   2024/02/23 18:50:35
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0141: Deep Learning for Natural Language Processing
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all process and attributes of machine translation.
The code refers to https://www.youtube.com/watch?v=P50EHx9DWDM.
"""

# here put the import lib

import torch.nn as nn
import numpy as np
from transformers import AdamW, AutoModelForSeq2SeqLM


class MT(nn.Module):
    def __init__(self, method, device, tokenizer, epochs=10, lr=1e-4, batch_size=8):
        super(MT, self).__init__()
        self.method = method
        self.device = device

        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.to(device)
        self.tokenizer = tokenizer

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def train(self, train_dataloader, train_dataset):
        """
        description: This is the method for training process.
        param {*} self
        param {*} train_dataloader
        param {*} train_dataset
        return {*}: results for training
        """
        print("Start training......")
        losses = []
        for epoch_idx in range(self.epochs):
            for batch_idx, (input_batch, label_batch) in enumerate(train_dataloader):

                self.optimizer.zero_grad()
                model_out = self.model.forward(
                    input_ids=input_batch, labels=label_batch
                )  # output
                loss = model_out.loss
                losses.append(loss.item())
                loss.backward()  # backpropagate
                self.optimizer.step()

                if (batch_idx + 1) % 10 == 0:
                    avg_loss = np.mean(losses[-10:])
                    print(
                        "Epoch: {} | Step: {} | Avg. loss: {:.3f}".format(
                            epoch_idx + 1, batch_idx + 1, avg_loss
                        )
                    )

        print("Finish training.")

        return losses

    def test(self, test_dataloader, max_iters=8):
        """
        description: This is the method for testing process.
        param {*} self
        param {*} test_dataloader
        param {*} max_iters
        return {*} test_losses: test results
        """
        print("Start testing......")
        test_losses = []
        for i, (input_batch, label_batch) in enumerate(test_dataloader):
            if i >= max_iters:
                break
            model_out = self.model.forward(input_ids=input_batch, labels=label_batch)
            test_losses.append(model_out.loss.item())
        print("Finish testing.")
        return test_losses
