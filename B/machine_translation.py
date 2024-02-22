"""
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2024-02-08 21:17:17
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2024-02-08 21:26:11
FilePath: /DLNLP_assignment23_24-SN23043574/A/sentiment_analysis/Pretrained.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import torch.nn as nn
import numpy as np

import torch.nn as nn
import numpy as np
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
        print("Start training......")
        losses = []
        for epoch_idx in range(self.epochs):
            for batch_idx, (input_batch, label_batch) in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                # Forward pass
                model_out = self.model.forward(
                    input_ids=input_batch, labels=label_batch
                )

                # Calculate loss and update weights
                loss = model_out.loss
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()

                # Print training update info
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
        print("Start testing......")
        test_losses = []
        for i, (input_batch, label_batch) in enumerate(test_dataloader):
            if i >= max_iters:
                break
            model_out = self.model.forward(input_ids=input_batch, labels=label_batch)
            test_losses.append(model_out.loss.item())
        print("Finish testing.")
        return test_losses
