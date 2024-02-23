# -*- encoding: utf-8 -*-
"""
@File    :   question_answering.py
@Time    :   2024/02/23 18:54:56
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0141: Deep Learning for Natural Language Processing
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all process and attributes of question answering.
The code refers to https://www.youtube.com/watch?v=DNRiUGtKlVU.
"""

# here put the import lib

import torch.nn as nn
from tqdm.auto import tqdm
import torch
from transformers import AdamW
from transformers import BertForQuestionAnswering
from sklearn.metrics import f1_score


class QA(nn.Module):
    def __init__(self, method, device, epochs=10, lr=1e-4, batch_size=8):
        super(QA, self).__init__()
        self.method = method
        self.device = device
        self.model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
        self.model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

    def train(self, train_dataloader, val_dataloader):
        """
        description: This is the method for training and validation process.
        param {*} self
        param {*} train_dataloader
        param {*} val_dataloader
        """
        print("Start training......")
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_running_loss = 0
            for idx, sample in enumerate(tqdm(train_dataloader)):
                # start and end position
                input_ids = sample["input_ids"].to(self.device)
                attention_mask = sample["attention_mask"].to(self.device)
                start_positions = sample["start_positions"].to(self.device)
                end_positions = sample["end_positions"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_running_loss += loss.item()
            train_loss = train_running_loss / (idx + 1)

            # validation
            self.model.eval()
            val_running_loss = 0
            with torch.no_grad():
                for idx, sample in enumerate(tqdm(val_dataloader)):
                    input_ids = sample["input_ids"].to(self.device)
                    attention_mask = sample["attention_mask"].to(self.device)
                    start_positions = sample["start_positions"].to(self.device)
                    end_positions = sample["end_positions"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions,
                    )
                    val_running_loss += outputs.loss.item()
                val_loss = val_running_loss / (idx + 1)

            print("-" * 30)
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print("-" * 30)

    def test(self, test_dataloader):
        """
        description: This is the method for testing process.
        param {*} self
        param {*} test_dataloader
        """
        print("Start testing......")
        self.model.eval()
        preds, true = [], []
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(test_dataloader)):
                input_ids = sample["input_ids"].to(self.device)
                attention_mask = sample["attention_mask"].to(self.device)
                start_positions = sample["start_positions"]
                end_positions = sample["end_positions"]

                # get predictions of start and end positions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                start_pred = torch.argmax(outputs["start_logits"], dim=1).cpu().detach()
                end_pred = torch.argmax(outputs["end_logits"], dim=1).cpu().detach()
                preds.extend([[int(i), int(j)] for i, j in zip(start_pred, end_pred)])
                true.extend(
                    [[int(i), int(j)] for i, j in zip(start_positions, end_positions)]
                )

        preds = [item for sublist in preds for item in sublist]
        true = [item for sublist in true for item in sublist]
        f1_value = f1_score(true, preds, average="macro")
        print(f"F1 Score: {f1_value}")
        print("Finish testing.")
