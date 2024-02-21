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
from sklearn.metrics import accuracy_score


class TextCNN(nn.Module):
    def __init__(
        self,
        method,
        device,
        input_dim,
        output_dim,
        n_filters=100,
        epochs=10,
        lr=1e-5,
        grained="fine",
        multilabel=False,
    ):
        super(TextCNN, self).__init__()
        self.method = method
        self.device = device
        self.multilabel = multilabel
        self.filters_sizes = [3, 4, 5]
        self.num_class = 20 if grained == "fine" else 2
        self.embedding = nn.Embedding(input_dim, output_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1, out_channels=n_filters, kernel_size=(fs, output_dim)
                )
                for fs in self.filters_sizes
            ]
        )

        self.fc = nn.Linear(len(self.filters_sizes) * n_filters, self.num_class)
        self.dropout = nn.Dropout(0.3)

        self.loss_fn = torch.nn.CrossEntropyLoss()  # based on logits directly
        self.lr = lr
        self.epochs = epochs

    def forward(self, x):
        embedded = self.embedding(x)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

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
                train_label = train_batch[1].to(self.device)  # 64  8

                self.optimizer.zero_grad()  # Zero the gradients
                train_output = model(train_input_ids)  # 8,8
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
                    for index, i in enumerate(train_batch[1].tolist()):
                        if i in top_indices[index]:
                            train_pred.append(i)
                        else:
                            train_pred.append(torch.argmax(train_output[index]).item())
                train_labels += train_batch[1].tolist()

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
            # self.model.to(device)
            val_pred, val_labels = [], []
            progress_bar_val = tqdm(range(len(val_dataloader)))

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
                    for index, i in enumerate(val_batch[1].tolist()):
                        if i in top_indices[index]:
                            val_pred.append(i)
                        else:
                            val_pred.append(torch.argmax(val_output[index]).item())
                val_labels += val_batch[1].tolist()
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
                test_loss = self.loss_fn(test_output, test_label).item()
                progress_bar_test.update(1)

                if self.multilabel == False:
                    test_pred += torch.argmax(
                        test_output, dim=-1
                    ).tolist()  # from logits argmax
                elif self.multilabel == True:
                    top_values, top_indices = torch.topk(test_output, 3, dim=1)
                    for index, i in enumerate(test_batch[1].tolist()):
                        if i in top_indices[index]:
                            test_pred.append(i)
                        else:
                            test_pred.append(torch.argmax(test_output[index]).item())
                test_labels += test_batch[1].tolist()
                test_losses.append(test_loss)
            test_pred = np.array(test_pred)
            print(f"Finish testing. Test loss: {np.array(test_losses).mean()}")

        return test_pred, test_labels
