
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


class TextCNN(nn.Module):
    def __init__(
        self,
        method,
        device,
        input_dim,
        output_dim,
        n_filters = 100,
        epochs=10,
        lr=1e-5,
        grained="fine" 
    ):
        super(TextCNN, self).__init__()
        self.method = method
        self.device = device
        self.filters_sizes = [3,4,5]
        self.out_features = 20 if grained == "fine" else 2

        self.embedding = nn.Embedding(input_dim, output_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, output_dim)) 
                                    for fs in self.filters_sizes
                                    ])
        
        self.fc = nn.Linear(len(self.filter_sizes) * n_filters, self.out_features)
        self.dropout = nn.Dropout(0.3)
       
        self.loss_fn = torch.nn.CrossEntropyLoss()  # based on logits directly
        self.lr = lr
        self.epochs = epochs
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def forward(self, x):
        embedded = self.embedding(x)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)

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
                train_loss = self.loss_fn(train_output[0], train_label).item()
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
                test_logits = test_output.logits

                test_pred.append(
                    torch.argmax(test_output[0], dim=-1)
                )  # from logits argmax
                test_labels.append(test_label)
                test_losses.append(test_loss)
            print(f"Finish testing. Test loss: {torch.stack(test_losses).mean()}")

        return test_pred, test_labels
