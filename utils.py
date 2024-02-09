# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2023/12/16 22:44:21
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for all utils function like visualization, data loading, model loading, etc.
"""

# here put the import lib
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    silhouette_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    auc,
)
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import os
from transformers import AutoTokenizer, DataCollatorWithPadding, LongformerTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset

from A.sentiment_analysis.Pretrained import Pretrained


"""
description: This function is used for loading data from preprocessed dataset into model input.
param {*} task: task Aor B
param {*} path: preprocessed dataset path
param {*} method: selected model for experiment
param {*} batch_size: batch size of NNs
return {*}: loaded model input 
"""


def load_data(batch_size=8, type="train"):
    # max length of each:
    # sentiment analysis: train 637  val: 593 test: 907
    folder = "Datasets/preprocessed/sentiment_analysis"
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    df_all = pd.read_csv(os.path.join(folder, f"all.csv"))
    label_map = {
        i: index
        for index, i in enumerate(sorted(list(set(df_all["sentiment"].to_list()))))
    }
    max_sentence_length = max(len(sentence) for sentence in df_all["tweet content"])
    df = pd.read_csv(os.path.join(folder, f"{type}.csv"))
    input_ids = [
        tokenizer.encode(
            sentence,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_sentence_length,
        )
        for sentence in df["tweet content"]
    ]
    print(np.array(input_ids).shape)  # (44802,637)
    attention_masks = [[1] * len(input_id) for input_id in input_ids]
    labels = df["sentiment"].map(lambda x: label_map[x]).to_list()
    dataset = TensorDataset(  # 44802
        torch.tensor(input_ids),
        torch.tensor(attention_masks),
        torch.tensor(labels),
    )
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    print(next(iter(dataloader)))
    return dataloader


"""
description: This function is used for loading selected model.
param {*} task: task A or B
param {*} method: selected model
param {*} multilabel: whether configuring multilabels setting (can only be used with MLP/CNN in task B)
param {*} lr: learning rate for adjustment and tuning
return {*}: constructed model
"""


def load_model(device, method, lr=0.001, epochs=10):
    if method == "Pretrained":
        model = Pretrained(device, device, lr=lr, epochs=epochs)
    # elif method == "MoE":
    #     model = MoE(method, lr=lr, batch_size=batch_size, epochs=epochs)
    # elif method in ["ResNet50", "InceptionV3", "MobileNetV2", "NASNetMobile", "VGG19"]:
    #     model = Pretrained(method, lr=lr, epochs=epochs, batch_size=batch_size)
    # elif method == "Multimodal":
    #     model = Multimodal(method, lr=lr, epochs=epochs, batch_size=batch_size)
    # elif method == "BaseGAN":
    #     model = BaseGAN(method, lr=lr, epochs=epochs, batch_size=batch_size)
    # elif method == "PencilGAN":
    #     model = PencilGAN(method, lr=lr, epochs=epochs, batch_size=batch_size)
    # elif method == "ConGAN":
    #     model = ConGAN(method, lr=lr, epochs=epochs, batch_size=batch_size)
    # elif method == "AdvCNN":
    #     model = AdvCNN(method, lr=lr, epochs=epochs, batch_size=batch_size)
    # elif method == "ViT":
    #     model = ViT(method, lr=lr, epochs=epochs, batch_size=batch_size)
    # elif method == "AutoEncoder":
    #     model = AutoEncoder(method, lr=lr, epochs=epochs, batch_size=batch_size)

    return model


"""
description: This function is used for visualizing confusion matrix.
param {*} task: task A or B
param {*} method: selected model
param {*} ytrain: train ground truth
param {*} yval: validation ground truth
param {*} ytest: test ground truth
param {*} train_pred: train prediction
param {*} val_pred: validation prediction
param {*} test_pred: test prediction
"""


def visual4cm(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
    # confusion matrix
    cms = {
        "train": confusion_matrix(ytrain, train_pred),
        "val": confusion_matrix(yval, val_pred),
        "test": confusion_matrix(ytest, test_pred),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey="row")
    for index, mode in enumerate(["train", "val", "test"]):
        disp = ConfusionMatrixDisplay(
            cms[mode], display_labels=sorted(list(set(ytrain)))
        )
        # print(sorted(list(set(ytrain))))
        # print(cms[mode])
        disp.plot(ax=axes[index])
        # disp.plot()
        disp.ax_.set_title(mode)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel("")
        if index != 0:
            disp.ax_.set_ylabel("")

    fig.text(0.45, 0.05, "Predicted label", ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)

    if not os.path.exists("outputs/image_classification/confusion_matrix/"):
        os.makedirs("outputs/image_classification/confusion_matrix/")
    fig.savefig(f"outputs/image_classification/confusion_matrix/{method}.png")
    plt.close()


"""
description: This function is used for calculating metrics performance including accuracy, precision, recall, f1-score.
param {*} task: task A or B
param {*} y: ground truth
param {*} pred: predicted labels
"""


def get_metrics(task, y, pred):
    result = {
        "acc": round(
            accuracy_score(np.array(y).astype(int), pred.astype(int)) * 100, 4
        ),
        "pre": round(
            precision_score(np.array(y).astype(int), pred.astype(int), average="macro")
            * 100,
            4,
        ),
        "rec": round(
            recall_score(np.array(y).astype(int), pred.astype(int), average="macro")
            * 100,
            4,
        ),
        "f1": round(
            f1_score(np.array(y).astype(int), pred.astype(int), average="macro") * 100,
            4,
        ),
    }
    return result


"""
description: This function is used for visualizing dataset label distribution.
param {*} task: task A or B
param {*} data: npz data
"""


def visual4label(task, data):
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(6, 3), subplot_kw=dict(aspect="equal"), dpi=600
    )

    for index, mode in enumerate(["train", "val", "test"]):
        pie_data = [
            np.count_nonzero(data[f"{mode}_labels"].flatten() == i)
            for i in range(len(set(data[f"{mode}_labels"].flatten().tolist())))
        ]
        labels = [
            f"label {i}"
            for i in sorted(list(set(data[f"{mode}_labels"].flatten().tolist())))
        ]
        wedges, texts, autotexts = ax[index].pie(
            pie_data,
            autopct=lambda pct: f"{pct:.2f}%\n({int(np.round(pct/100.*np.sum(pie_data))):d})",
            textprops=dict(color="w"),
        )
        if index == 2:
            ax[index].legend(
                wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
            )
        size = 6 if task == "A" else 3
        plt.setp(autotexts, size=size, weight="bold")
        ax[index].set_title(mode)
    plt.tight_layout()

    if not os.path.exists("outputs/images/"):
        os.makedirs("outputs/images/")
    fig.savefig(f"outputs/images/label_distribution_task{task}.png")
    plt.close()


def visual4loss(method, type, loss, acc):
    plt.figure()
    plt.title(f"Loss for epochs of {method}")
    plt.plot(
        range(len(loss)),
        loss,
        color="pink",
        linestyle="dashed",
        marker="o",
        markerfacecolor="grey",
        markersize=10,
    )
    plt.tight_layout()

    if not os.path.exists("outputs/image_classification/metric_lines"):
        os.makedirs("outputs/image_classification/metric_lines")
    plt.savefig(f"outputs/image_classification/metric_lines/{method}_{type}_loss.png")
    plt.close()

    plt.figure()
    plt.title(f"Accuracy for epochs of {method}")
    plt.plot(
        range(len(acc)),
        acc,
        color="pink",
        linestyle="dashed",
        marker="o",
        markerfacecolor="grey",
        markersize=10,
    )
    plt.tight_layout()
    plt.savefig(f"outputs/image_classification/metric_lines/{method}_{type}_acc.png")
    plt.close()
