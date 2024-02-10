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
from A.sentiment_analysis.Ensemble import Ensemble as SA_Ensemble
from A.sentiment_analysis.LSTM import LSTM as SA_LSTM
from A.sentiment_analysis.Pretrained import Pretrained as SA_Pretrained
from A.sentiment_analysis.RNN import RNN as SA_RNN
from A.intent_recognition.Ensemble import Ensemble as IR_Ensemble
from A.intent_recognition.LSTM import LSTM as IR_LSTM
from A.intent_recognition.Pretrained import Pretrained as IR_Pretrained
from A.intent_recognition.RNN import RNN as IR_RNN
from A.emotion_classification.Ensemble import Ensemble as EC_Ensemble
from A.emotion_classification.LSTM import LSTM as EC_LSTM
from A.emotion_classification.Pretrained import Pretrained as EC_Pretrained
from A.emotion_classification.RNN import RNN as EC_RNN
from A.fake_news.Ensemble import Ensemble as FN_Ensemble
from A.fake_news.LSTM import LSTM as FN_LSTM
from A.fake_news.Pretrained import Pretrained as FN_Pretrained
from A.fake_news.RNN import RNN as FN_RNN
from A.spam_detection.Ensemble import Ensemble as SD_Ensemble
from A.spam_detection.LSTM import LSTM as SD_LSTM
from A.spam_detection.Pretrained import Pretrained as SD_Pretrained
from A.spam_detection.RNN import RNN as SD_RNN
from torchview import draw_graph
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


import spacy
from keras.utils import pad_sequences

"""
description: This function is used for loading data from preprocessed dataset into model input.
param {*} task: task Aor B
param {*} path: preprocessed dataset path
param {*} method: selected model for experiment
param {*} batch_size: batch size of NNs
return {*}: loaded model input 
"""


def load_data(task, method, batch_size=8, type="train", grained="course"):
    # max length of each:
    # sentiment analysis: train 637  val: 593 test: 907
    folder = f"Datasets/preprocessed/{task}"
    df_all = pd.read_csv(os.path.join(folder, f"all.csv"))
    df = pd.read_csv(os.path.join(folder, f"{type}.csv"))

    if method in ["Pretrained", "RNN", "Ensemble"]:
        # word
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        vocab = tokenizer.get_vocab()
        max_sentence_length = max(len(sentence) for sentence in df_all.iloc[:, 1])
        input_ids = [
            tokenizer.encode(
                sentence,
                add_special_tokens=True,
                padding="max_length",
                max_length=max_sentence_length,
            )
            for sentence in df.iloc[:, 1]
        ]
        print(np.array(input_ids).shape)  # (44802,637)
    elif method in ["LSTM"]:
        spacy.cli.download("en_core_web_md")
        nlp_md = spacy.load("en_core_web_md")
        vocab = nlp_md.vocab
        sentences = nlp_md.pipe(
            df_all.iloc[:, 1], disable=["parser", "ner"], batch_size=100, n_process=3
        )
        max_sentence_length = max(len(sentence) for sentence in df_all.iloc[:, 1])
        all_tokens = []
        for sentence in sentences:
            tokens = []
            for token in sentence:
                if token.is_alpha and not token.is_stop:
                    tokens.append(token.lemma_.lower())
            all_tokens.append(tokens)
        vectors = nlp_md.vocab.vectors
        input_ids = [
            [vectors.find(key=word) for word in tokens if vectors.find(key=word) > -1]
            for tokens in all_tokens
        ]
        input_ids = pad_sequences(
            input_ids, maxlen=max_sentence_length, padding="post", truncating="post"
        )
        embeddings = nlp_md.vocab.vectors.data

    # labels
    label2numeric = {
        i: index
        for index, i in enumerate(sorted(list(set(df_all.iloc[:, 0].to_list()))))
    }
    numeric2label = {
        index: i
        for index, i in enumerate(sorted(list(set(df_all.iloc[:, 0].to_list()))))
    }
    labels = df.iloc[:, 0].map(lambda x: label2numeric[x]).to_list()  # course grain
    if grained == "coarse":
        if task == "intent_recognition":
            labels = [
                (
                    0
                    if numeric2label[i]
                    in [
                        "Complain",
                        "Praise",
                        "Apologize",
                        "Thank",
                        "Criticize",
                        "Care",
                        "Agree",
                        "Taunt",
                        "Flaunt",
                        "Oppose",
                        "Joke",
                    ]
                    else 1
                )
                for i in labels
            ]
            # emotion, 0
            # goal, 1
        elif task == "fake_news":
            labels = [0 if numeric2label[i] != "true" else 1 for i in labels]

    if method == "Pretrained":
        attention_masks = [[1] * len(input_id) for input_id in input_ids]
        dataset = TensorDataset(  # 44802
            torch.tensor(input_ids),
            torch.tensor(attention_masks),
            torch.tensor(labels),
        )
    elif method == "RNN":
        dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(labels))  # 44802
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    print(next(iter(dataloader)))

    if method == "Pretrained":
        return dataloader
    elif method in ["RNN", "Ensemble"]:
        return dataloader, vocab
    elif method == "LSTM":
        return dataloader, vocab, embeddings


# load_data("train")
"""
description: This function is used for loading selected model.
param {*} task: task A or B
param {*} method: selected model
param {*} multilabel: whether configuring multilabels setting (can only be used with MLP/CNN in task B)
param {*} lr: learning rate for adjustment and tuning
return {*}: constructed model
"""


def load_model(
    task,
    device,
    method,
    embeddings=None,
    vocab=None,
    output_dim=64,
    bidirectional=False,
    lr=0.001,
    epochs=10,
    alpha=0.5,
    grained="fine",
):  
    if task == "sentiment_analysis":
        if method == "Pretrained":
            model = SA_Pretrained(
                method=method, device=device, lr=lr, epochs=epochs, grained=grained
            )
        elif method == "RNN":
            model = SA_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "LSTM":
            model = SA_LSTM(
                method,
                device,
                embeddings,
                output_dim,
                bidrectional=False,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "Ensemble":
            model = SA_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                alpha=alpha,
                grained=grained,
            )
    elif task == "intent_recognition":
        if method == "Pretrained":
            model = IR_Pretrained(
                method=method, device=device, lr=lr, epochs=epochs, grained=grained
            )
        elif method == "RNN":
            model = IR_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "LSTM":
            model = IR_LSTM(
                method,
                device,
                embeddings,
                output_dim,
                bidrectional=False,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "Ensemble":
            model = IR_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                alpha=alpha,
                grained=grained,
            )
    elif task == "emotion_classification":
        if method == "Pretrained":
            model = EC_Pretrained(
                method=method, device=device, lr=lr, epochs=epochs, grained=grained
            )
        elif method == "RNN":
            model = EC_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "LSTM":
            model = EC_LSTM(
                method,
                device,
                embeddings,
                output_dim,
                bidrectional=False,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "Ensemble":
            model = EC_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                alpha=alpha,
                grained=grained,
            )
    elif task == "fake_news":
        if method == "Pretrained":
            model = FN_Pretrained(
                method=method, device=device, lr=lr, epochs=epochs, grained=grained
            )
        elif method == "RNN":
            model = FN_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "LSTM":
            model = FN_LSTM(
                method,
                device,
                embeddings,
                output_dim,
                bidrectional=False,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "Ensemble":
            model = FN_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                alpha=alpha,
                grained=grained,
            )
    elif task == "spam_detection":
        if method == "Pretrained":
            model = SD_Pretrained(
                method=method, device=device, lr=lr, epochs=epochs, grained=grained
            )
        elif method == "RNN":
            model = SD_RNN(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "LSTM":
            model = SD_LSTM(
                method,
                device,
                embeddings,
                output_dim,
                bidrectional=False,
                epochs=10,
                lr=1e-5,
                grained=grained,
            )
        elif method == "Ensemble":
            model = SD_Ensemble(
                method=method,
                device=device,
                input_dim=len(vocab),
                output_dim=output_dim,
                bidrectional=bidirectional,
                epochs=10,
                lr=1e-5,
                alpha=alpha,
                grained=grained,
            )
    
   

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

    if not os.path.exists(f"Outputs/{task}/confusion_matrix/"):
        os.makedirs(f"Outputs/{task}/confusion_matrix/")
    fig.savefig(f"Outputs/{task}/confusion_matrix/{method}.png")
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


def visual4model(task, method, model, input_data):
    model_graph = draw_graph(model, input_data=input_data)
    plt.figure(figsize=(8, 15))
    model_graph.visual_graph
    if not os.path.exists(f"Outputs/{task}/models/"):
        os.makedirs(f"Outputs/{task}/models/")
    plt.savefig(f"Outputs/{task}/models/{method}.png")


def visual4loss(task, method, train_loss, train_acc, val_loss, val_acc):
    plt.figure()
    plt.title(f"Loss for epochs of {method}")
    plt.plot(
        range(len(train_loss)),
        train_loss,
        color="pink",
        linestyle="dashed",
        marker="o",
        markerfacecolor="grey",
        markersize=10,
    )
    plt.plot(
        range(len(val_loss)),
        val_loss,
        color="yellow",
        linestyle="dashed",
        marker="*",
        markerfacecolor="orange",
        markersize=10,
    )
    plt.tight_layout()

    if not os.path.exists(f"Outputs/{task}/metric_lines"):
        os.makedirs(f"Outputs/{task}/metric_lines")
    plt.savefig(f"Outputs/{task}/metric_lines/{method}_loss.png")
    plt.close()

    plt.figure()
    plt.title(f"Accuracy for epochs of {method}")
    plt.plot(
        range(len(train_acc)),
        train_acc,
        color="pink",
        linestyle="dashed",
        marker="o",
        markerfacecolor="grey",
        markersize=10,
    )
    plt.plot(
        range(len(val_acc)),
        val_acc,
        color="blue",
        linestyle="dashed",
        marker="*",
        markerfacecolor="orange",
        markersize=10,
    )
    plt.tight_layout()
    plt.savefig(f"Outputs/{task}/metric_lines/{method}_acc.png")
    plt.close()


def visual4auc(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test):
    """
    This function is used for visualizing AUROC curve.
    :param label_dict: predict labels of various methods
    :param class_dict: true labels of various methods
    :param name: name of output picture (name of the method)
    """
    dictionary = {
        "train": (ytrain, pred_train),
        "val": (yval, pred_val),
        "test": (ytest, pred_test),
    }
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for index, (key, value) in enumerate(dictionary.items()):
        fpr, tpr, thre = roc_curve(
            value[0], value[1], pos_label=1, drop_intermediate=True
        )
        plt.plot(
            fpr,
            tpr,
            lw=1,
            label="{}(AUC={:.3f})".format(key, auc(fpr, tpr)),
            color=mcolors.TABLEAU_COLORS[colors[index]],
        )  # draw each one
    plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title("ROC Curve", fontsize=10)
    plt.legend(loc="lower right", fontsize=5)
    if not os.path.exists(f"Outputs/{task}/metric_lines"):
        os.makedirs(f"Outputs/{task}/metric_lines")
    plt.savefig(f"Outputs/{task}/metric_lines/{method}_auroc.png")
    plt.show()
