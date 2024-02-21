"""
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2024-02-08 15:42:41
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2024-02-09 20:19:58
FilePath: /DLNLP_assignment23_24-SN23043574/data_preprocessing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

import json
from sklearn.utils import shuffle
import pprint
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import os
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader, TensorDataset


def sentence_clean(sentence):
    # remove backslash-apostrophe
    sentence = re.sub("'", "", sentence)  # replace
    # remove everything except alphabets
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    # remove whitespaces
    sentence = " ".join(sentence.split())
    # remove white space in start and end
    sentence = re.sub(r"^\s+", "", sentence).lstrip()
    # remove special characters
    sentence = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", sentence)
    # convert text to lowercase
    sentence = sentence.lower()

    return sentence


# 1200, 400, 400
def sentiment_preprocessing():
    df_1 = pd.read_csv(
        "Datasets/raw/sentiment_analysis/twitter_training.csv",
        names=["tweet id", "entity", "sentiment", "tweet content"],
    )
    df_2 = pd.read_csv(
        "Datasets/raw/sentiment_analysis/twitter_validation.csv",
        names=["tweet id", "entity", "sentiment", "tweet content"],
    )
    df = (
        pd.concat((df_1, df_2), ignore_index=True)
        .drop(columns=["tweet id", "entity"])
        .dropna()
    )
    # print(df)
    fig = plt.figure()
    df["sentiment"].value_counts().plot.bar(
        rot=0, figsize=(8, 5), grid=True, align="center"
    )
    fig.savefig("Outputs/data_visualization/sentiment.png")

    print(df)
    df["tweet content"] = (
        df["tweet content"].astype(str).apply(lambda x: sentence_clean(x))
    )
    df = df[df["tweet content"] != ""].dropna(subset=["tweet content"])

    # 1200/4 300,300,300,300
    # 400/4 100 100 100 100
    train_index, val_index, test_index = [], [], []
    df_all = df.copy()
    for i in list(set(df["sentiment"].to_list())):
        index = list(df[df["sentiment"] == i].index)
        drop_index = random.sample(index, 300)
        train_index += drop_index
        df = df.drop(drop_index, axis=0)  # delete the sample case

        index = list(df[df["sentiment"] == i].index)
        drop_index = random.sample(index, 100)
        val_index += drop_index
        df = df.drop(drop_index, axis=0)  # delete the sample case

        index = list(df[df["sentiment"] == i].index)
        drop_index = random.sample(index, 100)
        test_index += drop_index
        df = df.drop(drop_index, axis=0)

    all_index = train_index + val_index + test_index
    df_train = shuffle(df_all.loc[train_index, :])
    df_val = shuffle(df_all.loc[val_index, :])
    df_test = shuffle(df_all.loc[test_index, :])
    df_all = shuffle(df_all.loc[all_index, :])

    if not os.path.exists("Datasets/preprocessed/sentiment_analysis"):
        os.makedirs("Datasets/preprocessed/sentiment_analysis")
    df_all.to_csv("Datasets/preprocessed/sentiment_analysis/all.csv", index=False)
    df_train.to_csv("Datasets/preprocessed/sentiment_analysis/train.csv", index=False)
    df_val.to_csv("Datasets/preprocessed/sentiment_analysis/val.csv", index=False)
    df_test.to_csv("Datasets/preprocessed/sentiment_analysis/test.csv", index=False)


# sentiment_preprocessing()


def emotion_preprocessing():
    folder = "Datasets/raw/emotion_classification"
    for index, file in enumerate(os.listdir(folder)):
        data = []
        with open(os.path.join(folder, file)) as f:
            data = json.load(f)
            utterance = []
            emotion = []
            for i in data:
                for j in i:
                    utterance.append(j["utterance"])
                    emotion.append(j["emotion"])
        df = pd.DataFrame(
            {"emotion": emotion, "utterance": utterance},
            columns=["emotion", "utterance"],
        )
        print(df)

        df["utterance"] = df["utterance"].astype(str).apply(lambda x: sentence_clean(x))
        df = df[df["utterance"] != ""].dropna(subset=["utterance"])

        # 1200  150
        # 400   50
        df_tmp = df.copy()
        tmp_index = []
        for i in list(set(df["emotion"].to_list())):
            type = list(df[df["emotion"] == i].index)
            if "train" in file:
                drop_index = random.sample(type, 150)
            else:
                drop_index = type if len(type) <= 50 else random.sample(type, 50)
            tmp_index += drop_index
            df = df.drop(drop_index, axis=0)  # delete the sample case
        df_tmp = df_tmp.loc[tmp_index, :]

        if index == 0:
            df_all = df_tmp
        else:
            df_all = pd.concat((df_all, df_tmp), ignore_index=True).dropna()
        df_tmp = shuffle(df_tmp)
        if not os.path.exists("Datasets/preprocessed/emotion_classification"):
            os.makedirs("Datasets/preprocessed/emotion_classification")
        df_tmp.to_csv(
            f"Datasets/preprocessed/emotion_classification/{file.split('.')[0].split('_')[1]}.csv",
            index=False,
        )
    fig = plt.figure()
    df_all["emotion"].value_counts().plot.bar(
        rot=0, figsize=(8, 5), grid=True, align="center"
    )
    df_all = shuffle(df_all)
    fig.savefig("Outputs/data_visualization/emotion.png")
    df_all.to_csv("Datasets/preprocessed/emotion_classification/all.csv", index=False)


# emotion_preprocessing()


def intent_recognition():
    folder = "Datasets/raw/intent_recognition"
    for index, file in enumerate(os.listdir(folder)):
        df = pd.read_csv(
            os.path.join(folder, file),
            sep="\t",
        )
        print(df)

        df["text"] = df["text"].astype(str).apply(lambda x: sentence_clean(x))
        df = (
            df[df["text"] != ""]
            .dropna(subset=["text"])
            .drop(columns=["season", "episode", "clip"])
        )
        df = pd.DataFrame(df, columns=["label", "text"])
        print(df)

        if index == 0:
            df_all = df
        else:
            df_all = pd.concat((df_all, df), ignore_index=True).dropna()
        if not os.path.exists("Datasets/preprocessed/intent_recognition"):
            os.makedirs("Datasets/preprocessed/intent_recognition")
        df.to_csv(
            f"Datasets/preprocessed/intent_recognition/{file.split('.')[0]}.csv",
            index=False,
        )
    fig = plt.figure()
    df_all["label"].value_counts().plot.bar(
        rot=0, figsize=(20, 5), grid=True, align="center"
    )
    fig.savefig("Outputs/data_visualization/intent.png")
    df_all.to_csv("Datasets/preprocessed/intent_recognition/all.csv", index=False)


# intent_recognition()


def spam_detection():
    df = pd.read_csv(
        "Datasets/raw/spam_detection/spam.csv", encoding="ISO-8859-1"
    ).iloc[:, :2]
    df.columns = ["label", "text"]

    fig = plt.figure()
    df["label"].value_counts().plot.bar(
        rot=0, figsize=(8, 5), grid=True, align="center"
    )
    fig.savefig("Outputs/data_visualization/spam.png")

    # 1200/2 600,600
    # 400/2 200 200
    train_index, val_index, test_index = [], [], []
    df_all = df.copy()
    for i in list(set(df["label"].to_list())):
        index = list(df[df["label"] == i].index)
        drop_index = (
            random.sample(index, 500) if i == "spam" else random.sample(index, 600)
        )
        train_index += drop_index
        df = df.drop(drop_index, axis=0)  # delete the sample case

        index = list(df[df["label"] == i].index)
        drop_index = index if len(index) < 200 else random.sample(index, 200)
        val_index += drop_index
        df = df.drop(drop_index, axis=0)  # delete the sample case

        index = list(df[df["label"] == i].index)
        drop_index = index if len(index) < 200 else random.sample(index, 200)
        test_index += drop_index
        df = df.drop(drop_index, axis=0)

    all_index = train_index + val_index + test_index
    df_train = shuffle(df_all.loc[train_index, :])
    df_val = shuffle(df_all.loc[val_index, :])
    df_test = shuffle(df_all.loc[test_index, :])
    df_all = shuffle(df_all.loc[all_index, :])

    if not os.path.exists("Datasets/preprocessed/spam_detection"):
        os.makedirs("Datasets/preprocessed/spam_detection")
    df_all.to_csv("Datasets/preprocessed/spam_detection/all.csv", index=False)
    df_train.to_csv("Datasets/preprocessed/spam_detection/train.csv", index=False)
    df_val.to_csv("Datasets/preprocessed/spam_detection/val.csv", index=False)
    df_test.to_csv("Datasets/preprocessed/spam_detection/test.csv", index=False)


# spam_detection()


def fake_news():
    folder = "Datasets/raw/fake_news"
    for index, file in enumerate(os.listdir(folder)):
        df = pd.read_csv(
            os.path.join(folder, file),
            sep="\t",
        ).iloc[:, 1:3]
        df.columns = ["label", "text"]
        print(df)

        df["text"] = df["text"].astype(str).apply(lambda x: sentence_clean(x))
        df = df[df["text"] != ""].dropna(subset=["text"])

        # 1200  200/200/
        # 420   70
        df_tmp = df.copy()
        tmp_index = []
        for i in list(set(df["label"].to_list())):
            type = list(df[df["label"] == i].index)
            if "train" in file:
                drop_index = random.sample(type, 200)
            else:
                drop_index = type if len(type) <= 70 else random.sample(type, 70)
            tmp_index += drop_index
            df = df.drop(drop_index, axis=0)  # delete the sample case
        df_tmp = df_tmp.loc[tmp_index, :]

        if index == 0:
            df_all = df_tmp
        else:
            df_all = pd.concat((df_all, df_tmp), ignore_index=True).dropna()
        if not os.path.exists("Datasets/preprocessed/fake_news"):
            os.makedirs("Datasets/preprocessed/fake_news")
        df_tmp = shuffle(df_tmp)
        df_tmp.to_csv(
            f"Datasets/preprocessed/fake_news/{file.split('.')[0]}.csv",
            index=False,
        )
    fig = plt.figure()
    df_all["label"].value_counts().plot.bar(
        rot=0, figsize=(20, 5), grid=True, align="center"
    )
    fig.savefig("Outputs/data_visualization/news.png")
    df_all = shuffle(df_all)
    df_all.to_csv("Datasets/preprocessed/fake_news/all.csv", index=False)


# fake_news()
