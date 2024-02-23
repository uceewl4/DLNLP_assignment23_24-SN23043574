# -*- encoding: utf-8 -*-
"""
@File    :   data_preprocessing.py
@Time    :   2024/02/23 15:36:52
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0141: Deep Learning for Natural Language Processing
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for data preprocessing from raw datasets of all tasks.
"""

# here put the import lib

import json
from sklearn.utils import shuffle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os


def sentence_clean(sentence):
    """
    description: This method is used for cleaning the sentences.
    Notice that this method refers to DLNLP lab 1.
    param {*} sentence: sentences to be cleaned
    return {*}: new sentence after cleaning
    """
    sentence = re.sub("'", "", sentence)  # backslash-apostrophe
    sentence = re.sub("[^a-zA-Z]", " ", sentence)  # everything except alphabets
    sentence = " ".join(sentence.split())  # whitespaces
    sentence = re.sub(r"^\s+", "", sentence).lstrip()  # white space in start and end
    sentence = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", sentence)  # special characters
    sentence = sentence.lower()  # lowercase

    return sentence


def sentiment_preprocessing():
    """
    description: This method is used for getting preprocessed data from raw dataset
    for sentiment preprocessing, which splits the dataset into train/validation/testing
    of 1200/400/400.
    """
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

    # label distribution
    fig = plt.figure()
    df["sentiment"].value_counts().plot.bar(
        rot=0, figsize=(8, 5), grid=True, align="center"
    )
    fig.savefig("Outputs/data_visualization/sentiment.png")

    df["tweet content"] = (
        df["tweet content"].astype(str).apply(lambda x: sentence_clean(x))
    )
    df = df[df["tweet content"] != ""].dropna(subset=["tweet content"])

    # dataset split
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
        df = df.drop(drop_index, axis=0)

        index = list(df[df["sentiment"] == i].index)
        drop_index = random.sample(index, 100)
        test_index += drop_index
        df = df.drop(drop_index, axis=0)

    all_index = train_index + val_index + test_index
    df_train = shuffle(df_all.loc[train_index, :])
    df_val = shuffle(df_all.loc[val_index, :])
    df_test = shuffle(df_all.loc[test_index, :])
    df_all = shuffle(df_all.loc[all_index, :])

    # load into preprocessed files
    if not os.path.exists("Datasets/preprocessed/sentiment_analysis"):
        os.makedirs("Datasets/preprocessed/sentiment_analysis")
    df_all.to_csv("Datasets/preprocessed/sentiment_analysis/all.csv", index=False)
    df_train.to_csv("Datasets/preprocessed/sentiment_analysis/train.csv", index=False)
    df_val.to_csv("Datasets/preprocessed/sentiment_analysis/val.csv", index=False)
    df_test.to_csv("Datasets/preprocessed/sentiment_analysis/test.csv", index=False)


def emotion_preprocessing():
    """
    description: This method is used for getting preprocessed data from raw dataset
    for emotion classification, which splits the dataset into train/validation/testing
    of 1200/400/400.
    """
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
        df["utterance"] = df["utterance"].astype(str).apply(lambda x: sentence_clean(x))
        df = df[df["utterance"] != ""].dropna(subset=["utterance"])

        # dataset split
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

    # label distribution
    fig = plt.figure()
    df_all["emotion"].value_counts().plot.bar(
        rot=0, figsize=(8, 5), grid=True, align="center"
    )

    df_all = shuffle(df_all)
    fig.savefig("Outputs/data_visualization/emotion.png")
    df_all.to_csv("Datasets/preprocessed/emotion_classification/all.csv", index=False)


def intent_recognition():
    """
    description: This method is used for getting preprocessed data from raw dataset
    for intent recognition, which splits the dataset into train/validation/testing.
    """
    folder = "Datasets/raw/intent_recognition"
    for index, file in enumerate(os.listdir(folder)):
        df = pd.read_csv(
            os.path.join(folder, file),
            sep="\t",
        )
        df["text"] = df["text"].astype(str).apply(lambda x: sentence_clean(x))
        df = (
            df[df["text"] != ""]
            .dropna(subset=["text"])
            .drop(columns=["season", "episode", "clip"])
        )
        df = pd.DataFrame(df, columns=["label", "text"])

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

    # label distribution
    fig = plt.figure()
    df_all["label"].value_counts().plot.bar(
        rot=0, figsize=(20, 5), grid=True, align="center"
    )
    fig.savefig("Outputs/data_visualization/intent.png")
    df_all.to_csv("Datasets/preprocessed/intent_recognition/all.csv", index=False)


def spam_detection():
    """
    description: This method is used for getting preprocessed data from raw dataset
    for spam detection, which splits the dataset into train/validation/testing
    of 1200/400/400.
    """
    df = pd.read_csv(
        "Datasets/raw/spam_detection/spam.csv", encoding="ISO-8859-1"
    ).iloc[:, :2]
    df.columns = ["label", "text"]

    fig = plt.figure()
    df["label"].value_counts().plot.bar(
        rot=0, figsize=(8, 5), grid=True, align="center"
    )
    fig.savefig("Outputs/data_visualization/spam.png")

    # dataset split
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
        df = df.drop(drop_index, axis=0)

        index = list(df[df["label"] == i].index)
        drop_index = index if len(index) < 200 else random.sample(index, 200)
        test_index += drop_index
        df = df.drop(drop_index, axis=0)

    all_index = train_index + val_index + test_index
    df_train = shuffle(df_all.loc[train_index, :])
    df_val = shuffle(df_all.loc[val_index, :])
    df_test = shuffle(df_all.loc[test_index, :])
    df_all = shuffle(df_all.loc[all_index, :])

    # load into files
    if not os.path.exists("Datasets/preprocessed/spam_detection"):
        os.makedirs("Datasets/preprocessed/spam_detection")
    df_all.to_csv("Datasets/preprocessed/spam_detection/all.csv", index=False)
    df_train.to_csv("Datasets/preprocessed/spam_detection/train.csv", index=False)
    df_val.to_csv("Datasets/preprocessed/spam_detection/val.csv", index=False)
    df_test.to_csv("Datasets/preprocessed/spam_detection/test.csv", index=False)


def fake_news():
    """
    description: This method is used for getting preprocessed data from raw dataset
    for fake news detection, which splits the dataset into train/validation/testing
    of 1200/400/400.
    """
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

    # label distribution
    fig = plt.figure()
    df_all["label"].value_counts().plot.bar(
        rot=0, figsize=(20, 5), grid=True, align="center"
    )
    fig.savefig("Outputs/data_visualization/news.png")
    df_all = shuffle(df_all)
    df_all.to_csv("Datasets/preprocessed/fake_news/all.csv", index=False)


def data_preprocess():
    """
    description: This method includes all data preprocessing methods involved.
    """
    sentiment_preprocessing()
    emotion_preprocessing()
    intent_recognition()
    spam_detection()
    fake_news()
