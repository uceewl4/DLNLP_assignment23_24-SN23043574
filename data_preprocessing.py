"""
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2024-02-08 15:42:41
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2024-02-08 15:43:14
FilePath: /DLNLP_assignment23_24-SN23043574/data_preprocessing.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEd
"""

import pprint
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
    df_train, df_left = train_test_split(df, test_size=0.4, random_state=42)
    df_train.index = [i for i in range(len(df_train))]
    df_val, df_test = train_test_split(df_left, test_size=0.5, random_state=42)
    df_val.index = [i for i in range(len(df_val))]
    df_test.index = [i for i in range(len(df_test))]
    if not os.path.exists("Datasets/preprocessed/sentiment_analysis"):
        os.makedirs("Datasets/preprocessed/sentiment_analysis")
    df.to_csv("Datasets/preprocessed/sentiment_analysis/all.csv", index=False)
    df_train.to_csv("Datasets/preprocessed/sentiment_analysis/train.csv", index=False)
    df_val.to_csv("Datasets/preprocessed/sentiment_analysis/val.csv", index=False)
    df_test.to_csv("Datasets/preprocessed/sentiment_analysis/test.csv", index=False)


# sentiment_preprocessing()
